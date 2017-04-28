
from darch.base import *
from darch.modules import *
from darch.initializers import *
import darch.datasets as ds
import darch.evaluators as ev
import darch.searchers as srch
import search_spaces as srch_sp
from pprint import pprint
import pickle
import dill
import os
import sys

# search space for searching over hyperparameters with two modes: 
# a lighter one and a more extensive one.
def get_hyperparam_search_space(hps_type):
    # 2 * 8 * 2 * 4 = 128 for light 
    if hps_type == "light":
        lr_inits = list( n  p.logspace(-2, -7, num=8) )
        rate_mults = [0.1, 0.5]
        rate_pats = [4, 8, 16, 32]
    # 2 * 32 * 8 * 8 = 4096 for heavy
    elif hps_type == "heavy":
        lr_inits = list( np.logspace(-2, -7, num=32) )
        rate_mults = list( np.logspace(-2, np.log10(0.9), num=8) )
        rate_pats = range(4, 33, 4)      
    else:
        raise ValueError

    return UserHyperparams(['optimizer_type',
                            'learning_rate_init',
                            'rate_mult',
                            'rate_patience', 
                            'stop_patience', 
                            'learning_rate_min' ],
                            [['adam', 'sgd_mom'], 
                            lr_inits, 
                            rate_mults,
                            rate_pats, 
                            [64], 
                            [1e-9] ])

class CustomEvaluator:
    """Custom evaluator whose performance depends on the values of certain
    hyperparameters specified in the hyperparameter module. Hyperparameters that 
    we do not expect to set this way, will take default values.
    """

    def __init__(self, train_dataset, val_dataset, test_dataset, in_d, nclasses, 
            max_minutes_per_model, model_path, output_to_terminal, 
            user_hyperparams_scope_name, args):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.in_d = in_d
        self.nclasses = nclasses
        self.max_minutes_per_model = max_minutes_per_model
        self.model_path = model_path
        self.output_to_terminal = output_to_terminal
        self.user_hyperparams_scope_name = user_hyperparams_scope_name
        self.args = args
        
    def eval_model(self, b):
        """Extract parameters from a UserHyperparams module and uses then to 
        udpate the values of certain hyperparameters of the evaluator. This 
        code is still very much based on ClassifierEvaluator.
        """

        ### this part is AWFUL
        # NOTE: I'm breaking encapsulation here for now.
        if self.args['bisect_search_space']:
            # has a ChoiceBisection, followed by Concat [UserHyperparams, b_search]
            b_hp, b_search = b.b.bs
        else:
            # just the concat module
            b_hp, b_search = b.bs
        b_hp.compile(None, None, None)
        hpsc_name = self.user_hyperparams_scope_name
        order = b_hp.scope.s[hpsc_name]['hyperp_names']
        vals = b_hp.scope.s[hpsc_name]['hyperp_vals']
        hps = dict(zip(order, vals))

        evaluator = ev.ClassifierEvaluator(train_dataset=self.train_dataset,
                                        val_dataset=self.val_dataset,
                                        test_dataset=self.test_dataset,
                                        in_d=self.in_d,
                                        nclasses=self.nclasses,
                                        training_epochs_max=int(1e6),
                                        time_minutes_max=self.max_minutes_per_model,
                                        display_step=1,
                                        stop_patience=hps['stop_patience'], ###
                                        rate_patience=hps['rate_patience'], ###
                                        batch_patience=int(1e6),
                                        save_patience=2, 
                                        rate_mult=hps['rate_mult'], ###
                                        optimizer_type=hps['optimizer_type'], ###
                                        learning_rate_init=hps['learning_rate_init'], ###
                                        learning_rate_min=hps['learning_rate_min'], ###
                                        batch_size_init=64,
                                        model_path=self.model_path,
                                        output_to_terminal=self.output_to_terminal)
        return evaluator.eval_model(b_search)

# loads the data.
def load_data(args):
    # information about the cifar-10
    in_d = (32, 32, 3)
    nclasses = 10

    # options for data augmentation
    trans_height = 32
    trans_width = 32
    p_flip = 0.5
    pad_size = 4 

    # load cifar
    (Xtrain, ytrain, Xval, yval, Xtest, ytest) = ds.load_cifar10(
            data_dir='data/cifar10/cifar-10-batches-py/', 
            flatten=False,
            one_hot=True,
            normalize_range=False,
            whiten_pixels=True,
            border_pad_size=pad_size)

    # augment data if the flag is set
    augment_train_fn = ds.get_augment_cifar_data_train(trans_height, trans_width, p_flip)
    augment_eval_fn = ds.get_augment_cifar_data_eval(trans_height, trans_width)
    in_d = (trans_height, trans_width, 3)

    # wrap data into a InMemoryDataset object
    train_dataset = ds.InMemoryDataset(Xtrain, ytrain, True, augment_train_fn)
    val_dataset = ds.InMemoryDataset(Xval, yval, False, augment_eval_fn)
    test_dataset = ds.InMemoryDataset(Xtest, ytest, False, augment_eval_fn)

    return (train_dataset, val_dataset, test_dataset, in_d, nclasses)

# may need some extra information for the models.s
# the different experiments are simply using different time limits.
def get_search_space(args):
    in_d = (32, 32, 3)
    num_classes = 10
    ss = {'tfrefconv' : srch_sp.tfref_convnet_ss0(num_classes),
          'resnet' : srch_sp.resnet_ss0(num_classes),
          'allconv' : srch_sp.allconvnet_cifar10_ss0(num_classes, in_d),
          'allconv2' : srch_sp.allconvnet_cifar10_ss1(in_d),
          'deepconv' : srch_sp.deepconv_ss0(num_classes) }
    b_search = ss[ args['search_space_type'] ]

    # add hyperparameters to make sure that it is working
    b_hp = get_hyperparam_search_space(args['search_over_hyperparams_type'])
    b_search = Concat([b_hp, b_search])
    # if bissecting the space, adds a bisection module.
    if args['bisect_search_space']:
        b_search = ChoiceBisection(b_search)

    return b_search

def save_checkpoint(out_path, args, searcher, b_search, scores, hists):
    with open(out_path, 'wb') as fp:
        randgen_state = np.random.get_state()
        d = {'args' : args, 'searcher' : searcher, 'b_search' : b_search,
            'scores' : scores, 'hists' : hists, 'randgen_state' : randgen_state }
        pickle.dump(d, fp)
    
def load_checkpoint(out_path):
    with open(out_path, 'rb') as fp:
        d = pickle.load(fp)
        return (d['args'], d['searcher'], d['b_search'], 
                d['scores'], d['hists'], d['randgen_state'])

def get_initial_state(args):
    # create the path if it does not exist.
    model_path = os.path.join(args['output_folder'], args['experiment_name'] + ".ckpt")
    out_path = os.path.join(args['output_folder'], args['experiment_name'] + '.pkl')
    args['out_path'] = out_path
    args['model_path'] = model_path
    # checking if folder creation is necessary.
    if not os.path.isdir(args['output_folder']):
        os.makedirs(args['output_folder'])

    # either load the initial state from the checkpoint or create it.
    if os.path.exists(out_path):
        print "Resuming from existing checkpoint for %s." % args['experiment_name']
        (ckp_args, searcher, b_search, scores, hists, randgen_state) = load_checkpoint(out_path)
        np.random.set_state(randgen_state)
        assert frozenset(ckp_args.items()) == frozenset(args.items())
    else:
        print "Model seach started for %s." % args['experiment_name']
        in_d = (32, 32, 3)
        b_search = get_search_space(args)
        if args['searcher_type'] == 'rand':
            searcher = srch.RandomSearcher(b_search, in_d)
        elif args['searcher_type'] == 'mcts' or args['searcher_type'] == 'mcts_bi':
            searcher = srch.MCTSearcher(b_search, in_d, 0.5)
        elif args['searcher_type'] == 'smbo':
            searcher = srch.SMBOLinearSearcher(b_search, in_d, ngram_maxlen=5,
                            thres=3, lamb_ridge=1.0e3)
        else:
            raise ValueError        
        scores = []
        hists = []
        np.random.seed(args['random_seed'])

        # if the first time that I'm looking at experiments for this, 
        # I will plot the args and the search space at the terminal.
        pprint( args, width=1)
        print 
        pprint( b_search.repr_program() , width=40, indent=2)
        print 

    return searcher, b_search, scores, hists

# max_evals is used to limit the total time that the process will take in the
# case where I'm running in a server that has limit on the time per job.
def run_searcher_with_checkpointing(args):
    (searcher, b_search, scores, hists) = get_initial_state(args)

    # load data and instantiate evaluator.
    (train_dataset, val_dataset, test_dataset, in_d, nclasses) = load_data(args)
    evaluator = CustomEvaluator(train_dataset=train_dataset, 
                                val_dataset=val_dataset,
                                test_dataset=test_dataset,
                                in_d=in_d, 
                                nclasses=nclasses,
                                max_minutes_per_model=args['max_minutes_per_model'],
                                model_path=args['model_path'],
                                output_to_terminal=True,
                                args=args,
                                user_hyperparams_scope_name='UserHyperparams-0') 

    # remaining samples given the checkpoint.
    num_samples_rem = min(  
        args['max_evals_per_process_run'], 
        args['num_samples'] - len(scores))

    # run for the samples remaining for this round.
    for _ in xrange(num_samples_rem):
        if args['searcher_type'] == 'rand':
            (new_scores, new_hists) = srch.run_random_searcher(evaluator, searcher,
                    num_models=1, output_to_terminal=True)
        elif args['searcher_type'] == 'mcts' or args['searcher_type'] == 'mcts_bi':
            (new_scores, new_hists) = srch.run_mcts_searcher(evaluator, searcher,
                    num_models=1, output_to_terminal=True)
        elif args['searcher_type'] == 'smbo':
            (new_scores, new_hists) = srch.run_smbo_searcher(evaluator, searcher,
                    nsamples_start=0, nsamples_after=1, nsamples_epoch=50,
                    refit_interval=1, explore_prob=0.25, output_to_terminal=True)
        else:
            raise ValueError 
    
        scores.extend(new_scores)
        hists.extend(new_hists)
        save_checkpoint(args['out_path'], args, searcher, b_search, scores, hists)
    
    if args['num_samples'] == len(hists):
        print "Experiment completed!" 
    print

#    if os.path.isfile(args['model_path']):
#        os.remove(args['model_path'])

# searchers : rand, mcts, mcts_bi, smbo
def run_searcher_comparison_experiment(searcher_type, search_space_type, seed):
    args = {'augment_data' : True,
            'search_space_type' : search_space_type,
            'search_over_hyperparams_type' : 'heavy',
            'bisect_search_space' : True if searcher_type == 'mcts_bi' else False,
            'random_seed' : seed, 
            'output_folder' : 'logs/searcher_comparison',
            'experiment_name' : "%s_%s_%d" % (search_space_type, searcher_type, seed) ,
            'searcher_type' : searcher_type,
            'num_samples' : 64,  # 64
            'max_minutes_per_model' : 30.0, # 60; maybe 30 minutes .more reps, less time.
            'max_evals_per_process_run' : 100 } # this field may change. 12

    run_searcher_with_checkpointing(args)

if __name__ == '__main__':
    experiment_type = sys.argv[1]

    # print experiment_type, search_space_type, searcher_type, seed

    if experiment_type == 'searcher_comparison':
        search_space_type = sys.argv[2]
        searcher_type = sys.argv[3]
        seed = int(sys.argv[4])

        run_searcher_comparison_experiment(search_space_type, searcher_type, seed)
        
    else:
        raise ValueError

