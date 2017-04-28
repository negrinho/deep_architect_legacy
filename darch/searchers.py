
import numpy as np
import scipy.sparse as sp
import sklearn.linear_model as lm
from pprint import pprint
from darch.base import *
import copy

def evaluate_and_print(evaluator, model,
        output_to_terminal, ignore_invalid_models):

    if output_to_terminal:
        pprint( model.repr_model() , width=40, indent=2)
        print 
        
    if ignore_invalid_models:
        try: 
            sc = evaluator.eval_model(model)

        except ValueError:
            if output_to_terminal:
                print "Invalid Model!"
                return None
    else:
        sc = evaluator.eval_model(model)   
    
    return sc

# save the history on the model. will be useful for cached evaluation.
def maybe_register_choice_hist(model, hist, save_hist_in_model):    
        if save_hist_in_model:
            assert not hasattr(model, 'choice_hist')
            model.choice_hist = hist

def walk_hist(b, hist):
    """Makes a sequence of choices specified by hist towards specifying b.
    This function directly changes b.

    """
    for ch_i in hist:
        b.choose(ch_i)

class EnumerationSearcher:
    def __init__(self, b_search, in_d):
        self.b_search = b_search
        self.in_d = in_d

    def _enumerate_models_iter(self, b, choice_hist):

        if( not b.is_specified() ):
            name, vals = b.get_choices()

            # recurse on the enumeration for each of the possible choices.
            for choice_i in xrange(len(vals)):
                bi = copy.deepcopy(b)
                bi.choose(choice_i)
                for (bk, choice_histk) in self._enumerate_models_iter(
                        bi, choice_hist + [choice_i]):
                    yield (bk, choice_histk)
        else:
            yield (b, choice_hist)

    def enumerate_models(self):
        b = copy.deepcopy(self.b_search)

        b.initialize(self.in_d, Scope())
        gen_models = self._enumerate_models_iter(b, [])
        for (bk, choice_histk) in gen_models:
            yield (bk, choice_histk)

def run_enumeration_searcher(evaluator, searcher,
        output_to_terminal=False, ignore_invalid_models=False,
        save_hist_in_model=False):

    scores = []
    hists = []
    for (mdl, h) in searcher.enumerate_models():

        sc = evaluate_and_print(evaluator, model, 
                output_to_terminal, ignore_invalid_models)
        if sc != None:
            scores.append(sc)
            hists.append(h)
    
    return (scores, hists)

class RandomSearcher:
    def __init__(self, b_search, in_d):
        self.b_search = b_search
        self.in_d = in_d

    def sample_models(self, nsamples):
        b = self.b_search

        samples = []
        choice_hists = []
        for _ in xrange(nsamples):
            bk = copy.deepcopy(b)
            bk.initialize(self.in_d, Scope())
            hist = []
            while( not bk.is_specified() ):
                name, vals = bk.get_choices()
                #print(name, vals)
                assert len(vals) > 1
                choice_i = np.random.randint(0, len(vals))
                bk.choose(choice_i)
                hist.append(choice_i)

            # keep the sampled model once specified.
            samples.append(bk)
            choice_hists.append(hist)

        return (samples, choice_hists)

def run_random_searcher(evaluator, searcher, num_models, 
        output_to_terminal=False, ignore_invalid_models=False,
        save_hist_in_model=False):

    srch_choice_hists = []
    srch_scores = []
    mdl_counter = 0

    while mdl_counter < num_models:
        (models, hists) = searcher.sample_models(1)
        mdl = models[0]
        hist = tuple(hists[0])
        maybe_register_choice_hist(mdl, hist, save_hist_in_model)

        sc = evaluate_and_print(evaluator, mdl, 
                output_to_terminal, ignore_invalid_models)
        if sc != None:
            srch_choice_hists.append(hist)
            srch_scores.append(sc)
            mdl_counter += 1

    return (srch_scores, srch_choice_hists)

class SMBOLinearSearcher:
    def __init__(self, b_search, in_d, ngram_maxlen, thres, lamb_ridge=1.0e3):
        self.epoch_i = 0
        self.sample_hist = {}
        self.histories = {}
        self.known_models = []
        self.known_hists = []
        self.known_scores = []

        self.in_d = in_d
        self.b_search = b_search

        # for computing the features for the models
        self.module_ngram_to_id = {}
        self.ngram_maxlen = ngram_maxlen
        self.thres = thres
        self.surr_model = None
        self.lamb_ridge = lamb_ridge

    def _build_feature_maps(self, bs, maxlen, thres):

        ngram_to_count = {}
        for b in bs:
            # filtering out empty modules in the sequence
            bls = [b[0] for b in tuple(b.repr_model()) if b[0] != "Empty"]

            for k in xrange(1, maxlen):
                for i in xrange(len(bls) - k):
                    ngram = tuple(bls[i:i + k])

                    if ngram not in ngram_to_count:
                        ngram_to_count[ngram] = 0
                    ngram_to_count[ngram] += 1

        # keeping only the ngram with counts above the threshold.
        filtered_ngrams = []
        for (ngram, c) in ngram_to_count.iteritems():
            if c >= thres:
                filtered_ngrams.append(ngram)

        self.module_ngram_to_id = dict(
            zip(filtered_ngrams, range(len(filtered_ngrams)) ) )

    def _compute_features(self, model):

        bls = [ b[0] for b in tuple(model.repr_model()) ]

        nfeats_other = 1
        nfeats_ngrams = len(self.module_ngram_to_id)
        nfeats = nfeats_other + nfeats_ngrams
        feats = sp.dok_matrix((1, nfeats), dtype=np.float32)

        # other features
        feats[0, 0] = len(bls)

        # ngrams features
        for k in xrange(1, self.ngram_maxlen):
            for i in xrange(len(bls) - k):
                ngram = tuple(bls[i:i + k])

                if ngram in self.module_ngram_to_id:
                    ngram_i = self.module_ngram_to_id[ngram]
                    feats_i = nfeats_other + ngram_i

                    feats[0, feats_i] += 1.0

        return sp.csr_matrix(feats)

    def predict_score(self, model):
        if self.surr_model == None:
            return 0.0

        feats = self._compute_features(model)
        sc = self.surr_model.predict(feats)[0]

        return sc

    def get_epoch_samples(self, epoch_i):
        if epoch_i not in self.sample_hist:
            raise KeyError("Epoch %d not present in history" % epoch_i)

        return self.sample_hist[epoch_i]

    def sample_new_epoch(self, nsamples):
        """Generates random specified models in the search space and evaluates
        them based on the currect surrogate model.

        The function returns the epoch number, the specified models, and the
        scores output by the surrogate function.

        """

        epoch_i = self.epoch_i
        samples = []
        scores = []
        choice_hists = []

        for _ in xrange(nsamples):
            bk = copy.deepcopy(self.b_search)
            bk.initialize(self.in_d, Scope())
            hist = []

            while( not bk.is_specified() ):
                _, vals = bk.get_choices()
                choice_i = np.random.randint(0, len(vals))
                bk.choose(choice_i)
                hist.append(choice_i)

            sc = self.predict_score(bk)
            samples.append(bk)
            scores.append(sc)
            choice_hists.append( tuple(hist) )

        sorting_inds = np.argsort(scores)[::-1]
        samples = [samples[i] for i in sorting_inds]
        scores = [scores[i] for i in sorting_inds]
        choice_hists = [choice_hists[i] for i in sorting_inds]

        self.sample_hist[epoch_i] = samples
        self.histories[epoch_i] = choice_hists
        self.epoch_i += 1

        return (epoch_i, tuple(samples), tuple(choice_hists), tuple(scores))

    def forget_epoch(self, epoch_i):
        """Removes a given sample epoch from history.

        After removing an epoch, the specific models can no longer be queried
        by the model.

        """

        if epoch_i not in self.sample_hist or epoch_i not in self.histories:
            raise KeyError("Epoch %d not present in history" % epoch_i)

        self.sample_hist.pop(epoch_i)
        self.histories.pop(epoch_i)

    def tell_observed_scores(self, epoch_i, sample_inds, scores):
        """Update the state of the searcher based on the actual scores of the 
        models proposed.

        """
        if len(sample_inds) != len(scores):
            raise ValueError
        if epoch_i not in self.sample_hist:
            raise KeyError

        epoch_samples = self.sample_hist[epoch_i]
        epoch_hists = self.histories[epoch_i]

        for i, sc in zip(sample_inds, scores):
            mdl = epoch_samples[i]
            hist = epoch_hists[i]

            self.known_models.append(mdl)
            self.known_hists.append(hist)
            self.known_scores.append(sc)

    def refit_model(self):
        """Learns a new surrogate model using the data observed so far.

        """

        # only fit the model if there is data for it.
        if len(self.known_models) > 0:

            self._build_feature_maps(self.known_models, self.ngram_maxlen, self.thres)

            X = sp.vstack([ self._compute_features(mdl)
                    for mdl in self.known_models], "csr")
            y = np.array(self.known_scores, dtype='float64')

            #A = np.dot(X.T, X) + lamb * np.eye(X.shape[1])
            #b = np.dot(X.T, y)
            self.surr_model = lm.Ridge(self.lamb_ridge)
            self.surr_model.fit(X, y)


# NOTE: if the search space has holes, it break. needs try/except module.
def run_smbo_searcher(evaluator, searcher,
        nsamples_start, nsamples_after, nsamples_epoch, refit_interval, explore_prob,
        output_to_terminal=False, ignore_invalid_models=False,
        save_hist_in_model=False):

    # initially, just sample a few models and evaluate them all.
    (epoch_i, models, choice_hists, pred_scores) =  \
        searcher.sample_new_epoch(nsamples_start)

    num_evals = 0
    ep_model_inds = []
    ep_true_scores = []
    for i in xrange(len(models)):
        mdl = models[i]
        hist = choice_hists[i]
        maybe_register_choice_hist(mdl, hist, save_hist_in_model)

        #sc = evaluator.eval_model(mdl)
        sc = evaluate_and_print(evaluator, mdl, 
                output_to_terminal, ignore_invalid_models)
        if sc != None:
            ep_model_inds.append(i)
            ep_true_scores.append(sc)
            num_evals += 1

    # forget the models after telling the values for the nodes.
    searcher.tell_observed_scores(epoch_i, ep_model_inds, ep_true_scores)
    searcher.forget_epoch(epoch_i)

    # compute the string representations from which the features are going to
    # be derived.
    evaluated_models = {b.repr_model() for b in models}
    for i in xrange(nsamples_after):
        if i % refit_interval == 0:
            searcher.refit_model()

        (epoch_i, models, choice_hists, pred_scores) = \
                searcher.sample_new_epoch(nsamples_epoch)

        # if it is an exploration episode, shuffle the order given by the 
        model_ordering = range(len(models))
        if np.random.rand() < explore_prob:
            np.random.shuffle(model_ordering)
        
        # goes through the models in the order specified.
        for mdl_i in model_ordering:
            mdl = models[mdl_i]
            hist = choice_hists[mdl_i]
            maybe_register_choice_hist(mdl, hist, save_hist_in_model)

            #sc = evaluator.eval_model(mdl)
            sc = evaluate_and_print(evaluator, mdl, 
                    output_to_terminal, ignore_invalid_models)
            if sc != None:
                searcher.tell_observed_scores(epoch_i, [mdl_i], [sc])
                num_evals += 1
                break

        searcher.forget_epoch(epoch_i)

    # retrieve the information about the models sampled.
    # NOTE: this returns all the scores that were already in the searcher
    # which is not desirable if the searcher already had information there.
    # this will be kept for now.
    # a fix would be to count the models evaluated and only return those.
    # NOTE: COMEBACK as it is now, it only returns the information about the 
    # the models that were evaluated in in this turn.
    srch_scores = searcher.known_scores[-num_evals:]
    srch_choice_hists = searcher.known_hists[-num_evals:]

    return (srch_scores, srch_choice_hists)

class MCTSearcher:
    def __init__(self, b_search, in_d, exploration_bonus=1.0):
        self.in_d = in_d
        self.b_search = b_search
        self.mcts_root_node = MCTSTreeNode(None)
        self.exploration_bonus = exploration_bonus

    def sample_models(self, num_samples):
        """Generates random specified models in the search space and evaluates
        them based on the currect surrogate model.

        The function returns the epoch number, the specified models, and the
        history of choices that lead to those particular models.

        """

        models = []
        choice_hists = []

        for _ in xrange(num_samples):
            # initialization of the model.
            bk = copy.deepcopy(self.b_search)
            bk.initialize(self.in_d, Scope())

            tree_hist = self._tree_walk(bk)
            #print tree_hist
            roll_hist = self._rollout_walk(bk)
            hist = ( tuple(tree_hist), tuple(roll_hist) )

            models.append(bk)
            choice_hists.append( hist )

        return (tuple(models), tuple(choice_hists))

    def tell_observed_scores(self, choice_hists, scores):
        # this is going to be the update to the scores.
        for hist, sc in zip(choice_hists, scores):
            (tree_hist, _) = hist
            self._update_stats(tree_hist, sc)

    # NOTE: all three functions below have side effects, namely with
    # regard to b_start, b_partial, and the tree itself.
    def _tree_walk(self, b_start):
        hist = []
        node = self.mcts_root_node
        while not node.is_leaf():
            (node, i) = node.best_child(self.exploration_bonus)
            b_start.choose(i)
            hist.append(i)

        # expand the leaf node in the tree for which we leave.
        if not b_start.is_specified():
            _, choices = b_start.get_choices()
            node.expand( len(choices) )

        return hist

    # b_partial assumes that some part of the model was already done.
    def _rollout_walk(self, b_partial):
        hist = []
        while( not b_partial.is_specified() ):
            _, vals = b_partial.get_choices()
            choice_i = np.random.randint(0, len(vals))
            b_partial.choose(choice_i)
            hist.append(choice_i)

        return hist

    def _update_stats(self, tree_hist, score):
        node = self.mcts_root_node
        node.update_stats(score)

        for ch_i in tree_hist:
            node = node.children[ch_i]
            node.update_stats(score)

# keeps the statistics and knows how to update information related to a node.
class MCTSTreeNode:
    def __init__(self, parent_node):
        self.num_trials = 0
        self.sum_scores = 0.0

        self.parent = parent_node
        self.children = None

    def is_leaf(self):
        return self.children == None

    def update_stats(self, score):
        self.sum_scores += score
        self.num_trials += 1

    # returns the child with the highest UCT score.
    def best_child(self, exploration_bonus):
        assert not self.is_leaf()

        # if two nodes have the same score. 
        best_inds = None
        best_score = -np.inf

        parent_log_nt = np.log(self.num_trials)
        for (i, node) in enumerate(self.children):
            # NOTE: potentially, do a different definition for the scores.
            # especially once the surrogate model is introduced.
            # selection policy may be somewhat biased towards what the 
            # rollout policy based on surrogate functions says.
            # think about how to extend this.
            if node.num_trials > 0:
                score = ( node.sum_scores / node.num_trials + 
                            exploration_bonus * np.sqrt(
                                2.0 * parent_log_nt / node.num_trials) )
            else:
                score = np.inf
            
            # keep the best node.
            if score > best_score:
                best_inds = [i]
                best_score = score
            elif score == best_score:
                best_inds.append(i)
            
            # draw a child at random and expand.
            best_i = np.random.choice(best_inds)
            best_child = self.children[best_i]

        return (best_child, best_i)

    # expands a node creating all the placeholders for the children.
    def expand(self, num_children):
        self.children = [MCTSTreeNode(self) for _ in xrange(num_children)]

# NOTE: if the search space has holes, it break. needs try/except module.
def run_mcts_searcher(evaluator, searcher, num_models,
        output_to_terminal=False, ignore_invalid_models=False,
        save_hist_in_model=False):

    srch_choice_hists = []
    srch_scores = []

    for _ in xrange(num_models):
        (models, hists) = searcher.sample_models(1)
        mdl = models[0]
        # has to join the tree and rollout histories to make a normal history.
        hist = tuple(hists[0])
        cache_hist = hist[0] + hist[1]
        maybe_register_choice_hist(mdl, cache_hist, save_hist_in_model)

        # evaluation of the model.
        sc = evaluate_and_print(evaluator, mdl, 
                output_to_terminal, ignore_invalid_models)
        if sc != None:
        #sc = np.random.random() ### come back here.
            searcher.tell_observed_scores([hist], [sc])
            srch_choice_hists.append(hist)
            srch_scores.append(sc)

    return (srch_scores, srch_choice_hists)
