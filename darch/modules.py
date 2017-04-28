
import base 
import numpy as np
import tensorflow as tf
import copy

### Basic modules describing the core search and computational functionality.
class BasicModule(object):
    """Basic search and compilation functionality for basic modules.

    Basic modules are composed of parameters and hyperparameters. They are not 
    composite in the sense that do not have submodules as it is the case of 
    Concat and Or modules. This class is not meant to be used directly. 
    It has to be inherited and the functions that compute the output dimension 
    and compile the module to tensorflow has to be implemented for the specific 
    module considered. See Affine and Dropout class definitions for examples
    of how to extend this class.

    """

    def __init__(self):
        self.order = []
        self.domains = []
        self.chosen = []
        self.in_d = None
        self.scope = None
        self.namespace_id = None

    # for search
    def initialize(self, in_d, scope):
        if self.in_d != None or self.scope != None:
            raise AssertionError
        #print(in_d)
        if len(in_d) < 1:
            raise ValueError

        self.in_d = in_d
        self.scope = scope
        
        # registers itself in the scope
        prefix = self.__class__.__name__ 
        name = scope.get_valid_name(prefix)
        self.namespace_id = name
        scope.register_namespace(name)
        propagate(self)

    def get_outdim(self):
        raise NotImplemented
        
    def is_specified(self):
        return (self.scope != None and 
                self.in_d != None and 
                len(self.chosen) == len(self.order))

    def get_choices(self):
        assert not self.is_specified()

        i = len(self.chosen)
        return (self.order[i], self.domains[i])

    def choose(self, choice_i):
        assert not self.is_specified()

        # print self.chosen, choice_i
        if len(self.domains[len(self.chosen)]) <= choice_i:
            raise ValueError

        self.chosen.append(choice_i)
        propagate(self)

    # for printing
    def repr_program(self):
        name = self.__class__.__name__
        return (name, ) + tuple(self.domains) 

    def repr_model(self):
        name = self.__class__.__name__ 
        vals = [dm[i] for i, dm in zip(self.chosen, self.domains)]
        r = (name, ) + tuple(vals)
        return (r, )

    # for generating the tensorflow code
    def compile(self, in_x, train_feed, eval_feed):
        raise NotImplemented

class Empty(BasicModule):
    """Empty Module.

    Compiles to a wire. Directly passes the input to the output without 
    any transformation.

    """

    def get_outdim(self):
        return self.in_d

    def compile(self, in_x, train_feed, eval_feed):
        return in_x

class Dropout(BasicModule):
    """Dropout module.

    Dropout has different training behavior depending on whether the network
    is being run on a training or on evaluation phase. Getting the desired 
    behavior in each case is achieved through the use of scopes.

    """

    def __init__(self, ps):
        super(Dropout, self).__init__()
        self.order.append("keep_prob")
        self.domains.append(ps)

    def get_outdim(self):
        return self.in_d

    def compile(self, in_x, train_feed, eval_feed):
        p_name = self.namespace_id + '_' + self.order[0] 
        p_var = tf.placeholder(tf.float32, name=p_name)

        # during training the value of the dropout probability (keep_prob) is 
        # set to the actual chosen value. 
        # during evalution, it is set to 1.0. 
        p_val = self.domains[0][self.chosen[0]]
        train_feed[p_var] = p_val
        eval_feed[p_var] = 1.0
        out_y = tf.nn.dropout(in_x, p_var)
        return out_y

class BatchNormalization(BasicModule):
    def get_outdim(self):
        return self.in_d

    def compile(self, in_x, train_feed, eval_feed):
        p_name = self.namespace_id + '_' + "IsTraining"
        p_var = tf.placeholder(tf.bool, name=p_name)
        train_feed[p_var] = True
        eval_feed[p_var] = False
        out_y = tf.contrib.layers.batch_norm(
                    in_x, decay=0.9, is_training=p_var, updates_collections=None)
        return out_y

class ReLU(BasicModule):
    def get_outdim(self):
        return self.in_d

    def compile(self, in_x, train_feed, eval_feed):
        out_y = tf.nn.relu(in_x)
        return out_y

class Sigmoid(BasicModule):
    def get_outdim(self):
        return self.in_d

    def compile(self, in_x, train_feed, eval_feed):
        out_y = tf.nn.sigmoid(in_x)
        return out_y

class Tanh(BasicModule):
    def get_outdim(self):
        return self.in_d

    def compile(self, in_x, train_feed, eval_feed):
        out_y = tf.nn.tanh(in_x)
        return out_y

class Affine(BasicModule):
    def __init__(self, ms, param_init_fns):
        super(Affine, self).__init__()
        self.order.extend(["num_hidden_units", "param_init_fn"])
        self.domains.extend([ms, param_init_fns])

    def get_outdim(self):
        return (self.domains[0][self.chosen[0]], )

    def compile(self, in_x, train_feed, eval_feed):
        n = np.product(self.in_d)
        m, param_init_fn = [dom[i] for (dom, i) in zip(self.domains, self.chosen)]

        #sc = np.sqrt(6.0) / np.sqrt(m + n)
        #W = tf.Variable(tf.random_uniform([n, m], -sc, sc))
        W = tf.Variable( param_init_fn( [n, m] ) )
        b = tf.Variable(tf.zeros([m]))

        # if the number of input dimensions is larger than one, flatten the 
        # input and apply the affine transformation. 
        if len(self.in_d) > 1:
            in_x_flat = tf.reshape(in_x, shape=[-1, n])
            out_y = tf.add(tf.matmul(in_x_flat, W), b)
        else:
            out_y = tf.add(tf.matmul(in_x, W), b)
        return out_y

# computes the output dimension based on the padding scheme used.
# this comes from the tensorflow documentation
def compute_padded_dims(in_height, in_width, window_len, stride, padding):

    if padding == "SAME":
        out_height = int(np.ceil(float(in_height) / float(stride)))
        out_width  = int(np.ceil(float(in_width) / float(stride)))
    elif padding == "VALID":
        out_height = int(np.ceil(float(in_height - window_len + 1) / float(stride)))
        out_width  = int(np.ceil(float(in_width - window_len + 1) / float(stride)))
    else:
        raise ValueError

    return (out_height, out_width)

class MaxPooling2D(BasicModule):
    def __init__(self, window_lens, strides, paddings):
        super(MaxPooling2D, self).__init__()

        self.order.extend(["window_len", "stride", "padding"])
        self.domains.extend([window_lens, strides, paddings])

    # does additional error checking on the dimension.
    def initialize(self, in_d, scope):
        if len(in_d) != 3:
            raise ValueError
        else:
            super(MaxPooling2D, self).initialize(in_d, scope)

    def get_outdim(self):
        in_height, in_width, in_nchannels = self.in_d 
        window_len, stride, padding = [dom[i] 
                for (dom, i) in zip(self.domains, self.chosen)]

        out_height, out_width = compute_padded_dims(
            in_height, in_width, window_len, stride, padding)
        out_d = (out_height, out_width, in_nchannels)

        return out_d
            
    def compile(self, in_x, train_feed, eval_feed):
        in_height, in_width, in_nchannels = self.in_d 
        window_len, stride, padding = [dom[i] 
                for (dom, i) in zip(self.domains, self.chosen)]

        out_y = tf.nn.max_pool( 
            in_x, ksize=[1, window_len, window_len, 1], 
            strides=[1, stride, stride, 1], padding=padding)

        return out_y

# NOTE: for now, this is close to a replication of the max pooling layer.
# this may change later if we can capture most of the pooling layers in the 
# same format. for now, an auxiliary function is provided.
class AvgPooling2D(BasicModule):
    def __init__(self, window_lens, strides, paddings):
        super(AvgPooling2D, self).__init__()

        self.order.extend(["window_len", "stride", "padding"])
        self.domains.extend([window_lens, strides, paddings])

    # does additional error checking on the dimension.
    def initialize(self, in_d, scope):
        if len(in_d) != 3:
            raise ValueError
        else:
            super(AvgPooling2D, self).initialize(in_d, scope)

    def get_outdim(self):
        in_height, in_width, in_nchannels = self.in_d 
        window_len, stride, padding = [dom[i] 
                for (dom, i) in zip(self.domains, self.chosen)]

        out_height, out_width = compute_padded_dims(
            in_height, in_width, window_len, stride, padding)
        out_d = (out_height, out_width, in_nchannels)

        return out_d
            
    def compile(self, in_x, train_feed, eval_feed):
        in_height, in_width, in_nchannels = self.in_d 
        window_len, stride, padding = [dom[i] 
                for (dom, i) in zip(self.domains, self.chosen)]

        out_y = tf.nn.avg_pool( 
            in_x, ksize=[1, window_len, window_len, 1], 
            strides=[1, stride, stride, 1], padding=padding)

        return out_y

class Conv2D(BasicModule):

    def __init__(self, filter_numbers, filter_lens, strides, paddings, 
            param_init_fns):
        super(Conv2D, self).__init__()

        self.order.extend(["filter_number", "filter_len", "stride", "padding",
            "param_init_fn"])
        self.domains.extend([filter_numbers, filter_lens, strides, paddings, 
            param_init_fns])

    # does additional error checking on the dimension.
    def initialize(self, in_d, scope):
        #print(in_d)
        if len(in_d) != 3:
            raise ValueError
        else:
            super(Conv2D, self).initialize(in_d, scope)

    def get_outdim(self):
        in_height, in_width, in_nchannels = self.in_d 
        nfilters, filter_len, stride, padding, _ = [dom[i] 
                for (dom, i) in zip(self.domains, self.chosen)]

        out_height, out_width = compute_padded_dims(
                in_height, in_width, filter_len, stride, padding)
        out_d = (out_height, out_width, nfilters)

        return out_d
            
    def compile(self, in_x, train_feed, eval_feed):
        in_height, in_width, in_nchannels = self.in_d 
        nfilters, filter_len, stride, padding, param_init_fn = [dom[i] 
                for (dom, i) in zip(self.domains, self.chosen)]

        # Creation and initialization of the parameters. Should take size of 
        # the filter into account.
        W = tf.Variable(
                param_init_fn( [filter_len, filter_len, in_nchannels, nfilters]) )
        b = tf.Variable(tf.zeros([nfilters]))

        # create the output and add the bias.
        out_yaux = tf.nn.conv2d(in_x, W, strides=[1, stride, stride, 1], padding=padding)
        out_y = tf.nn.bias_add(out_yaux, b)

        #print(in_x.get_shape(), self.get_outdim(), out_y.get_shape())

        return out_y

class UserHyperparams(BasicModule):
    """Used by the user to specify an additional set of hyperparameters that 
    the user also wants to search over. The behavior of the program with 
    respect to these hyperparameters is determined by the user.
    """
    def __init__(self, order, domains):
        super(UserHyperparams, self).__init__()
        self.order = order
        self.domains = domains

    def get_outdim(self):
        return self.in_d

    # only publishes the values of the hyperparameters to the scope.
    def compile(self, in_x, train_feed, eval_feed):
        namespace = self.scope.get_namespace(self.namespace_id)
        namespace["hyperp_names"] = self.order
        namespace["choices"] = tuple(self.chosen)
        namespace["hyperp_vals"] = [dom[i] for (dom, i) in zip(self.domains, self.chosen)]
        
        return in_x

### Auxiliary function for dimension and choice propagation.
def propagate_seq(bs, i):
    """ Propagates choices in a sequence of modules.

    If the module in the current position of the sequence is specified, we can 
    initialize the next module in the sequence (if there is any), and go to 
    the next module in the chain if the initialized module becomes specified.
    
    """

    while bs[i].is_specified():
        prev_scope = bs[i].scope
        prev_out_d = bs[i].get_outdim()

        i += 1
        if i < len(bs):
            bs[i].initialize(prev_out_d, prev_scope)
        else:
            break
    return i

def propagate(b):
    """ Propagates choices in a module.

    While the module is in a state where there is only one option available for 
    the next choice, we take that choice. This function leaves the module 
    specified or in a state where there multiple choices. This function will 
    typically be called by the submodule when initialize or choose is called 
    on that module.

    """
    while not b.is_specified() and len(b.get_choices()[1]) == 1:
        b.choose(0)

### Composite modules that take other modules as input. 
class Concat:
    def __init__(self, bs):
        if len(bs) == 0:
            raise ValueError

        self.bs = bs
        self.in_d = None 
        self.scope = None

    def initialize(self, in_d, scope):
        self.in_d = in_d
        self.scope = scope

        self.b_index = 0
        self.bs[0].initialize(self.in_d, self.scope)
        self.b_index = propagate_seq(self.bs, self.b_index)

    def get_outdim(self):
        return self.bs[-1].get_outdim()
        
    def is_specified(self):
        return self.bs[-1].is_specified()

    def get_choices(self):
        return self.bs[self.b_index].get_choices()

    def choose(self, choice_i):
        self.bs[self.b_index].choose(choice_i)
        self.b_index = propagate_seq(self.bs, self.b_index)

    def repr_program(self):
        name = self.__class__.__name__
        args = [b.repr_program() for b in self.bs]
        return (name, ) + tuple(args) 

    def repr_model(self):
        vals = []
        for b in self.bs:
            vals.extend(b.repr_model()) 
        return tuple(vals)

    def compile(self, in_x, train_feed, eval_feed):
        out_y_prev = in_x
        for b in self.bs:
            out_y_prev = b.compile(out_y_prev, train_feed, eval_feed)
        return out_y_prev

class Or:
    def __init__(self, bs):
        if len(bs) == 0:
            raise ValueError

        self.order = ["or_branch"]
        self.domains = [ range(len(bs)) ]
        self.bs = bs
        self.in_d = None 
        self.chosen = []
        self.scope = None

    def initialize(self, in_d, scope):
        self.in_d = in_d
        self.scope = scope
        propagate(self)

    def get_outdim(self):
        i = self.chosen[0]

        return self.bs[i].get_outdim()

    def is_specified(self):
        if len(self.chosen) == 1:
            i = self.chosen[0]
            return self.bs[i].is_specified()
        else:
            return False

    def get_choices(self):
        if len(self.chosen) == 0:
            return (self.order[0], self.domains[0])
        else:
            i = self.chosen[0]
            return self.bs[i].get_choices()

    def choose(self, choice_i):
        if len(self.chosen) == 0:
            self.chosen.append(choice_i)
            self.bs[choice_i].initialize(self.in_d, self.scope)
        else:
            b_i = self.chosen[0]
            self.bs[b_i].choose(choice_i)

    def repr_program(self):
        name = self.__class__.__name__
        args = [b.repr_program() for b in self.bs]
        return (name, ) + tuple(args) 

    def repr_model(self):
        vals = self.bs[self.chosen[0]].repr_model()
        return vals

    def compile(self, in_x, train_feed, eval_feed):
        out_y = self.bs[self.chosen[0]].compile(in_x, train_feed, eval_feed)
        return out_y

class Repeat:
    """Repeat module.

    Takes a module as input and repeats it some number of times. The number of 
    repeats is itself an hyperparameter. Hyperparameters of each repeat are not 
    tied across repeats. See the RepeatTied module for tied hyperparameters 
    across repeats.

    """
    def __init__(self, b, ks):
        if any([k < 1 for k in ks]): 
            raise ValueError

        self.order = ["num_repeats"]
        self.domains = [ks]
        self.b = b

        # used during search
        self.in_d = None 
        self.chosen = []
        self.active_bs = None
        self.b_index = None

    def initialize(self, in_d, scope):
        self.in_d = in_d
        self.scope = scope
        propagate(self)

    def get_outdim(self):
        return self.active_bs[-1].get_outdim()
        
    def is_specified(self):
        if len(self.chosen) == 1:
            return self.active_bs[-1].is_specified()
        else:
            return False

    def get_choices(self):
        if len(self.chosen) == 0:
            return (self.order[0], self.domains[0])
        else:
            return self.active_bs[self.b_index].get_choices()

    def choose(self, choice_i):
        if len(self.chosen) == 0:
            self.chosen.append(choice_i)
            k = self.domains[0][choice_i]

            self.b_index = 0
            self.active_bs = [copy.deepcopy(self.b) for _ in xrange(k)] 
            self.active_bs[0].initialize(self.in_d, self.scope) 
        else:
            self.active_bs[self.b_index].choose(choice_i)

        self.b_index = propagate_seq(self.active_bs, self.b_index)

    def repr_program(self):
        name = self.__class__.__name__
        args = self.b.repr_program()

        return (name, ) + tuple(args) + tuple(self.domains)

    def repr_model(self):
        vals = []
        for b in self.active_bs:
            vals.extend(b.repr_model()) 
        return tuple(vals)

    def compile(self, in_x, train_feed, eval_feed):
        out_y_prev = in_x
        for b in self.active_bs:
            out_y_prev = b.compile(out_y_prev, train_feed, eval_feed)
        return out_y_prev


class RepeatTied:
    """RepeatTied module.

    Like a Repeat module, but now the hyperparameters are tied across repeats.
    Note that only the hyperparameters, not the parameters, are tied.
    Tying parameters is done through other route.
    
    """
    def __init__(self, b, ks):
        if any([k < 1 for k in ks]): 
            raise ValueError

        self.order = ["num_repeats"]
        self.domains = [ks]
        self.b = b

        # used during search
        self.in_d = None 
        self.chosen = []
        self.active_bs = None
        self.b_index = None
        # this field is for the reuse in setting the repeats b
        self.b0_choose_hist = []

    def initialize(self, in_d, scope):
        self.in_d = in_d
        self.scope = scope
        propagate(self)

    def get_outdim(self):
        return self.active_bs[-1].get_outdim()
        
    def is_specified(self):
        if len(self.chosen) == 1:
            return self.active_bs[-1].is_specified()
        else:
            return False

    def get_choices(self):
        if len(self.chosen) == 0:
            return (self.order[0], self.domains[0])
        else:
            return self.active_bs[self.b_index].get_choices()

    def choose(self, choice_i):
        if len(self.chosen) == 0:
            self.chosen.append(choice_i)
            k = self.domains[0][choice_i]

            self.b_index = 0
            self.active_bs = [copy.deepcopy(self.b) for _ in xrange(k)] 
            self.active_bs[0].initialize(self.in_d, self.scope) 
        else:
            self.active_bs[self.b_index].choose(choice_i)
            self.b0_choose_hist.append(choice_i)

        self.b_index = propagate_seq(self.active_bs, self.b_index)

        # as soon as the pointer moves to 1, set all others by copying the 
        # choices.
        if self.b_index == 1:
            for b in self.active_bs[1:]:
                for ch in self.b0_choose_hist:
                    b.choose(ch)
                
                # this should advance the pointer by one.
                self.b_index = propagate_seq(self.active_bs, self.b_index)
        
            # it should be specified at the end of this.
            assert self.is_specified()

    def repr_program(self):
        name = self.__class__.__name__
        args = self.b.repr_program()

        return (name, ) + tuple(args) + tuple(self.domains)

    def repr_model(self):
        vals = []
        for b in self.active_bs:
            vals.extend(b.repr_model()) 
        return tuple(vals)

    def compile(self, in_x, train_feed, eval_feed):
        out_y_prev = in_x
        for b in self.active_bs:
            out_y_prev = b.compile(out_y_prev, train_feed, eval_feed)
        return out_y_prev

class Residual:
    """ Residual skip connection.

    Introduces a skip connection between the module passed as argument. 
    The module taken as argument can have hyperparameters to be specified.

    If the input and output do not have the same dimensions, padding needs
    to be done for the results to be combined in a sum or product. We briefly
    discuss the different cases:
    (1)| both input and output have the same number of dimensions and the same 
    sizes for paired dimensions. 
    => simply do the entrywise operation without doing any changes.
    (2)| both input and output have the same number of dimensions, but they 
    have different sizes for paired dimensions.
    => pad the smallest dimensions on either input or output such that the 
    result after padding can be combined; both input and output can be changed 
    in this case if none strictly dominates the other in terms of dimensions.
    (3)| input and ouput have different number of dimensions.
    => perhaps the most straightforward solution is to flatten both input and 
    output and combine the flattened versions. another possibility is to add 
    extra dimensions and pad the smallest dimensions with zeros.

    The most straightforward solutions have been implemented for now.

    """

    def __init__(self, b):
        self.b = b
        self.in_d = None 
        self.scope = None

    def initialize(self, in_d, scope):
        self.in_d = in_d
        self.scope = scope
        self.b.initialize(in_d, scope)
        propagate(self.b)

    def get_outdim(self):
        #assert in_x == self.b.get_outdim()
        # relaxing input dimension equal to output dimension. taking into
        # account the padding scheme considered.
        out_d_b = self.b.get_outdim()
        in_d = self.in_d

        if len(out_d_b) == len(in_d):
            out_d = tuple(
                [max(od_i, id_i) for (od_i, id_i) in zip(out_d_b, in_d)])

        else:
            # flattens both input and output. 
            out_d_b_flat = np.product(out_d_b)
            in_d_flat = np.product(in_d)
            out_d = (max(out_d_b_flat, in_d_flat) ,)

        return out_d
        
    def is_specified(self):
        return self.b.is_specified()

    def get_choices(self):
        return self.b.get_choices()

    def choose(self, choice_i):
        self.b.choose(choice_i)

    def repr_program(self):
        name = self.__class__.__name__
        args = self.b.repr_program()

        return (name, args)

    def repr_model(self):
        name = self.__class__.__name__ 
        b = self.b.repr_model()
        r = (name, b[0])

        return (r, )

    def compile(self, in_x, train_feed, eval_feed):
        # NOTE: this function requires that target dims dominate (bigger or
        # equal component wise) in_dims. this is the case in how it is used
        # currently in the code below.
        compute_padding_fn = lambda in_dims, out_dims: [ [0, max(0, od_i - id_i)] 
                for (id_i, od_i) in zip(in_dims, out_dims) ]

        out_d_b = self.b.get_outdim()
        in_d = self.in_d
        out_d = self.get_outdim()
        out_y_b = self.b.compile(in_x, train_feed, eval_feed)

        # if the number of dimensions is not the same, flatten both.
        if len(out_d) != len(in_d):
            out_d_b_flat = np.product(out_d_b)
            out_y_b = tf.reshape(out_y_b, [-1, out_d_b_flat])
            out_d_b = (out_d_b_flat, )

            in_d_flat = np.product(in_d)
            in_x = tf.reshape(in_x, [-1,  in_d_flat])
            in_d = (in_d_flat, )

        # computing the padding for both b and the input.
        # NOTE: adds no padding to the data dimension (i.e., the initial [0, 0])
        paddings_b = [[0, 0]] + compute_padding_fn(out_d_b, out_d)
        out_y_b_padded = tf.pad(out_y_b, paddings_b, "CONSTANT") 

        paddings_in = [[0, 0]] + compute_padding_fn(in_d, out_d)
        in_x_padded = tf.pad(in_x, paddings_in, "CONSTANT") 

        # finally combine the results with the appropriate dimensions.
        out_y = out_y_b_padded + in_x_padded

        return out_y

class Squeeze(BasicModule):
    def __init__(self):
        super(Squeeze, self).__init__()

    def initialize(self, in_d, scope):
        super(Squeeze, self).initialize(in_d, scope)

    def get_outdim(self):
        out_d = filter(lambda a: a != 1, self.in_d)
        return out_d

    def compile(self, in_x, train_feed, eval_feed):
        out_y = tf.squeeze(in_x)
        return out_y

# it is kind of like the previous one.
class ChoiceBisection:
    """Does bissection on all the hyperparameters of module taken as argument. 
    This can be useful to increase sharing between hyperparameter values in 
    approaches such as MCTS.
    """
    def __init__(self, b):
        self.b = b
        self.is_bisecting = False
        self.left = None
        self.right = None
        self.hist = []
        self.cur_name = None
        self.cur_vals = None
        # hist is kept to create the create a name for the binary choice with
        # a suffix that corresponds to the sequence of binary decisions done 
        # so far.

    def initialize(self, in_d, scope):
        self.b.initialize(in_d, scope)

    def get_outdim(self):
        return self.b.get_outdim()

    def is_specified(self):
        return self.b.is_specified()

    def get_choices(self):
        # only potentially called in the first call to get_choices.
        if not self.is_bisecting:
            self.cur_name, self.cur_vals = self.b.get_choices()
            self.left = 0
            self.right = len(self.cur_vals)
            self.is_bisecting = True
            assert len(self.cur_vals) > 1

        name = self.cur_name + '_' + "".join(map(str, self.hist))
        choices = (0, 1)

        return (name, choices)

    def choose(self, choice_i):
        if not self.is_bisecting:
            self.cur_name, self.cur_vals = self.b.get_choices()
            self.left = 0
            self.right = len(self.cur_vals)
            self.is_bisecting = True
            assert len(self.cur_vals) > 1

        assert self.is_bisecting and choice_i == 0 or choice_i == 1

        # doing the bissection
        mid = int( (self.left + self.right) / 2.0 ) 
        if choice_i == 0: 
            self.right = mid
        else:
            self.left = mid

        self.hist.append(choice_i)

        # comitting to the choice when there is a single one left.
        if self.right - self.left == 1:
            self.b.choose(self.left)
            self.is_bisecting = False
            self.left = None
            self.right = None
            self.hist = []

    def repr_program(self):
        name = self.__class__.__name__
        args = self.b.repr_program()

        return (name, args)

    def repr_model(self):
        return self.b.repr_model()

    # for generating the tensorflow code
    def compile(self, in_x, train_feed, eval_feed):
        return self.b.compile(in_x, train_feed, eval_feed)

# auxiliary functions for some common operations.
def Optional_fn(b):
    return Or([Empty(), b])

def Nonlinearity_fn(nonlin_types):
    bs = [] 

    for t in nonlin_types:
        if t == "relu":
            b = ReLU()
        elif t == 'sigmoid':
            b = Sigmoid()
        elif t == 'tanh':
            b = Tanh()
        else:
            raise ValueError

        bs.append(b)

    return Or(bs)

def Pooling2D_fn(pooling_types, window_lens, strides, paddings):
    bs = [] 

    for t in pooling_types:
        if t == "max":
            b = MaxPooling2D(window_lens, strides, paddings)
        elif t == 'avg':
            b = AvgPooling2D(window_lens, strides, paddings)
        else:
            raise ValueError

        bs.append(b)

    return Or(bs)

def MaybeSwap_fn(b1, b2):
    """Builds a module that has a parameter to swapping the order of modules 
    passed as argument.
    """

    b = Or([
            Concat([b1, b2]), 
            Concat([b2, b1])
        ])
    return b
