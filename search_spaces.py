
from darch.modules import *
from darch.initializers import *

# this search space is used for exploration of the different search methods.
# 5 repetitions are to be run, with 15 minutes per model. 100 models total.
# 2 days per repetition per method, divided by the number of gpus used.
# *** has 2200 different models
def tfref_convnet_ss0(num_classes):
    """ Naive conv net approximately as described in 
    https://www.tensorflow.org/tutorials/mnist/pros/ . See docstring for 
    tfref_convnet_single. Search space built around that.
    """ 

    conv_initers = [ kaiming2015delving_initializer_conv(1.0) ]
    aff_initers = [ invsqrt_size_gaussian_initializer_affine( np.sqrt(2.0) )]

    # some parameters.
    filter_ns = [32, 48, 64, 96, 128]
    filter_ls = [1, 3, 5]
    aff_ns = [256, 512, 1024]
    keep_ps = [0.25, 0.5, 0.75]

    b_search = Concat([
                    Conv2D(filter_ns, filter_ls, [1], ["SAME"], conv_initers), 
                    ReLU(),
                    MaxPooling2D([2], [2], ["SAME"]),
                    Conv2D(filter_ns, filter_ls, [1], ["SAME"], conv_initers), 
                    ReLU(),
                    MaxPooling2D([2], [2], ["SAME"]),
                    Affine(aff_ns, aff_initers),
                    ReLU(),
                    Dropout(keep_ps),
                    Affine([num_classes], aff_initers) 
               ])
    return b_search

# this search space is used for searching for models with a larger time (e.g. 
# 3h). run on 15 models; then take the 3 most promising ones and train them 
# for 10 tries with a larger computational budget, but still with the stopping 
# condition 12 hours at most.
# try for random.
# these search spaces are much bigger.
def deepconv_ss0(num_classes):
    conv_initers = [ kaiming2015delving_initializer_conv(1.0) ]
    aff_initers = [ xavier_initializer_affine( 1.0 )]

    def Module_fn(filter_ns, filter_ls, keep_ps, repeat_ns):
        b = RepeatTied(
                Concat([
                    Conv2D(filter_ns, filter_ls, [1], ["SAME"], conv_initers),
                    MaybeSwap_fn( ReLU(), BatchNormalization() ),
                    Optional_fn( Dropout(keep_ps) )
            ]), repeat_ns)
        return b

    filter_numbers_min = range(48, 129, 16)
    repeat_numbers = [2 ** i for i in xrange(6)]
    mult_fn = lambda ls, alpha: list(alpha * np.array(ls))

    b_search = Concat([
            # this reduction layer wasn't here before, but it is convenient.
            Conv2D(filter_numbers_min, [3, 5, 7], [2], ["SAME"], conv_initers), 
            Module_fn(filter_numbers_min, [3, 5], [0.5, 0.9], repeat_numbers),
            Conv2D(filter_numbers_min, [3, 5, 7], [2], ["SAME"], conv_initers),
            Module_fn(mult_fn(filter_numbers_min, 2), [3, 5], [0.5, 0.9], repeat_numbers),
            Affine([num_classes], aff_initers)
        ])
    return b_search

# use the same type of experimental setting as the deepconv model.
# this search space is much bigger.
def resnet_ss0(num_classes):
    gains = [1.0]
    aff_initers = tuple([ xavier_initializer_affine(g) for g in gains ])
    conv2d_initers = [kaiming2015delving_initializer_conv(g) for g in gains]
    
    def Res_fn(filter_ns, filter_ls, keep_ps, repeat_inner_ns):
        return Concat([
                Residual(
                    Concat([
                        RepeatTied(
                            Concat([
                                Conv2D(filter_ns, filter_ls, [1], ["SAME"], conv2d_initers),
                                MaybeSwap_fn( BatchNormalization(), ReLU() )
                            ]), 
                        repeat_inner_ns),
                        Conv2D(filter_ns, filter_ls, filter_ls, ["SAME"], conv2d_initers)
                    ])
                ),
                Optional_fn( Dropout(keep_ps) )
            ])

    filter_ns = range(48, 129, 16) 
    filter_ls = [1, 3, 5]
    repeat_numbers = [2 ** i for i in xrange(6)]
    mult_fn = lambda ls, alpha: list(alpha * np.array(ls))
    repeat_inner_ns = [1, 2, 4]
    repeat_outer_ns = [1, 2, 4, 8]
    keep_ps = [0.5, 0.9]

    b_search = Concat([
                    Repeat( 
                        Res_fn(filter_ns, filter_ls, keep_ps, repeat_inner_ns),
                    repeat_outer_ns),
                    Optional_fn(
                        Concat([
                            AvgPooling2D([3, 5], [2], ["SAME"]), 
                            Repeat(
                                Res_fn(mult_fn(filter_ns, 2), filter_ls, keep_ps, repeat_inner_ns),
                            repeat_outer_ns)
                        ])
                    ),
                    Affine([num_classes], aff_initers)
                ])
    return b_search
