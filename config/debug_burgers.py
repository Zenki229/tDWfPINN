import ml_collections
'''
The code structrue is referred from https://github.com/PredictiveIntelligenceLab/jaxpi
'''

def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()
    config.mode = "train"
 
    # save
    config.version = "v1"
    config.dir = ml_collections.ConfigDict()
    config.dir.name='burgers/debug'
    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "Mlp"
    arch.input_dim = 2
    arch.num_layers = 2
    arch.hidden_dim = 2
    arch.output_dim = 1
    arch.activation = "tanh"
    # # Optim
    config.adam = adam = ml_collections.ConfigDict()
    adam.lr = 1e-3
    adam.epoch = 400
    config.lbfgs= lbfgs = ml_collections.ConfigDict()
    lbfgs.use=True
    lbfgs.max_iter=10
    lbfgs.max_eval=10
    lbfgs.lr = 0.1
    lbfgs.epoch=2
    # scheduler

    # quadrature
    config.method = 'GJ-II'
    config.MC = MC = ml_collections.ConfigDict()
    MC.nums = 50
    MC.eps  =1e-10
    config.GJ = GJ = ml_collections.ConfigDict()
    GJ.nums = 1000

    #pde setting
    config.al = 1.5
    config.datafile  = "./data/burgers_150.npz"
    config.beta = 2.0
    config.xlim = [[0,1]]
    config.tlim = [0,1]
    config.weighting = {'in':1, 'bd':1, 'init': 1, 'init_dt':1}
    # # Training
    config.cuda_dev = "0" # Default on cuda:0
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 2
    training.batch = {'in':1000,'bd':10,'init':10}
    # RAD setting
    config.RAD = RAD = ml_collections.ConfigDict() 
    RAD.use = True 
    RAD.ratio = 0.8
    RAD.batch = {'in':10000,'bd':2,'init':2}

    return config
