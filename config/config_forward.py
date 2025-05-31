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
    config.dir.name='forward'
    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "Mlp"
    arch.input_dim = 2
    arch.num_layers = 7
    arch.hidden_dim = 20
    arch.output_dim = 1
    arch.activation = "tanh"
    # # Optim
    config.adam = adam = ml_collections.ConfigDict()
    adam.lr = 1e-3
    adam.epoch = 5000
    config.lbfgs= lbfgs = ml_collections.ConfigDict()
    lbfgs.use=False
    lbfgs.max_iter=2500
    lbfgs.max_eval=2500
    lbfgs.lr = 0.3
    lbfgs.epoch=1
    # scheduler

    # quadrature
    config.method = 'GJ-II'
    config.MC = MC = ml_collections.ConfigDict()
    MC.nums = 50
    MC.eps  =1e-7
    config.GJ = GJ = ml_collections.ConfigDict()
    GJ.nums = 80

    #pde setting
    config.al = 1.75
    config.k=1
    config.lam='1.0'
    config.xlim = [[0,1]]
    config.tlim = (0,1)
    config.a=1.0
    config.b=-0.5
    config.weight_param = (1,1,1,1)
    # # Training
    config.cuda_dev = '0' # Default on cuda:0
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 3
    training.nums = (5000,1000,1000)
    # RAD setting
    config.RAD = RAD = ml_collections.ConfigDict() 
    RAD.use = False
    RAD.ratio = 0.3
    RAD.batch = {'in':10000,'bd':2,'init':2}

    return config
