from libs import *
from absl import app
from absl import flags
from absl import logging
from libs.pde_forward import DWForward
from ml_collections import config_flags
FLAGS = flags.FLAGS
flags.DEFINE_string("workdir", '.', "Directory to store model data.")
flags.DEFINE_boolean("overwrite", True, "Overwrite existing workdir if it exists.")
config_flags.DEFINE_config_file(
    "config",
    "./config/debug_forward.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)
def main(argv):
    # set torch float64 as default 
    torch.set_default_dtype(torch.float64)
    config = FLAGS.config
    config.workdir = FLAGS.workdir
    config.overwrite = FLAGS.overwrite
    path_save = os.path.join(FLAGS.workdir, 'results', FLAGS.config.dir.name,FLAGS.config.version)
    config.path_save = path_save
    generate_path_save(config)
    with open(os.path.join(path_save,'config.txt'), 'w') as f:
        f.write(str(config))
    config.dev = torch.device('cuda:'+config.cuda_dev if torch.cuda.is_available() else 'cpu')
    logger = log_gen(path=config.path_save)
    net = Mlp(config.arch)
    net.to(device=config.dev)
    adam = torch.optim.Adam(net.parameters(), lr=config.adam.lr)
    if config.lbfgs.use== False:
        lbfgs = None
    else:
        lbfgs = torch.optim.LBFGS(net.parameters(), lr=config.lbfgs.lr, max_iter=config.lbfgs.max_iter, max_eval=config.lbfgs.max_eval, history_size=50, tolerance_grad=1e-7, tolerance_change=1.0 * np.finfo(float).eps, line_search_fn="strong_wolfe")
    sampler = iter(TimeSpaceEasySampler(axeslim=config.xlim, tlim=config.tlim, dev=config.dev, batch=config.training.batch))
    rad_sampler = iter(TimeSpaceEasySampler(axeslim=config.xlim, tlim=config.tlim, dev=config.dev, batch=config.RAD.batch))
    pde = DWForward(config)
    # START TRAINING
    total_count = 0 
    criterion = torch.nn.MSELoss()
    columns = ['epoch', 'name','in','bd','init','init_dt'] 
    points = sampler.sample()
    t_start = time.time()
    adam_end_time = 0.0
    lbfgs_end_time = 0.0
    optim_name='adam'
    def closure():
        nonlocal total_count, optim_name
        adam.zero_grad() if optim_name == 'adam' else lbfgs.zero_grad()
        residuals = pde.residual(net, points)
        loss = torch.zeros((1,)).to(config.dev)
        losses = {}
        for key in residuals.keys():
            losses[key] = criterion(residuals[key], torch.zeros_like(residuals[key])) 
            loss += config.weighting[key] * losses[key]
        loss_save.append(loss.item())
        loss.backward()
        values = [total_count, optim_name] + [losses[key].item() for key in ['in', 'bd', 'init', 'init_dt']]
        table = tabulate.tabulate([values], columns, tablefmt="simple", floatfmt="5.4e")
        if total_count % 500 == 0:
            table = table.split("\n")
            table = "\n".join([table[1]] + table)
        else:
            table = table.split("\n")[2]
        if total_count % 100 == 0:
            print(table)
        if total_count % 200 == 0:
            logger.info(table)
        total_count += 1
        return loss.item() 
    for count in range(config.training.max_steps):
        message = '=' * 3 + f'{count}-th ' + 'training'+'='*10
        logger.info(message)
        print(message)
        t1=time.time()
        loss_save = []
        optim_name='adam'
        for _ in range(config.adam.epoch):
            adam.step(closure) 
        adam_end_time = time.time()-t1
        if lbfgs:
            optim_name='lbfgs'
            for _ in range(config.lbfgs.epoch):
                lbfgs.step(closure)
            lbfgs_end_time = time.time()-t1-adam_end_time
        t2=time.time()-t1 
        message = '=' * 3 + f'{count}-th training done' + time.strftime("%H:%M:%S", time.gmtime(t2)) + '=' * 10
        logger.info(message)
        print(message)
        time_message = f'AdamTime: {time.strftime("%H:%M:%S", time.gmtime(adam_end_time))}, LBFGSTime: {time.strftime("%H:%M:%S", time.gmtime(lbfgs_end_time))}'
        logger.info(time_message)
        print(time_message)
        points_new = sampler.sample()
        # RAD sampling 
        if config.RAD.use:
            node_rad = rad_sampler.sample()
            residuals = pde.residual(net, node_rad)
            rad_points = sampler.rad_sampler(residuals['in'], node_rad['in'], int(config.RAD.ratio*config.training.batch['in']))
            del residuals, node_rad
            size = rad_points.shape[0]
            ind = np.random.choice(points['in'].shape[0], points['in'].shape[0]-size, replace=False)
            select_points = points['in'][ind]  
            pde.rad_plot(select_points, rad_points,config.path_save, count)
            points['in'] = torch.cat([select_points, rad_points], dim=0)
            points['bd'] = points_new['bd']
            points['init'] = points_new['init']
            del points_new, rad_points, select_points 
        net.save_model(os.path.join(config.path_save, 'net', f'model_{count}.pt'))
        with open(os.path.join(config.path_save, 'train', f'loss_{count}.pkl'), 'wb') as f:
            pickle.dump(loss_save, f)   
        pde.evaluator(net, count)
        total_training_time = time.time() - t_start
    total_time_message = f'TotalTime: {time.strftime("%H:%M:%S", time.gmtime(total_training_time))}, '
    logger.info(total_time_message)
    print(total_time_message)
    loss_plot(config.path_save)

        
    



if __name__ == "__main__":
    # flags.mark_flags_as_required(["overwrite"])
    app.run(main)
