from libs import *
from pymittagleffler import *

class DWBurgers(PINN):
    '''
    Case I: compute the Initial Value Problem (IVP) given analytic exact solution
    Notice that the form of x across functions should be torch tensor, try to minimize the interaction between gpu and cpu.
    '''
    def __init__(self, config):
        super().__init__(config)
        self.al = config.al
        self.datafile = config.datafile
        self.beta = config.beta
        self.xlim=config.xlim 
        self.tlim=config.tlim
        if 'GJ' in config.method:
            GJ = config.GJ
            nums = GJ.nums
            quad_t, quad_wt = roots_jacobi(nums, 0, 1-self.al)
            self.quad_t = (quad_t + 1) / 2
            self.quad_w = quad_wt * (1 / 2) ** (self.al-1)
    def u_net(self, net, points):
        # compute the exact solution of the IVP
        return net(points)
    def source(self, points): #points N*2 
        return torch.zeros_like(points[:, 0:1])
    def residual(self, net, points_all):
        """
        compute frac diff is integrated here since we may change the hyperparameters or realizations many times.
        """
        losses = {}
        # generate residual in the domain
        points = points_all['in']
        dt, dx, dxx = self.frac_diff(net, points) 
        val = self.u_net(net, points)
        losses['in'] = dt+val*dx - 0.01/torch.pi*dxx #N*1 
        # generate residual on boundary 
        points = points_all['bd']
        pred = self.u_net(net,points)
        losses['bd'] = pred
        # generate residual on initial condition
        points = points_all['init']
        pred = -torch.sin(np.pi*points[:, 1:2]).to(device=self.config.dev) #N*1
        losses['init'] = self.u_net(net,points) - pred #N*1
        # generate residual on 1st derivative initial condition
        points = points_all['init']
        pred = self.beta*torch.sin(np.pi*points[:, 1:2]).to(device=self.config.dev) #N*1
        points.requires_grad = True
        val = self.u_net(net,points)
        dt  = torch.autograd.grad(outputs=val, inputs=points, grad_outputs=torch.ones_like(val),retain_graph=True,    create_graph=True)[0][:,0:1]#N*1 
        losses['init_dt'] =dt-pred 
        return losses 
    def frac_diff(self, net, points):
        config = self.config
        points.requires_grad = True
        val = self.u_net(net,points) #N*1
        d = torch.autograd.grad(outputs=val,
                                inputs=points,
                                grad_outputs=torch.ones_like(val),
                                retain_graph=True,
                                create_graph=True)[0]
        dt = d[:, 0:1] #N*1
        dx = d[:, 1:2] #N*1
        dxx = torch.autograd.grad(inputs=points,
                                    outputs=dx,
                                    grad_outputs=torch.ones_like(dx),
                                    retain_graph=True,
                                    create_graph=True)[0][:, 1:2] #N*1
        points.detach() 
        points.requires_grad = False
        if config.method == 'MC-I':
            # Monte Carlo Integration
            MC = config.MC
            al = self.al
            nums = MC.nums
            lens = len(points)
            eps = MC.eps 
            coeff = sp.gamma(2-al) 
            taus = beta.rvs(2-al,1,size=nums)
            taus = torch.from_numpy(taus).to(device=config.dev)
            # compute u'(0,x)
            t0 = torch.cat((torch.zeros_like(points[:, 0:1]), points[:,1:]), dim=1).to(device=config.dev)
            t0.requires_grad = True
            val0 = self.u_net(net,t0) #N*1
            dt0 = torch.autograd.grad(outputs=val0, inputs=t0, grad_outputs=torch.ones_like(val0),retain_graph=True,
                            create_graph=True)[0][:,0:1]#N*1
            t0.requires_grad = False
            # compute u'(t-t tau,x) 
            t = points[:, 0:1] #N*1
            taus = taus.reshape(-1,1) #M*1
            x = points[:, 1] #N*1
            t_tau = t.T * taus.reshape(-1,1) #M*N 
            t_t_tau = points[:, 0]-t_tau #M*N 
            t_tau_max = torch.max(t_tau, torch.tensor([eps], device=config.dev)) #M*N
            new_x = x.unsqueeze(0).expand(nums, lens) #M*N
            new_points = torch.stack((t_t_tau,new_x), dim=2) #M*N*2 
            new_points.detach()
            new_points.requires_grad = True 
            valttau = self.u_net(net,new_points)# f(t-ttau) M*N*1
            dttau =  torch.autograd.grad(outputs=valttau, inputs=new_points, grad_outputs=torch.ones_like(valttau),retain_graph=True, create_graph=True)[0][:,:, 0:1]#M*N*1 
            part1 = torch.mean((dt-dttau)/t_tau_max.unsqueeze(-1), dim=0) # N*1
            part1 = part1*(al-1.0)/(2.0-al)*torch.pow(points[:, 0:1], 2-al) 
            part2 = (dt-dt0)*torch.pow(points[:, 0:1],1-al) 
            dtal = (part1+part2)/coeff #N*1 
            new_points.requires_grad=False  
            return dtal, dx, dxx
        if config.method =='MC-II': 
            MC = config.MC
            al = self.al
            nums = MC.nums
            lens = len(points)
            eps = MC.eps 
            coeff = sp.gamma(2-al) 
            taus = beta.rvs(2-al,1,size=nums)
            taus = torch.from_numpy(taus).to(device=config.dev)
             # compute u'(0,x)
            t0 = torch.cat((torch.zeros_like(points[:, 0:1]), points[:,1:]), dim=1).to(device=config.dev)
            t0.requires_grad = True
            val0 = self.u_net(net,t0) #N*1
            dt0 = torch.autograd.grad(outputs=val0, inputs=t0, grad_outputs=torch.ones_like(val0),retain_graph=True,
                            create_graph=True)[0][:,0:1]#N*1
            t0.requires_grad = False
            # compute u'(t-t tau,x) 
            t = points[:, 0:1] #N*1
            taus = taus.reshape(-1,1) #M*1
            x = points[:, 1] #N*1
            t_tau = t.T * taus.reshape(-1,1) #M*N
            t_t_tau = points[:, 0]-t_tau #M*N 
            t_tau_max = torch.max(t_tau, torch.tensor([eps], device=config.dev)) #M*N
            new_x = x.unsqueeze(0).expand(nums, lens) #M*N
            new_points = torch.stack((t_t_tau,new_x), dim=2) #M*N*2 
            new_points.detach()
            # new_points.requires_grad = True 
            val2 = self.u_net(net,new_points)# f(t-ttau) M*N*1
            val3 = t_tau.unsqueeze(-1) * dt.unsqueeze(0) #M*N*1
            part1 = al*(al-1)/(2-al)*torch.pow(points[:, 0:1], 2-al)*torch.mean((val-val2-val3)/(torch.pow(t_tau_max.unsqueeze(-1), 2)),dim=0) #N*1 
            part2 = (al-1)*(val-val0-points[:, 0:1]*dt)/torch.pow(points[:, 0:1],al) #N*1
            part3 = (dt-dt0)/(torch.pow(points[:, 0:1],al-1)) #N*1
            dtal = (part3-part2-part1)/coeff
            return dtal, dx, dxx
        if config.method == 'GJ-I':
            al = self.al
            nums = config.GJ.nums
            lens = len(points)
            taus = torch.from_numpy(self.quad_t).to(device=config.dev) #M
            quad_w = torch.from_numpy(self.quad_w).to(device=config.dev) #M
            quad_w = quad_w.unsqueeze(-1).unsqueeze(-1) #M*1*1
            coeff = sp.gamma(2-al) 
            # compute u'(0,x)
            t0 = torch.cat((torch.zeros_like(points[:, 0:1]), points[:,1:]), dim=1).to(device=config.dev)
            t0.requires_grad = True
            val0 = self.u_net(net,t0) #N*1
            dt0 = torch.autograd.grad(outputs=val0, inputs=t0, grad_outputs=torch.ones_like(val0),retain_graph=True,
                            create_graph=True)[0][:,0:1]#N*1
            t0.requires_grad = False
            # compute u'(t-t tau,x) 
            t = points[:, 0:1] #N*1
            taus = taus.unsqueeze(-1) #M*1
            x = points[:, 1] #N*1
            t_tau = t.T * taus #M*N 
            t_t_tau = points[:, 0]-t_tau #M*N 
            new_x = x.unsqueeze(0).expand(nums, lens) #M*N
            new_points = torch.stack((t_t_tau,new_x), dim=2) #M*N*2 
            new_points.detach()
            new_points.requires_grad = True 
            valttau = self.u_net(net,new_points)# f(t-ttau) M*N*1
            dttau =  torch.autograd.grad(outputs=valttau, inputs=new_points, grad_outputs=torch.ones_like(valttau),retain_graph=True, create_graph=True)[0][:,:, 0:1]#M*N*1 
            part1 = torch.sum(quad_w*(dt-dttau)/t_tau.unsqueeze(-1), dim=0) # N*1
            part1 = part1*(al-1.0)*torch.pow(points[:, 0:1], 2-al) 
            part2 = (dt-dt0)*torch.pow(points[:, 0:1],1-al) 
            dtal = (part1+part2)/coeff #N*1 
            new_points.requires_grad=False  
            return dtal, dx, dxx
        if config.method =='GJ-II':
            al = self.al
            coeff = sp.gamma(2-al) 
            nums = config.GJ.nums
            lens = len(points)
            taus = torch.from_numpy(self.quad_t).to(device=config.dev) #M
            quad_w = torch.from_numpy(self.quad_w).to(device=config.dev) #M
            quad_w = quad_w.unsqueeze(-1).unsqueeze(-1) #M*1*1
            # compute u'(0, x)
            t0 = torch.cat((torch.zeros_like(points[:, 0:1]), points[:,1:]), dim=1).to(device=config.dev)
            t0.requires_grad = True
            val0 = self.u_net(net,t0) #N*1
            dt0 = torch.autograd.grad(outputs=val0, inputs=t0, grad_outputs=torch.ones_like(val0),retain_graph=True,
                            create_graph=True)[0][:,0:1]#N*1
            t0.requires_grad = False
            # compute u'(t-t tau,x) 
            t = points[:, 0] #N
            x = points[:, 1] #N
            t = t.unsqueeze(0).unsqueeze(-1)  #1*N*1
            # compute ttau
            t = points[:, 0:1] #N*1
            taus = taus.unsqueeze(-1) #M*1
            x = points[:, 1] #N*1
            t_tau = t.T * taus #M*N 
            t_t_tau = points[:, 0]-t_tau #M*N 
            new_x = x.unsqueeze(0).expand(nums, lens) #M*N
            new_points = torch.stack((t_t_tau,new_x), dim=2) #M*N*2 
            new_points.detach()
            # new_points.requires_grad = True 
            val2 = self.u_net(net,new_points)# f(t-ttau) M*N*1
            val3 = t_tau*(dt.squeeze(-1)) #M*N  !!!
            part1 = al*(al-1)*torch.pow(points[:, 0:1], 2-al)*torch.sum(quad_w*((val-val2-val3.unsqueeze(-1))/(torch.pow(t_tau.unsqueeze(-1), 2))),dim=0) #N*1 !!!
            part2 = (al-1)*(val-val0-points[:, 0:1]*dt)/torch.pow(points[:, 0:1],al) #N*1
            part3 = (dt-dt0)/(torch.pow(points[:, 0:1],al-1)) #N*1
            dtal = (part3-part2-part1)/coeff
            return dtal, dx, dxx

    def exact(self, points):
        data = np.load(self.datafile)
        t = data['t'] #(200,) 
        x = data['x']#(100,)
        u = data['u'] #(200,100)
        return u.reshape(-1,1) # 40000*1
    def gen_err(self, net, points):
        pass

    def evaluator(self, net, count):
        config = self.config
        data = np.load(self.datafile)
        t = data['t'] #(200,)
        x = data['x']#(100,)
        u = data['u'].T #(100,200)
        mesh_t, mesh_x = np.meshgrid(t,x)
        points = np.stack([mesh_t.flatten(), mesh_x.flatten()], axis=1) #N*2
        points_tensor = torch.tensor(points, device=config.dev)
        val = self.u_net(net, points_tensor).detach().cpu().numpy() #N*1
        exact = u.reshape(-1,1) #N*1 
        err = np.sqrt(np.sum(np.power(val-exact,2))/np.sum(np.power(exact,2)))# N*1
        err_plt = np.abs(val-exact)
        err_max =np.max(err_plt) 
        # plot absolute point-wise error
        fig, ax = plt.subplots(layout='constrained', figsize=(6.4, 4.8))
        plot = ax.pcolormesh(mesh_t, mesh_x, err_plt.reshape(mesh_x.shape), shading='gouraud', cmap='jet', vmin=0, vmax=np.max(err_plt))
        fig.colorbar(plot, ax=ax, format="%1.1e")
        ax.set_title(f'{count+1}-th error is {round(err,4)}')
        ax.set_xlabel('t')
        ax.set_ylabel('x')
        fig.savefig(os.path.join(config.path_save, 'img', f'{count}_abs.png'), dpi=100)
        plt.close(fig)
        # val plot
        fig, ax = plt.subplots(layout='constrained', figsize=(6.4, 4.8))
        plot = ax.pcolormesh(mesh_t, mesh_x, val.reshape(mesh_x.shape), shading='gouraud', cmap='jet', vmin=np.min(exact), vmax=np.max(exact))
        fig.colorbar(plot,  ax=ax, format="%1.1e")
        ax.set_xlabel('t')
        ax.set_ylabel('x')
        fig.savefig(os.path.join(config.path_save, 'img', f'{count}_sol.png'), dpi=100)
        plt.close(fig)
        # exact plot
        if count == 0:
            fig, ax = plt.subplots(layout='constrained', figsize=(6.4, 4.8))
            plot = ax.pcolormesh(mesh_t, mesh_x, exact.reshape(mesh_x.shape), shading='gouraud', cmap='jet', vmin=np.min(exact), vmax=np.max(exact))
            fig.colorbar(plot, ax=ax, format="%1.1e")
            ax.set_xlabel('t')
            ax.set_ylabel('x')
            fig.savefig(os.path.join(config.path_save, 'img', 'exact.png'), dpi=100)
            plt.close(fig)
        return err 
        
    def rad_plot(self, points, rad_points, path_save, count):
        ts,te,xs,xe = self.tlim[0], self.tlim[1], self.xlim[0][0], self.xlim[0][1]
        fig, ax = plt.subplots(layout='constrained', figsize=(6.4, 4.8))
        ax.set_xlim(ts - (te - ts) * 0.05, te + (te - ts) * 0.20)
        ax.set_ylim(xs - (xe - xs) * 0.05, xe + (xe - xs) * 0.20)
        points = points.detach().cpu().numpy()
        rad_points = rad_points.detach().cpu().numpy()
        ax.scatter(points[:, 0], points[:, 1], c='b', marker='.', s=np.ones_like(points[:, 0]), alpha=0.3,
                   label=f'uni')
        ax.scatter(rad_points[:, 0], rad_points[:, 1], c='r', marker='.', s=np.ones_like(rad_points[:, 0]), alpha=1.0,
                   label=f'RAD')
        ax.legend(loc='upper right')
        ax.set_xlabel('t')
        ax.set_ylabel('x')
        fig.savefig(os.path.join(path_save, 'img', f'{count}_node.png'), dpi=100)
        plt.close(fig)

    
