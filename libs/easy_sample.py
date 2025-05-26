import torch 
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
class BaseEasySampler(Dataset):
    def __init__(self, batch, dev):
        self.dev = dev
        self.batch = batch 
    def __iter__(self):
        """返回自身作为迭代器"""
        return self
    
    def __next__(self):
        """生成一批数据"""
        points = self.sample()
        return points
    def sample(**args):
        raise NotImplementedError("Subclasses should implement this!")

        
class TimeEasySampler(BaseEasySampler):
    def __init__(self, tlim, dev, batch):
        super().__init__(batch, dev)
        self.tlim = tlim
    def sample(self):
        size = self.batch
        points = {}
        size_in = size['in']
        node_in = torch.zeros((size_in, 1)).to(self.dev)
        leftlim = self.tlim[0]
        rightlim = self.tlim[1]
        length = rightlim - leftlim
        node_in[:, 0] = torch.rand((size_in, ))*length+torch.ones((size_in, )) * leftlim
        node_in = node_in.to(device=self.dev)
        points['in'] = node_in
        return points


class TimeSpaceEasySampler(BaseEasySampler):
    def __init__(self, axeslim, tlim, dev, batch):
        super().__init__(batch, dev)
        self.axeslim = axeslim
        self.tlim = tlim
        self.dim = len(axeslim)
    def sample(self):
        size = self.batch
        points = {}
        #sample in the domain
        size_in = size['in']
        node_in = torch.zeros((size_in, self.dim+1)).to(self.dev)
        leftlim = self.tlim[0]
        rightlim = self.tlim[1]
        length = rightlim - leftlim
        node_in[:, 0] = torch.rand((size_in, ))*length+torch.ones((size_in, )) * leftlim
        for i in range(self.dim):
            leftlim = self.axeslim[i][0]
            rightlim = self.axeslim[i][1]
            length = rightlim - leftlim
            node_in[:, i+1] = torch.rand((size_in, ))*length+torch.ones((size_in, )) * leftlim
        node_in = node_in.to(device=self.dev)
        points['in'] = node_in 
        #sample in the boundary
        size_bd = size['bd']
        bd_num = torch.randint(low=0, high=2*self.dim, size=(size_bd,))
        node_bd = list(range(2*self.dim))
        for i in range(2*self.dim):
            ind = bd_num[bd_num == i]
            num = bd_num[ind].shape[0]
            m, n = i//2, i % 2
            node_bd[i] = torch.rand([num, 1])*(self.tlim[1] - self.tlim[0])+torch.ones([num, 1])*self.tlim[0]
            for j in range(self.dim):
                if j != m:
                    node_bd[i] = torch.cat([node_bd[i],
                                            torch.rand([num, 1])*(self.axeslim[j][1] - self.axeslim[j][0])+torch.ones([num, 1])*self.axeslim[j][0]], dim=1)
                else:
                    node_bd[i] = torch.cat([node_bd[i],
                                            torch.ones([num, 1]) * self.axeslim[m][n]], dim=1)
        node_bd = torch.cat(node_bd, dim=0).to(device=self.dev)
        points['bd'] = node_bd
        #sample in the initial condition
        size_init = size['init']
        node_init = torch.zeros((size_init, self.dim+1)).to(self.dev)
        node_init[:, 0] = torch.zeros((size_init,), device=self.dev)
        for i in range(self.dim):
            leftlim = self.axeslim[i][0]
            rightlim = self.axeslim[i][1]
            length = rightlim - leftlim
            node_init[:, i+1] = torch.rand((size_init, ))*length+torch.ones((size_init, )) * leftlim
        points['init'] = node_init
        return points
    def rad_sampler(self, residual, points, num_outputs):
        """
        RAD sampling based on the residual (N*1 tensor), choose nums_outputs points as outputs, works on numpy
        """
        node = points.detach().cpu().numpy() 
        residual = residual.detach().cpu().numpy()
        err = np.power(residual, 2)
        err_normal = err / np.sum(err)
        size = node.shape[0] 
        ind = np.random.choice(size, num_outputs, replace=False, p=err_normal.flatten())
        points_output = points[ind]
        return points_output

if  __name__ == '__main__':
    sampler = iter(TimeSpaceEasySampler(axeslim=[[0, 1]], tlim=[0, 1], dev='cpu', batch={'in': 10000, 'bd': 100, 'init': 100}))
    points = next(sampler)
    residual = torch.exp(-(points['in'][:, 0] -0.5)**2*10-(points['in'][:, 1] -0.5)**2*10)**2 
    ratio=0.1
    points_rad = sampler.rad_sampler(residual, points['in'], ratio)
    plt.scatter(points_rad[:, 0], points_rad[:, 1],c='r', marker='.', s=np.ones_like(points_rad[:, 0]), alpha=1.0,)
    plt.scatter(points['in'][:, 0], points['in'][:, 1],c='b', marker='.', s=np.ones_like(points['in'][:, 0]), alpha=0.3,) 
    # plt.scatter(points['bd'][:, 0], points['bd'][:, 1])
    # plt.scatter(points['init'][:, 0], points['init'][:, 1])
    plt.show()