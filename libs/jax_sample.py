import numpy as np
import jax.numpy as jnp
import jax


class BaseEasySampler:
    def __init__(self, batch, n_devices=1, shard=False):
        self.batch = batch
        self.n_devices = n_devices
        self.shard = shard

    def __iter__(self):
        return self

    def __next__(self):
        batch = self.sample()
        if self.shard or self.n_devices > 1:
            batch = self._shard(batch)
        return batch

    def sample(self, **args):
        raise NotImplementedError

    def _shard(self, data):
        """Reshape dict of arrays to (n_devices, B//n_devices, ...)."""
        sharded = {}
        for k, v in data.items():
            if v.shape[0] % self.n_devices != 0:
                new_size = (v.shape[0] // self.n_devices) * self.n_devices
                v = v[:new_size]
            B = v.shape[0]
            shape = (self.n_devices, B // self.n_devices) + v.shape[1:]
            sharded[k] = v.reshape(shape)
        return sharded


class TimeSpaceEasySampler(BaseEasySampler):
    """Samples points in the domain, on boundaries, and on initial surface."""

    def __init__(self, axeslim, tlim, batch, n_devices=1, key=jax.random.PRNGKey(0),
                 shard=False):
        super().__init__(batch, n_devices, shard)
        self.axeslim = axeslim
        self.tlim = tlim
        self.dim = len(axeslim)
        self.key = key
        self._rng = np.random.RandomState(0)

    def sample(self):
        size = self.batch
        points = {}

        # Interior domain points
        size_in = size['in']
        node_in = np.zeros((size_in, self.dim + 1))
        node_in[:, 0] = self._rng.rand(size_in) * (self.tlim[1] - self.tlim[0]) + self.tlim[0]
        for i in range(self.dim):
            l, r = self.axeslim[i]
            node_in[:, i + 1] = self._rng.rand(size_in) * (r - l) + l
        points['in'] = node_in

        # Boundary points (spatial boundaries at random times)
        size_bd = size['bd']
        bd_num = self._rng.randint(0, 2 * self.dim, size=size_bd)
        node_bd_list = []
        for i in range(2 * self.dim):
            idx = np.where(bd_num == i)[0]
            num = len(idx)
            if num == 0:
                continue
            m, n = i // 2, i % 2
            curr = np.zeros((num, self.dim + 1))
            curr[:, 0] = self._rng.rand(num) * (self.tlim[1] - self.tlim[0]) + self.tlim[0]
            for j in range(self.dim):
                if j != m:
                    l, r = self.axeslim[j]
                    curr[:, j + 1] = self._rng.rand(num) * (r - l) + l
                else:
                    curr[:, j + 1] = self.axeslim[m][n]
            node_bd_list.append(curr)
        if node_bd_list:
            points['bd'] = np.concatenate(node_bd_list, axis=0)
        else:
            points['bd'] = np.zeros((0, self.dim + 1))

        # Initial condition points (t=0)
        size_init = size['init']
        node_init = np.zeros((size_init, self.dim + 1))
        node_init[:, 0] = 0.0
        for i in range(self.dim):
            l, r = self.axeslim[i]
            node_init[:, i + 1] = self._rng.rand(size_init) * (r - l) + l
        points['init'] = node_init

        return points

    def rad_sampler(self, residual, points, num_outputs, key=None):
        """Residual-based Adaptive Distribution (RAD) sampling."""
        r = np.asarray(residual).ravel()
        p = np.asarray(points)
        err = r ** 2
        err_sum = err.sum()
        if err_sum < 1e-10:
            prob = np.ones(len(err)) / len(err)
        else:
            prob = err / err_sum
        ind = np.random.choice(len(err), size=num_outputs, replace=False, p=prob)
        return p[ind]
