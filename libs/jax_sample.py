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


class IrregularHoleSampler(BaseEasySampler):
    """Sampler for (-1, 1)^2 with an off-center circular hole."""

    def __init__(self, tlim, batch, center=(-0.3, 0.2), r0=0.25, n_devices=1,
                 seed=0, shard=False):
        super().__init__(batch, n_devices, shard)
        self.tlim = tlim
        self.center = np.asarray(center, dtype=float)
        self.r0 = float(r0)
        self._rng = np.random.RandomState(seed)

    def _sample_time(self, size):
        return self._rng.rand(size) * (self.tlim[1] - self.tlim[0]) + self.tlim[0]

    def _inside_domain(self, xy):
        in_square = np.all((xy > -1.0) & (xy < 1.0), axis=1)
        outside_hole = np.sum((xy - self.center) ** 2, axis=1) > self.r0 ** 2
        return in_square & outside_hole

    def _sample_spatial(self, size):
        chunks = []
        remaining = size
        while remaining > 0:
            proposal = self._rng.uniform(-1.0, 1.0, size=(max(remaining * 2, 16), 2))
            accepted = proposal[self._inside_domain(proposal)]
            if accepted.size == 0:
                continue
            take = min(remaining, accepted.shape[0])
            chunks.append(accepted[:take])
            remaining -= take
        return np.concatenate(chunks, axis=0)

    def _sample_boundary(self, size):
        node = np.zeros((size, 3))
        node[:, 0] = self._sample_time(size)
        pieces = self._rng.randint(0, 5, size=size)

        for piece in range(5):
            idx = np.where(pieces == piece)[0]
            if len(idx) == 0:
                continue

            if piece < 4:
                vals = self._rng.uniform(-1.0, 1.0, size=len(idx))
                if piece == 0:
                    node[idx, 1] = -1.0
                    node[idx, 2] = vals
                elif piece == 1:
                    node[idx, 1] = 1.0
                    node[idx, 2] = vals
                elif piece == 2:
                    node[idx, 1] = vals
                    node[idx, 2] = -1.0
                else:
                    node[idx, 1] = vals
                    node[idx, 2] = 1.0
            else:
                theta = self._rng.uniform(0.0, 2.0 * np.pi, size=len(idx))
                node[idx, 1] = self.center[0] + self.r0 * np.cos(theta)
                node[idx, 2] = self.center[1] + self.r0 * np.sin(theta)

        return node

    def sample(self):
        points = {}

        size_in = self.batch["in"]
        points["in"] = self.sample_interior(size_in)

        points["bd"] = self._sample_boundary(self.batch["bd"])

        size_init = self.batch["init"]
        xy_init = self._sample_spatial(size_init)
        points["init"] = np.column_stack([np.zeros(size_init), xy_init])
        return points

    def sample_interior(self, size):
        xy = self._sample_spatial(size)
        return np.column_stack([self._sample_time(size), xy])


class LShapeSampler(BaseEasySampler):
    """Sampler for Omega_L = [-1, 1]^2 \\ [0, 1]^2."""

    def __init__(self, tlim, batch, n_devices=1, seed=0, shard=False):
        super().__init__(batch, n_devices, shard)
        self.tlim = tlim
        self._rng = np.random.RandomState(seed)

    def _sample_time(self, size):
        return self._rng.rand(size) * (self.tlim[1] - self.tlim[0]) + self.tlim[0]

    def _inside_domain(self, xy):
        x = xy[:, 0]
        y = xy[:, 1]
        in_square = (x > -1.0) & (x < 1.0) & (y > -1.0) & (y < 1.0)
        outside_removed_quadrant = (x < 0.0) | (y < 0.0)
        return in_square & outside_removed_quadrant

    def _sample_spatial(self, size):
        chunks = []
        remaining = size
        while remaining > 0:
            proposal = self._rng.uniform(-1.0, 1.0, size=(max(remaining * 2, 16), 2))
            accepted = proposal[self._inside_domain(proposal)]
            if accepted.size == 0:
                continue
            take = min(remaining, accepted.shape[0])
            chunks.append(accepted[:take])
            remaining -= take
        return np.concatenate(chunks, axis=0)

    def _sample_boundary(self, size):
        node = np.zeros((size, 3))
        node[:, 0] = self._sample_time(size)
        pieces = self._rng.randint(0, 6, size=size)

        for piece in range(6):
            idx = np.where(pieces == piece)[0]
            if len(idx) == 0:
                continue
            vals = self._rng.uniform(-1.0, 1.0, size=len(idx))
            vals01 = self._rng.uniform(0.0, 1.0, size=len(idx))
            vals_neg = self._rng.uniform(-1.0, 0.0, size=len(idx))

            if piece == 0:
                node[idx, 1] = -1.0
                node[idx, 2] = vals
            elif piece == 1:
                node[idx, 1] = vals
                node[idx, 2] = -1.0
            elif piece == 2:
                node[idx, 1] = 1.0
                node[idx, 2] = vals_neg
            elif piece == 3:
                node[idx, 1] = vals_neg
                node[idx, 2] = 1.0
            elif piece == 4:
                node[idx, 1] = 0.0
                node[idx, 2] = vals01
            else:
                node[idx, 1] = vals01
                node[idx, 2] = 0.0

        return node

    def sample(self):
        points = {}

        size_in = self.batch["in"]
        points["in"] = self.sample_interior(size_in)

        points["bd"] = self._sample_boundary(self.batch["bd"])

        size_init = self.batch["init"]
        xy_init = self._sample_spatial(size_init)
        points["init"] = np.column_stack([np.zeros(size_init), xy_init])
        return points

    def sample_interior(self, size):
        xy = self._sample_spatial(size)
        return np.column_stack([self._sample_time(size), xy])
