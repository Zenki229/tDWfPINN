import numpy as np   
import matplotlib.pyplot as plt


data_path = './burgers_150.npz'
data = np.load(data_path)
t = data["t"]
x = data["x"]
u = data["u"]

fig, ax = plt.subplots(layout='constrained', figsize=(6.4, 4.8))
plot = ax.pcolormesh(t, x, u.T, shading='gouraud', cmap='jet')
fig.colorbar(plot, ax=ax, format="%1.1e")
ax.set_title(data_path.split('/')[-1].split('.')[0])
ax.set_xlabel('t')
ax.set_ylabel('x')

fig.savefig(data_path.split('/')[-1].split('.')[0] + '.jpg', dpi=100)