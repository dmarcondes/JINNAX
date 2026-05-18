#####SV-PINN: Example 3 - 2D Poisson Equation with variable coefficients#####
import jax
import jax.numpy as jnp
from jinnax import nn as nn
from jinnax import data as jd
import os
jax.config.update("jax_default_matmul_precision", "highest")
jax.config.update("jax_enable_x64", True)

#Parameters of the simulation
L = [1,1]
N = 128
d = 2
bsize = 25000
pi = jnp.pi
ku = 10.0 * pi
ka = 20.0 * pi
beta = 0.75

for tau in [1,0]:
    for opt in ['LBFGS','GD']:
        for rep in range(3):
            if opt != 'LBFGS' or tau == 1:
                fn = 'Example3/Final_Example3_PoissonVariable_2D' + '_sigma' + str(tau) + '_opt' + opt + '_rep' + str(rep)
                if not os.path.isfile(fn +  '_epoch' + str(5000).rjust(6, '0') + '.pickle'):
                    #Solution, source and coefficients
                    def u(xy):
                        x = xy[:,0].reshape((xy.shape[0],1))
                        y = xy[:,1].reshape((xy.shape[0],1))
                        return jnp.sin(ku * x) * jnp.sin(ku * y)
                    def ux(xy):
                        x = xy[:,0].reshape((xy.shape[0],1))
                        y = xy[:,1].reshape((xy.shape[0],1))
                        return ku * jnp.cos(ku * x) * jnp.sin(ku * y)
                    def uy(xy):
                        x = xy[:,0].reshape((xy.shape[0],1))
                        y = xy[:,1].reshape((xy.shape[0],1))
                        return ku * jnp.sin(ku * x) * jnp.cos(ku * y)
                    def uxx(xy):
                        x = xy[:,0].reshape((xy.shape[0],1))
                        y = xy[:,1].reshape((xy.shape[0],1))
                        return -(ku**2) * jnp.sin(ku * x) * jnp.sin(ku * y)
                    def uyy(xy):
                        x = xy[:,0].reshape((xy.shape[0],1))
                        y = xy[:,1].reshape((xy.shape[0],1))
                        return -(ku**2) * jnp.sin(ku * x) * jnp.sin(ku * y)
                    def a(xy):
                        x = xy[:,0].reshape((xy.shape[0],1))
                        y = xy[:,1].reshape((xy.shape[0],1))
                        return 1.0 + beta * jnp.sin(ka * x) * jnp.sin(ka * y)
                    def ax(xy):
                        x = xy[:,0].reshape((xy.shape[0],1))
                        y = xy[:,1].reshape((xy.shape[0],1))
                        return beta * ka * jnp.cos(ka * x) * jnp.sin(ka * y)
                    def ay(xy):
                        x = xy[:,0].reshape((xy.shape[0],1))
                        y = xy[:,1].reshape((xy.shape[0],1))
                        return beta * ka * jnp.sin(ka * x) * jnp.cos(ka * y)
                    def f(xy):
                        return -(ax(xy)*ux(xy) + ay(xy)*uy(xy) + a(xy)*(uxx(xy) + uyy(xy)))
                    #Generate data
                    test_data = jd.generate_PINNdata(u = u,xl = [0,0],xu = L,Ns = 2*N,d = 2)
                    xy = test_data['sensor']
                    #Operator
                    def oper(u,xy):
                        x = xy[:,0].reshape((xy.shape[0],1))
                        y = xy[:,1].reshape((xy.shape[0],1))
                        ux = lambda x,y : jax.grad(lambda x,y : jnp.sum(u(jnp.append(x,y,1))),0)(x,y)
                        uy = lambda x,y : jax.grad(lambda x,y : jnp.sum(u(jnp.append(x,y,1))),1)(x,y)
                        F = lambda x,y: a(jnp.append(x,y,1)) * jnp.append(ux(x,y),uy(x,y),1)
                        Fx = jax.grad(lambda x,y: jnp.sum(F(x,y)[:,0]),0)
                        Fy = jax.grad(lambda x,y: jnp.sum(F(x,y)[:,1]),1)
                        return -(Fx(x,y) + Fy(x,y)) - f(xy)
                    #Train
                    data = jd.generate_PINNdata(u = u,xl = [0,0],xu = L,Nc = N,d = 2)
                    fargs = [L]
                    q = 0
                    width = [2] + [64] + 3 * [512] + [1]
                    res = nn.train_SV_PINN(data,width,oper,test_data,resample = False,d = d,N = N,L = L,alpha = 1,kappa = 1,tau = tau,bsize = bsize,epochs = 5001,at_each = 5000,epoch_print = 10000,save = True,file_name = fn,lr = 0.001,exp_decay = True,transition_steps = 100,decay_rate = 0.9,mlp = True,ftype = 'daff',fargs = fargs,q = q,key = rep,opt = opt,float64 = True)

#Solution
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from matplotlib.colors import TwoSlopeNorm
import dill
def u(xy):
    x = xy[:,0].reshape((xy.shape[0],1))
    y = xy[:,1].reshape((xy.shape[0],1))
    return jnp.sin(ku * x) * jnp.sin(ku * y)

path = 'Example3_final/'
fn = path + 'Final_Example3_PoissonVariable_2D_sigma1_optLBFGS_rep0'
forward = dill.load(open(fn + '_config.pickle', "rb"),fix_imports=True, encoding="latin1")['forward']
params = dill.load(open(fn + '_epoch' + str(5000).rjust(6, '0') + '.pickle', "rb"),fix_imports=True, encoding="latin1")['params']
test_data = jd.generate_PINNdata(u = u,xl = [0,0],xu = L,Ns = 2*N,d = 2)
pred = forward(test_data['sensor'],params['net']).reshape((2*N,2*N))
true = test_data['usensor'].reshape((2*N,2*N))
error = true - pred
norm = colors.Normalize(vmin = -1.05,vmax = 1.05)
cmap = cm.viridis

fig, axes = plt.subplots(nrows = 1,ncols = 3, figsize = (32,32/3),constrained_layout = True) #
axes = axes.flatten()

#True and pred
im = axes[0].imshow(true,aspect = 'auto',norm = norm,cmap = cmap)
axes[0].set_title('Solution',fontweight = 'bold',fontsize = 30)
axes[0].axis('off')
im = axes[1].imshow(pred,aspect = 'auto',norm = norm,cmap = cmap)
axes[1].set_title('Approximation by SV-PINN',fontweight = 'bold',fontsize = 30)
axes[1].axis('off')

#Error
l = jnp.maximum(-jnp.min(error),jnp.max(error))
im = axes[2].imshow(error,aspect = 'auto',cmap = cm.seismic,norm = TwoSlopeNorm(vmin = -l,vcenter = 0,vmax = l))
axes[2].set_title('Error',fontweight = 'bold',fontsize = 30)
axes[2].axis('off')
cbar = fig.colorbar(im,ax = axes.ravel().tolist(),pad = 0.01, format='%.1e')
cbar.ax.tick_params(labelsize=24)
cbar.locator = ticker.MaxNLocator(nbins = 8)
cbar.update_ticks()


plt.savefig('sol_Ex3.pdf',bbox_inches = 'tight',pad_inches = 0.1)
plt.close()
