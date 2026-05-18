#####SV - PINN: Example 2 - 2D Poisson Equation#####
import jax
import jax.numpy as jnp
from jinnax import nn as nn
from jinnax import data as jd
import pickle
import pandas as pd
import os
import gc
jax.config.update("jax_default_matmul_precision", "highest")
jax.config.update("jax_enable_x64", True)

#Parameters of the simulation
L = [1,1]
N = 128
d = 2
bsize = 25000

# Train Matern
for tau in [1,0]:
    for opt in ['LBFGS','GD']:
        for a in [1,10,25,50,75]:
            for rep in range(3):
                fn = 'Example2/Final_Example2_Poisson_2D_sigma' + str(tau) + '_a' + str(a) + '_opt' + opt + '_rep' + str(rep)
                if not os.path.isfile(fn +  '_epoch' + str(5000).rjust(6, '0') + '.pickle'):
                    if opt != 'LBFGS' or tau == 1:
                        #Solution
                        def u(x):
                            x1 = x[:,0].reshape((x.shape[0],1))
                            x2 = x[:,1].reshape((x.shape[0],1))
                            envelope = jnp.sin(jnp.pi * x1) * jnp.sin(jnp.pi * x2)    # zero on all edges
                            content  = ( jnp.sin(a * (x1 + x2))
                                       + jnp.sin(2.0 * jnp.pi * x1)
                                       + jnp.cos(3.0 * jnp.pi * x2) )
                            return envelope * content
                        #Generate data
                        test_data = jd.generate_PINNdata(u = u,xl = [0,0],xu = L,Ns = 2*N,d = 2)
                        #Laplacian of solution
                        def upp(x1,x2):
                            # envelope E
                            E = jnp.sin(jnp.pi * x1) * jnp.sin(jnp.pi * x2)
                            # content C
                            C = ( jnp.sin(a * (x1 + x2))
                                + jnp.sin(2.0 * jnp.pi * x1)
                                + jnp.cos(3.0 * jnp.pi * x2) )
                            # First derivatives of envelope
                            Ex1 = jnp.pi * jnp.cos(jnp.pi * x1) * jnp.sin(jnp.pi * x2)
                            Ex2 = jnp.pi * jnp.sin(jnp.pi * x1) * jnp.cos(jnp.pi * x2)
                            # Second derivatives of envelope
                            Ex1x1 = - (jnp.pi**2) * E
                            Ex2x2 = - (jnp.pi**2) * E
                            lapE = Ex1x1 + Ex2x2   # = -2 pi^2 E
                            # First derivatives of content
                            Cx1 = a * jnp.cos(a * (x1 + x2)) + 2*jnp.pi * jnp.cos(2*jnp.pi * x1)
                            Cx2 = a * jnp.cos(a * (x1 + x2)) - 3*jnp.pi * jnp.sin(3*jnp.pi * x2)
                            # Second derivatives of content
                            Cx1x1 = -a**2 * jnp.sin(a*(x1 + x2)) - (2*jnp.pi)**2 * jnp.sin(2*jnp.pi * x1)
                            Cx2x2 = -a**2 * jnp.sin(a*(x1 + x2)) - (3*jnp.pi)**2 * jnp.cos(3*jnp.pi * x2)
                            lapC = Cx1x1 + Cx2x2
                            # Gradient dot-product
                            grad_dot = Ex1 * Cx1 + Ex2 * Cx2
                            # Final Laplacian
                            lap_u = C * lapE + E * lapC + 2.0 * grad_dot
                            return lap_u
                        #Operator
                        def oper(u,x):
                            def u_scalar(x):
                                val = u(x[None, :])   # (1,) or (1,1)
                                return jnp.squeeze(val)[()]   # ()-scalar
                            H = jax.hessian(u_scalar)
                            return jax.vmap(lambda xi: jnp.trace(H(xi)))(x).reshape((x.shape[0],1)) - upp(x[:,0:1],x[:,1:2])
                        #Train
                        data = jd.generate_PINNdata(u = u,xl = [0,0],xu = L,Nc = N,d = 2)
                        fargs = [L]
                        q = 0
                        width = [2] + [64] + 3 * [512] + [1]
                        res = nn.train_SV_PINN(data,width,oper,test_data,resample = False,d = d,N = N,L = L,alpha = 1,kappa = 1,tau = tau,bsize = bsize,epochs = 5001,at_each = 5000,epoch_print = 5000,save = True,file_name = fn,lr = 0.001,exp_decay = True,transition_steps = 100,decay_rate = 0.9,mlp = True,ftype = 'daff',fargs = fargs,q = q,key = rep,opt = opt,float64 = True)
                        del res, data, test_data
                        gc.collect()

#Plot solution
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
from matplotlib.colors import TwoSlopeNorm
import matplotlib.colors as colors
import dill
a = [1,10,25,50,75]
titles = ['a = ' + str(i) for i in a]
fig, axes = plt.subplots(nrows = len(a),ncols = 3, figsize = (1.2*3*6.4,len(a)*6.4),constrained_layout = True) #
path = 'Example2_final/'
norm = colors.Normalize(vmin = -3.05,vmax = 3.05)
cmap = cm.viridis

for v in range(len(a)):
    ax = axes[v,:]
    #Solution
    def u(x):
        x1 = x[:,0].reshape((x.shape[0],1))
        x2 = x[:,1].reshape((x.shape[0],1))
        envelope = jnp.sin(jnp.pi * x1) * jnp.sin(jnp.pi * x2)    # zero on all edges
        content  = ( jnp.sin(a[v] * (x1 + x2))
                   + jnp.sin(2.0 * jnp.pi * x1)
                   + jnp.cos(3.0 * jnp.pi * x2) )
        return envelope * content
    test_data = jd.generate_PINNdata(u = u,xl = [0,0],xu = L,Ns = 2*N,d = 2)
    ax[0].imshow(test_data['usensor'].reshape(2*N,2*N), aspect='auto',norm = norm,cmap = cmap)
    ax[0].set_title('Solution ' + titles[v],fontweight = 'bold',fontsize = 24)
    ax[0].axis('off')
    #Predicted
    fn = path + 'Final_Example2_Poisson_2D_sigma1_a' + str(a[v]) + '_optLBFGS_rep0'
    forward = dill.load(open(fn + '_config.pickle', "rb"),fix_imports=True, encoding="latin1")['forward']
    params = dill.load(open(fn + '_epoch' + str(5000).rjust(6, '0') + '.pickle', "rb"),fix_imports=True, encoding="latin1")['params']
    pred = forward(test_data['sensor'],params['net']).reshape((2*N,2*N))
    true = test_data['usensor'].reshape((2*N,2*N))
    error = true - pred
    ax[1].imshow(pred, aspect='auto',norm = norm,cmap = cmap)
    ax[1].set_title('Approximation by SV-PINN ' + titles[v],fontweight = 'bold',fontsize = 24)
    ax[1].axis('off')
    #Error
    l = jnp.maximum(-jnp.min(error),jnp.max(error))
    im = ax[2].imshow(error, aspect='auto',cmap = cm.seismic,norm = TwoSlopeNorm(vmin = -l,vcenter = 0,vmax = l))
    ax[2].set_title('Error ' + titles[v],fontweight = 'bold',fontsize = 24)
    ax[2].axis('off')
    cbar = fig.colorbar(im,ax = ax[2],pad = 0.01, format='%.1e')
    cbar.ax.tick_params(labelsize=14)
    cbar.locator = ticker.MaxNLocator(nbins = 8)
    cbar.update_ticks()

plt.savefig('sol_ex2.pdf')
plt.close()
