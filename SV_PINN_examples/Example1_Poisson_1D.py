#####SV-PINN: 1D Poisson Equation#####
import jax
import os
import jax.numpy as jnp
from jinnax import nn as nn
from jinnax import data as jd
jax.config.update("jax_default_matmul_precision", "highest")
jax.config.update("jax_enable_x64", True)

#Parameters of the simulation
L = [1]
N = 1024
d = 1
bsize = 25000

for a in [1,25,50,100,150,200,250,300]:
    for type in ['daff','ff','None']:
        for opt in ['LBFGS','GD']:
            for rep in range(3):
                fn = 'Example1/Final_Example1_a' + str(a) +'_type' + str(type) + '_opt' + opt + '_rep' + str(rep)
                if not os.path.isfile(fn +  '_weak_epoch' + str(5000).rjust(6, '0') + '.pickle'):
                    #Solution
                    def u(x):
                        return jnp.sin(2 * jnp.pi * x) + 0.1 * jnp.sin(a * jnp.pi * x) - x * (jnp.sin(2 * jnp.pi) + 0.1 * jnp.sin(a * jnp.pi))
                    #Equation
                    def upp(x):
                        return -4*jnp.pi**2 * jnp.sin(2*jnp.pi*x) - 0.1 * (a*jnp.pi)**2 * jnp.sin(a*jnp.pi*x)
                    def oper(u,x):
                        H = jax.hessian(u)
                        return jax.vmap(lambda xi: jnp.trace(H(xi)))(x).reshape((x.shape[0],1)) - upp(x)
                    #Generate data
                    data = jd.generate_PINNdata(u = u,xl = 0,xu = L,Nb = N,Nc = N)
                    if type == 'daff':
                        data['boundary'] = None
                        fargs = [L]
                        q = 0
                        width = [1] + [64] + 3 * [512] + [1]
                        tau = 0.1
                    elif type == 'ff':
                        fargs = [1,10]
                        q = 4
                        width = [1] + [64] + 3 * [512] + [1]
                        tau = 0.1
                    else:
                        type = None
                        fargs = None
                        q = 4
                        width = [1] + 3 * [512] + [1]
                        tau = 0.1
                    if opt == 'LBFGS':
                        q = 0
                    test_data = jd.generate_PINNdata(u = u,xl = 0,xu = L,Ns = 2*N)
                    #Strong form
                    if opt != 'LBFGS':
                        resS = nn.train_SV_PINN(data,width.copy(),oper,test_data,tau = 0,epochs = 5001,at_each = 5000,epoch_print = 5000,save = True,file_name = fn + '_strong',lr = 0.001,exp_decay = True,transition_steps = 100,decay_rate = 0.9,mlp = True,key = rep,ftype = type,fargs = fargs,q = q,opt = opt,float64 = True)
                    #Weak form
                    resW = nn.train_SV_PINN(data,width.copy(),oper,test_data,resample = False,d = d,N = N,L = L,alpha = 1,kappa = 1,tau = tau,bsize = bsize,epochs = 5001,at_each = 5000,epoch_print = 5000,save = True,file_name = fn + '_weak',lr = 0.001,exp_decay = True,transition_steps = 100,decay_rate = 0.9,mlp = True,ftype = type,fargs = fargs,q = q,key = rep,opt = opt,float64 = True)

#Plot solution
import matplotlib.pyplot as plt
import dill
import matplotlib.ticker as mticker
path = 'Example1_final/'
titles = ['a = ' + str(i) for i in [1,25,50,100,150,200,250,300]]
fig, axes = plt.subplots(nrows = 4,ncols = 4, figsize=(32*0.9,28*0.9)) #
axes = axes.flatten()

a = [1,25,50,100,150,200,250,300]
for i in range(len(a)):
    fn = path + 'Final_Example1_a' + str(a[i]) +'_typedaff_optLBFGS_rep0_weak'
    forward = dill.load(open(fn + '_config.pickle', "rb"),fix_imports=True, encoding="latin1")['forward']
    params = dill.load(open(fn + '_epoch' + str(5000).rjust(6, '0') + '.pickle', "rb"),fix_imports=True, encoding="latin1")['params']
    ax = axes[i]
    def u(x):
        return jnp.sin(2 * jnp.pi * x) + 0.1 * jnp.sin(a[i] * jnp.pi * x) - x * (jnp.sin(2 * jnp.pi) + 0.1 * jnp.sin(a[i] * jnp.pi))
    grid = jnp.linspace(0,1,2048).reshape((2048,1))
    ax.plot(grid,u(grid),color = 'red')
    ax.plot(grid,forward(grid,params['net']),color = 'blue',ls = '--')
    ax.set_title('Solution ' + titles[i], fontweight='bold', fontsize = 28)
    ax.tick_params(axis='both', which='major', labelsize=22)

for i in range(len(a),2*len(a)):
    fn = path + 'Final_Example1_a' + str(a[i - len(a)]) +'_typedaff_optLBFGS_rep0_weak'
    forward = dill.load(open(fn + '_config.pickle', "rb"),fix_imports=True, encoding="latin1")['forward']
    params = dill.load(open(fn + '_epoch' + str(5000).rjust(6, '0') + '.pickle', "rb"),fix_imports=True, encoding="latin1")['params']
    ax = axes[i]
    def u(x):
        return jnp.sin(2 * jnp.pi * x) + 0.1 * jnp.sin(a[i - len(a)] * jnp.pi * x) - x * (jnp.sin(2 * jnp.pi) + 0.1 * jnp.sin(a[i - len(a)] * jnp.pi))
    error = u(grid) - forward(grid,params['net'])
    ax.plot(grid,error,color = 'black')
    ax.set_title('Error ' + titles[i - len(a)], fontweight='bold', fontsize = 28)
    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1e'))

plt.tight_layout()
plt.savefig('sol_ex1.pdf')
plt.close()
