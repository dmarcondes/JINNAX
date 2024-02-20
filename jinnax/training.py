#Functions to train NN
import jax
import jax.numpy as jnp
import optax
from alive_progress import alive_bar
import math
import time
from jinnax import arch as jar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

__docformat__ = "numpy"

#MSE
@jax.jit
def MSE(true,pred):
  return jnp.mean((true - pred)**2)

#MSE self-adaptative
@jax.jit
def MSE_SA(true,pred,wheight):
  return jnp.mean(jax.nn.sigmoid(wheight) * (true - pred)**2)

#L2 error
@jax.jit
def L2error(pred,true):
  return jnp.sqrt(jnp.sum((true - pred)**2))/jnp.sqrt(jnp.sum(true ** 2))

#Croos entropy
@jax.jit
def CE(true,pred):
  return jnp.mean((- true * jnp.log(pred + 1e-5) - (1 - true) * jnp.log(1 - pred + 1e-5)))

#Croos entropy self-adaptative
@jax.jit
def CE_SA(true,pred,wheight):
  return jnp.mean(jax.nn.sigmoid(wheight) * (- true * jnp.log(pred + 1e-5) - (1 - true) * jnp.log(1 - pred + 1e-5)))

#IoU
@jax.jit
def IoU(true,pred):
  return 1 - (jnp.sum(2 * true * pred) + 1)/(jnp.sum(true + pred) + 1)

#IoU self-adaptative
@jax.jit
def IoU_SA(true,pred,wheight):
  return 1 - (jnp.sum(jax.nn.sigmoid(wheight) *2 * true * pred) + 1)/(jnp.sum(jax.nn.sigmoid(wheight) * (true + pred + 1)))

#Training function MNN
def train_morph(x,y,forward,params,loss,sa = False,epochs = 1,batches = 1,lr = 0.001,b1 = 0.9,b2 = 0.999,eps = 1e-08,eps_root = 0.0,key = 0,notebook = False,epoch_print = 100):
    #Key
    key = jax.random.split(jax.random.PRNGKey(key),epochs)

    #Batch size
    bsize = int(math.floor(x.shape[0]/batches))

    #Self-adaptative
    if sa:
        params.append({'w': jnp.zeros((y.shape)) + 1.0})
        @jax.jit
        def lf(params,x,y):
            return jnp.mean(jax.vmap(loss,in_axes = (0,0,0))(forward(x,params[:-1]),y,params[-1]['w']))
    else:
        #Loss function
        @jax.jit
        def lf(params,x,y):
            return jnp.mean(jax.vmap(loss,in_axes = (0,0))(forward(x,params),y))

    #Optmizer NN
    optimizer = optax.adam(lr,b1,b2,eps,eps_root)
    opt_state = optimizer.init(params)

    #Training function
    grad_loss = jax.jit(jax.grad(lf,0))
    @jax.jit
    def update(opt_state,params,x,y):
      grads = grad_loss(params,x,y)
      if sa:
          grads[-1]['w'] = - grads[-1]['w']
      updates, opt_state = optimizer.update(grads, opt_state)
      params = optax.apply_updates(params, updates)
      return opt_state,params

    #Train
    t0 = time.time()
    with alive_bar(epochs) as bar:
        for e in range(epochs):
            if not sa:
                #Permutate x
                x = jax.random.permutation(jax.random.PRNGKey(key[e,0]),x,0)
                for b in range(batches):
                    if b < batches - 1:
                        xb = jax.lax.dynamic_slice(x,(b*bsize,0,0),(bsize,x.shape[1],x.shape[2]))
                        yb = jax.lax.dynamic_slice(x,(b*bsize,0,0),(bsize,x.shape[1],x.shape[2]))
                    else:
                        xb = x[b*bsize:x.shape[0],:,:]
                        yb = y[b*bsize:y.shape[0],:,:]
                    opt_state,params = update(opt_state,params,xb,yb)
            else:
                opt_state,params = update(opt_state,params,x,y)
            l = str(jnp.round(lf(params,x,y),10))
            if(e % epoch_print == 0 and notebook):
                print('Epoch: ' + str(e) + ' Time: ' + str(jnp.round(time.time() - t0,2)) + ' s Loss: ' + l)
            if not notebook:
                bar.title("Loss: " + l)
                bar()

    return params


#Training function FCNN
def train_fcnn(x,y,forward,params,loss,sa = False,epochs = 1,batches = 1,lr = 0.001,b1 = 0.9,b2 = 0.999,eps = 1e-08,eps_root = 0.0,key = 0,notebook = False,epoch_print = 1000):
    #Key
    key = jax.random.split(jax.random.PRNGKey(key),epochs)

    #Batch size
    bsize = int(math.floor(x.shape[0]/batches))

    #Self-adaptative
    if sa:
        params.append({'w': jnp.zeros((y.shape)) + 1.0})
        @jax.jit
        def lf(params,x,y):
            return jnp.mean(jax.vmap(loss,in_axes = (0,0,0))(forward(x,params[:-1]),y,params[-1]['w']))
    else:
        #Loss function
        @jax.jit
        def lf(params,x,y):
            return jnp.mean(jax.vmap(loss,in_axes = (0,0))(forward(x,params),y))

    #Optmizer NN
    optimizer = optax.adam(lr,b1,b2,eps,eps_root)
    opt_state = optimizer.init(params)

    #Training function
    grad_loss = jax.jit(jax.grad(lf,0))
    @jax.jit
    def update(opt_state,params,x,y):
      grads = grad_loss(params,x,y)
      if sa:
          grads[-1]['w'] = - grads[-1]['w']
      updates, opt_state = optimizer.update(grads, opt_state)
      params = optax.apply_updates(params, updates)
      return opt_state,params

    #Train
    t0 = time.time()
    with alive_bar(epochs) as bar:
        for e in range(epochs):
            #Permutate x
            if not sa:
                x = jax.random.permutation(jax.random.PRNGKey(key[e,0]),x,0)
                for b in range(batches):
                    if b < batches - 1:
                        xb = jax.lax.dynamic_slice(x,(b*bsize,0),(bsize,x.shape[1]))
                        yb = jax.lax.dynamic_slice(x,(b*bsize,0),(bsize,x.shape[1]))
                    else:
                        xb = x[b*bsize:x.shape[0],:]
                        yb = y[b*bsize:y.shape[0],:]
                    opt_state,params = update(opt_state,params,xb,yb)
            else:
                opt_state,params = update(opt_state,params,x,y)
            l = str(jnp.round(lf(params,x,y),10))
            if(e % epoch_print == 0 and notebook):
                print('Epoch: ' + str(e) + ' Time: ' + str(jnp.round(time.time() - t0,2)) + ' s Loss: ' + l)
            if not notebook:
                bar.title("Loss: " + l)
                bar()

    del params[-1]
    return params

#Training PINN
def train_pinn(data,width,pde,test_data = None,epochs = 100,activation = jax.nn.tanh,lr = 0.001,b1 = 0.9,b2 = 0.999,eps = 1e-08,eps_root = 0.0,key = 0,epoch_print = 100,plot = False,times = 3,d2 = False,save = False,file_name = 'result_pinn'):
    """Train a Physics-informed Neural Network.

    Parameters
    ----------
    data : dict

        Data generated by the jinnax.data.generate_PINNdata

    width : list

        A list with the width of each layer

    pde : function

        The partial differential operator. Its arguments are u, x and t

    test_data : dict

        A dictionay with test data for L2 error calculationgenerated by the jinnax.data.generate_PINNdata. Default None for not calculating L2 error

    epochs : int

        Number of training epochs. Default 100

    activation : jax.nn activation

        The activation function of the neural network. Default jax.nn.tanh

    lr,b1,b2,eps,eps_root: float

        Hyperparameters of the Adam algorithm. Default lr = 0.001,b1 = 0.9,b2 = 0.999,eps = 1e-08,eps_root = 0.0

    key : int

        Seed for parameters initialization. Default 0

    epoch_print : int

        Number of epochs to calculate and print test error. Default 100

    plot : logical

        Whether to plot the results from time to time. Default False

    times : int

        Number of times to plot. Default 3

    d2 : logical

        Whether to plot 2D plot. Default False

    save : logical

        Whether to save the plots, L2 error and current parameters. Default False

    file_name : str

        File prefix to save the plots and L2 error. Default 'result_pinn'

    Returns
    -------
    dict-like object with estimated function
    """

    #Initialize architecture
    nnet = jar.fconNN(width,activation,key)
    forward = nnet['forward']
    params = nnet['params']

    #Loss function
    @jax.jit
    def lf(params,x):
        loss = 0
        if x['sensor'] is not None:
            loss = loss + jnp.mean(jax.vmap(MSE,in_axes = (0,0))(forward(x['sensor'],params),x['usensor']))
        if x['boundary'] is not None:
            loss = loss + jnp.mean(jax.vmap(MSE,in_axes = (0,0))(forward(x['boundary'],params),x['uboundary']))
        if x['initial'] is not None:
            loss = loss + jnp.mean(jax.vmap(MSE,in_axes = (0,0))(forward(x['initial'],params),x['uinitial']))
        if x['collocation'] is not None:
            x_col = x['collocation'][:,:-1].reshape((x['collocation'].shape[0],x['collocation'].shape[1] - 1))
            t_col = x['collocation'][:,-1].reshape((x['collocation'].shape[0],1))
            loss = loss + MSE(pde(lambda x,t: forward(jnp.append(x,t,1),params),x_col,t_col),0)
        return loss

    #Optmizer NN
    optimizer = optax.adam(lr,b1,b2,eps,eps_root)
    opt_state = optimizer.init(params)

    #Training function
    grad_loss = jax.jit(jax.grad(lf,0))
    @jax.jit
    def update(opt_state,params,x):
      grads = grad_loss(params,x)
      updates, opt_state = optimizer.update(grads, opt_state)
      params = optax.apply_updates(params, updates)
      return opt_state,params

    #Train
    t0 = time.time()
    with alive_bar(epochs) as bar:
        for e in range(epochs):
            opt_state,params = update(opt_state,params,data)
            if e % epoch_print == 0:
                l = 'Time: ' + str(round(time.time() - t0)) + ' s Loss: ' + str(jnp.round(lf(params,data),6))
                if test_data is not None:
                    res = process_result(test_data,lambda xt: forward(xt,params),data,plot = plot,times = times,d2 = d2,save = save,file_name = file_name + '_epoch' + str(e).rjust(6, '0'),print = False)
                    l = l + ' L2 error: ' + str(jnp.round(res['l2_error'][0],6))
                print(l)
                if save:
                    pickle.dump({'params': params,'width': width},open(file_name + '_epoch' + str(e).rjust(6, '0') + '.pickle','wb'), protocol = pickle.HIGHEST_PROTOCOL)
            bar()
    def u(xt):
        return forward(xt,params)

    return {'u': u,'params': params,'forward': forward}

#Process result
def process_result(test_data,u_trained,train_data,plot = True,times = 3,d2 = True,save = False,file_name = 'result_pinn',print = True):
    """Process the results of a Physics-informed Neural Network.

    Parameters
    ----------
    test_data : dict

        A dictionay with test data for L2 error calculation. Must have the xt data ('xt') and the solution at each point ('u').

    u_trained : function

        Function estimated by the trained PINN

    train_data : dict

        Data generated by the jinnax.data.generate_PINNdata

    plot : logical

        Wheter to generate plots comparing the exact and estimated solutions. Default True

    times : int

        Number of times to plot. Default 3

    d2 : logical

        Whether to plot 2D plot. Default True

    save : logical

        Whether to save the plots and L2 error. Default False

    file_name : str

        File prefix to save the plots and L2 error. Default 'result_pinn'

    print : logical

        Wheter to print the error. Default True

    Returns
    -------
    pandas data frame with L2 error
    """

    #Dimension
    d = test_data['xt'].shape[1] - 1

    #Number of plots multiple of 5
    times = 5 * round(times/5.0)

    #Data
    xt = test_data['xt']
    u = test_data['u']
    upred = u_trained(xt)

    #Results
    l2_error = L2error(upred,u)
    sensor_sample = train_data['sensor'].shape[0]
    boundary_sample = train_data['boundary'].shape[0]
    initial_sample = train_data['initial'].shape[0]
    collocation_sample = train_data['collocation'].shape[0]
    df = pd.DataFrame(np.array([sensor_sample,boundary_sample,initial_sample,collocation_sample,l2_error.tolist()]).reshape((1,5)), columns=['sensor_sample','boundary_sample','initial_sample','collocation_sample','l2_error'])
    if save:
        df.to_csv(file_name + '.csv',index = False)
    if print:
        print('L2 error: ' + str(jnp.round(l2_error,6)))

    #Plots
    if d == 1:
        fig, ax = plt.subplots(int(times/5),5)
        fig.tight_layout()
        tlo = jnp.min(xt[:,-1])
        tup = jnp.max(xt[:,-1])
        ylo = jnp.min(jnp.append(u,upred,0))
        yup = jnp.max(jnp.append(u,upred,0))
        k = 0
        t_values = np.linspace(tlo,tup,times)
        for i in range(int(times/5)):
            for j in range(5):
                if k < len(t_values):
                    t = t_values[k]
                    t = xt[jnp.abs(xt[:,-1] - t) == jnp.min(jnp.abs(xt[:,-1] - t)),-1][0].tolist()
                    x_plot = xt[xt[:,-1] == t,:-1]
                    y_plot = upred[xt[:,-1] == t,:]
                    u_plot = u[xt[:,-1] == t,:]
                    #ax[k].set_aspect(1)
                    if int(times/3) > 1:
                        ax[i,j].plot(x_plot[:,0],u_plot[:,0],'b-',linewidth=2,label='Exact')
                        ax[i,j].plot(x_plot[:,0],y_plot,'r--',linewidth=2,label='Prediction')
                        ax[i,j].set_title('$t = %.2f$' % (t),fontsize=10)
                        ax[i,j].set_xlabel('$x$')
                        ax[i,j].set_ylim([1.3 * ylo.tolist(),1.3 * yup.tolist()])
                    else:
                        ax[j].plot(x_plot[:,0],u_plot[:,0],'b-',linewidth=2,label='Exact')
                        ax[j].plot(x_plot[:,0],y_plot,'r--',linewidth=2,label='Prediction')
                        ax[j].set_title('$t = %.2f$' % (t),fontsize=10)
                        ax[j].set_xlabel('$x$')
                        ax[j].set_ylim([1.3 * ylo.tolist(),1.3 * yup.tolist()])
                    k = k + 1


        fig = plt.gcf()
        if plot:
            plt.show()
        if save:
            fig.savefig(file_name + '_slices.png')
        plt.close()

        #2d plot
        if d2:
            fig, ax = plt.subplots(1,2)
            l = int(jnp.sqrt(xt.shape[0]).tolist())
            ax[0].pcolormesh(xt[:,-1].reshape((l,l)),xt[:,0].reshape((l,l)),u[:,0].reshape((l,l)),cmap = 'RdBu',vmin = ylo.tolist(),vmax = yup.tolist())
            ax[0].set_title('Exact')
            ax[1].pcolormesh(xt[:,-1].reshape((l,l)),xt[:,0].reshape((l,l)),upred[:,0].reshape((l,l)),cmap = 'RdBu',vmin = ylo.tolist(),vmax = yup.tolist())
            ax[1].set_title('Predicted')
            #fig.colorbar(c, ax=ax)

            fig = plt.gcf()
            if plot:
                plt.show()
            if save:
                fig.savefig(file_name + '_2d.png')
            plt.close()

    return df
