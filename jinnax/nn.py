#Functions to train NN
import jax
import jax.numpy as jnp
import optax
from alive_progress import alive_bar
import math
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dill as pickle
from genree import bolstering as gb
from genree import kernel as gk
from jax import random

__docformat__ = "numpy"

#MSE
@jax.jit
def MSE(pred,true):
    """
    Mean square error
    ----------

    Parameters
    ----------
    pred : jax.numpy.array

        A JAX numpy array with the predicted values

    true : jax.numpy.array

        A JAX numpy array with the true values

    Returns
    -------
    mean square error
    """
    return jnp.mean((true - pred)**2)

#L2 error
@jax.jit
def L2error(pred,true):
    """
    L2-error
    ----------

    Parameters
    ----------
    pred : jax.numpy.array

        A JAX numpy array with the predicted values

    true : jax.numpy.array

        A JAX numpy array with the true values

    Returns
    -------
    L2-error
    """
    return jnp.sqrt(jnp.sum((true - pred)**2))/jnp.sqrt(jnp.sum(true ** 2))

#Simple fully connected architecture. Return the initial parameters and the function for the forward pass
def fconNN(width,activation = jax.nn.tanh,key = 0):
    """
    Initialize fully connected neural network
    ----------

    Parameters
    ----------
    width : list

        List with the layers width

    activation : jax.nn activation

        The activation function. Default jax.nn.tanh

    key : int

        Seed for parameters initialization. Default 0

    Returns
    -------
    dict with initial parameters and the function for the forward pass
    """
    #Initialize parameters with Glorot initialization
    initializer = jax.nn.initializers.glorot_normal()
    key = jax.random.split(jax.random.PRNGKey(key),len(width)-1) #Seed for initialization
    params = list()
    for key,lin,lout in zip(key,width[:-1],width[1:]):
        W = initializer(key,(lin,lout),jnp.float32)
        B = initializer(key,(1,lout),jnp.float32)
        params.append({'W':W,'B':B})

    #Define function for forward pass
    @jax.jit
    def forward(x,params):
      *hidden,output = params
      for layer in hidden:
        x = activation(x @ layer['W'] + layer['B'])
      return x @ output['W'] + output['B']

    #Return initial parameters and forward function
    return {'params': params,'forward': forward}

#Training PINN
def train_PINN(data,width,pde,test_data = None,epochs = 100,activation = jax.nn.tanh,lr = 0.001,b1 = 0.9,b2 = 0.999,eps = 1e-08,eps_root = 0.0,key = 0,epoch_print = 100,save = False,file_name = 'result_pinn'):
    """
    Train a Physics-informed Neural Network
    ----------

    Parameters
    ----------
    data : dict

        Data generated by the jinnax.data.generate_PINNdata function

    width : list

        A list with the width of each layer

    pde : function

        The partial differential operator. Its arguments are u, x and t

    test_data : dict, None

        A dictionay with test data for L2 error calculation generated by the jinnax.data.generate_PINNdata function. Default None for not calculating L2 error

    epochs : int

        Number of training epochs. Default 100

    activation : jax.nn activation

        The activation function of the neural network. Default jax.nn.tanh

    lr,b1,b2,eps,eps_root: float

        Hyperparameters of the Adam algorithm. Default lr = 0.001, b1 = 0.9, b2 = 0.999, eps = 1e-08, eps_root = 0.0

    key : int

        Seed for parameters initialization. Default 0

    epoch_print : int

        Number of epochs to calculate, save and print test error, and display and save plots. Default 100

    save : logical

        Whether to save the current parameters. Default False

    file_name : str

        File prefix to save the plots, the L2 error and the current parameters. Default 'result_pinn'

    Returns
    -------
    dict-like object with the estimated function, the estimated parameters and the neural network function for the forward pass
    """

    #Initialize architecture
    nnet = fconNN(width,activation,key)
    forward = nnet['forward']
    params = nnet['params']

    #Save config
    if save:
        pickle.dump({'train_data': data,'epochs': epochs,'activation': lambda x: activation(x),'init_params': params,'width': width,'pde': pde,'lr': lr,'b1': b1,'b2': b2,'eps': eps,'eps_root': eps_root,'key': key},open(file_name + '_config.pickle','wb'), protocol = pickle.HIGHEST_PROTOCOL)

    #Define loss function
    @jax.jit
    def lf(params,x):
        loss = 0
        if x['sensor'] is not None:
            #Term that refers to sensor data
            loss = loss + jnp.mean(jax.vmap(MSE,in_axes = (0,0))(forward(x['sensor'],params),x['usensor']))
        if x['boundary'] is not None:
            #Term that refers to boundary data
            loss = loss + jnp.mean(jax.vmap(MSE,in_axes = (0,0))(forward(x['boundary'],params),x['uboundary']))
        if x['initial'] is not None:
            #Term that refers to initial data
            loss = loss + jnp.mean(jax.vmap(MSE,in_axes = (0,0))(forward(x['initial'],params),x['uinitial']))
        if x['collocation'] is not None:
            #Term that refers to collocation points
            x_col = x['collocation'][:,:-1].reshape((x['collocation'].shape[0],x['collocation'].shape[1] - 1))
            t_col = x['collocation'][:,-1].reshape((x['collocation'].shape[0],1))
            loss = loss + MSE(pde(lambda x,t: forward(jnp.append(x,t,1),params),x_col,t_col),0)
        return loss

    #Initialize Adama oOptmizer
    optimizer = optax.adam(lr,b1,b2,eps,eps_root)
    opt_state = optimizer.init(params)

    #Define the gradient function
    grad_loss = jax.jit(jax.grad(lf,0))

    #Define update function
    @jax.jit
    def update(opt_state,params,x):
        #Compute gradient
        grads = grad_loss(params,x)
        #Calculate parameters updates
        updates, opt_state = optimizer.update(grads, opt_state)
        #Update parameters
        params = optax.apply_updates(params, updates)
        #Return state of optmizer and updated parameters
        return opt_state,params

    ###Training###
    t0 = time.time()
    #Initialize alive_bar for tracing in terminal
    with alive_bar(epochs) as bar:
        #For each epoch
        for e in range(epochs):
            #Update optimizer state and parameters
            opt_state,params = update(opt_state,params,data)
            #After epoch_print epochs
            if e % epoch_print == 0:
                #Compute elapsed time and current error
                l = 'Time: ' + str(round(time.time() - t0)) + ' s Loss: ' + str(jnp.round(lf(params,data),6))
                #If there is test data, compute current L2 error
                if test_data is not None:
                    #Compute L2 error
                    l2_test = L2error(forward(test_data['xt'],params),test_data['u']).tolist()
                    l = l + ' L2 error: ' + str(jnp.round(l2_test,6))
                #Print
                print(l)
            if save:
                #Save current parameters
                pickle.dump({'params': params,'width': width,'time': time.time() - t0,'loss': lf(params,data)},open(file_name + '_epoch' + str(e).rjust(6, '0') + '.pickle','wb'), protocol = pickle.HIGHEST_PROTOCOL)
            #Update alive_bar
            bar()
    #Define estimated function
    def u(xt):
        return forward(xt,params)

    return {'u': u,'params': params,'forward': forward,'time': time.time() - t0}

#Process result
def process_result(test_data,fit,train_data,plot = True,times = 5,d2 = True,save = False,file_name = 'result_pinn',print_res = True):
    """
    Process the results of a Physics-informed Neural Network
    ----------

    Parameters
    ----------
    test_data : dict

        A dictionay with test data for L2 error calculation generated by the jinnax.data.generate_PINNdata function

    fit : dict

        Object returned by train_PINN function

    train_data : dict

        Training data generated by the jinnax.data.generate_PINNdata

    plot : logical

        Wheter to generate plots comparing the exact and estimated solutions when the spatial dimension is one. Default True

    times : int

        Number of points along the time interval to plot. Default 5

    d2 : logical

        Whether to plot 2D plot when the spatial dimension is one. Default True

    save : logical

        Whether to save the plots and the L2 error. Default False

    file_name : str

        File prefix to save the plots and the L2 error. Default 'result_pinn'

    print_res : logical

        Wheter to print the L2 error. Default True

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
    upred = fit['u'](xt)
    if train_data['sensor'] is not None:
        upred_train = fit['u'](train_data['sensor'])

    #Results
    l2_error_test = L2error(upred,u).tolist()
    MSE_test = MSE(upred,u).tolist()
    if train_data['sensor'] is not None:
        sensor_sample = train_data['sensor'].shape[0]
        l2_error_train = L2error(upred_train,train_data['usensor']).tolist()
        MSE_train = MSE(upred_train,train_data['usensor']).tolist()
    else:
        sensor_sample = 0
        l2_error_train = -1
        MSE_train = -1
    if train_data['boundary'] is not None:
        boundary_sample = train_data['boundary'].shape[0]
    else:
        boundary_sample = 0
    if train_data['initial'] is not None:
        initial_sample = train_data['initial'].shape[0]
    else:
        initial_sample = 0
    if train_data['collocation'] is not None:
        collocation_sample = train_data['collocation'].shape[0]
    else:
        collocation_sample = 0
    df = pd.DataFrame(np.array([fit['time'],sensor_sample,boundary_sample,initial_sample,collocation_sample,l2_error_test,MSE_test,l2_error_train,MSE_train]).reshape((1,9)),
        columns=['training_time','sensor_sample','boundary_sample','initial_sample','collocation_sample','l2_error_test','MSE_test','l2_error_train','MSE_train'])
    if save:
        df.to_csv(file_name + '.csv',index = False)
    if print_res:
        print('L2 error test: ' + str(jnp.round(l2_error_test,6)) + ' L2 error train: ' + str(jnp.round(l2_error_train,6)) + ' MSE error test: ' + str(jnp.round(MSE_test,6)) + ' MSE error train: ' + str(jnp.round(MSE_train,6)) )

    #Plots
    if d == 1:
        plot_pinn1D(times,xt,u,upred,d2,save,file_name)

    return df

#Plot results for d = 1
def plot_pinn1D(times,xt,u,upred,d2 = True,show = True,save = False,file_name = 'result_pinn'):
    """
    Plot the prediction of a 1D PINN
    ----------

    Parameters
    ----------
    times : int

        Number of points along the time interval to plot. Default 5

    xt : array

        Test data xt array

    u : array

        Test data u(x,t) array

    upred : array

        Predicted upred(x,t) array

    d2 : logical

        Whether to plot 2D plot. Default True

    save : logical

        Whether to save the plots. Default False

    show : logical

        Whether to show the plots. Default True

    file_name : str

        File prefix to save the plots. Default 'result_pinn'

    Returns
    -------
    pandas data frame with L2 error
    """
    #Initialize
    fig, ax = plt.subplots(int(times/5),5,figsize = (10*int(times/5),2*int(times/5)))
    tlo = jnp.min(xt[:,-1])
    tup = jnp.max(xt[:,-1])
    ylo = jnp.min(jnp.append(u,upred,0))
    yup = jnp.max(jnp.append(u,upred,0))
    k = 0
    t_values = np.linspace(tlo,tup,times)

    #Create
    for i in range(int(times/5)):
        for j in range(5):
            if k < len(t_values):
                t = t_values[k]
                t = xt[jnp.abs(xt[:,-1] - t) == jnp.min(jnp.abs(xt[:,-1] - t)),-1][0].tolist()
                x_plot = xt[xt[:,-1] == t,:-1]
                y_plot = upred[xt[:,-1] == t,:]
                u_plot = u[xt[:,-1] == t,:]
                if int(times/5) > 1:
                    ax[i,j].plot(x_plot[:,0],u_plot[:,0],'b-',linewidth=2,label='Exact')
                    ax[i,j].plot(x_plot[:,0],y_plot,'r--',linewidth=2,label='Prediction')
                    ax[i,j].set_title('$t = %.2f$' % (t),fontsize=10)
                    ax[i,j].set_xlabel(' ')
                    ax[i,j].set_ylim([1.3 * ylo.tolist(),1.3 * yup.tolist()])
                else:
                    ax[j].plot(x_plot[:,0],u_plot[:,0],'b-',linewidth=2,label='Exact')
                    ax[j].plot(x_plot[:,0],y_plot,'r--',linewidth=2,label='Prediction')
                    ax[j].set_title('$t = %.2f$' % (t),fontsize=10)
                    ax[j].set_xlabel(' ')
                    ax[j].set_ylim([1.3 * ylo.tolist(),1.3 * yup.tolist()])
                k = k + 1

    #Show and save
    fig = plt.gcf()
    if show:
        plt.show()
    if save:
        fig.savefig(file_name + '_slices.png')
    plt.close()

    #2d plot
    if d2:
        #Initialize
        fig, ax = plt.subplots(1,2)
        l1 = jnp.unique(xt[:,-1]).shape[0]
        l2 = jnp.unique(xt[:,0]).shape[0]

        #Create
        ax[0].pcolormesh(xt[:,-1].reshape((l2,l1)),xt[:,0].reshape((l2,l1)),u[:,0].reshape((l2,l1)),cmap = 'RdBu',vmin = ylo.tolist(),vmax = yup.tolist())
        ax[0].set_title('Exact')
        ax[1].pcolormesh(xt[:,-1].reshape((l2,l1)),xt[:,0].reshape((l2,l1)),upred[:,0].reshape((l2,l1)),cmap = 'RdBu',vmin = ylo.tolist(),vmax = yup.tolist())
        ax[1].set_title('Predicted')

        #Show and save
        fig = plt.gcf()
        if show:
            plt.show()
        if save:
            fig.savefig(file_name + '_2d.png')
        plt.close()

#Process training
def process_training(test_data,file_name,at_each = 100,bolstering = True,mc_sample = 10000,save = False,file_name_save = 'result_pinn',key = 435,ec = 1e-6,lamb = 1):
    """
    Process the training of a Physics-informed Neural Network
    ----------

    Parameters
    ----------
    test_data : dict

        A dictionay with test data for L2 error calculation generated by the jinnax.data.generate_PINNdata function

    file_name: str

        Name of the files saved during training

    at_each: int

        Compute results for epochs multiple of at_each. Default 100

    bolstering: logical

        Whether to compute bolstering mean square error

    mc_sample: int

        Number of sample for Monte Carlo integration in bolstering

    save : logical

        Whether to save the training results. Default False

    file_name_save : str

        File prefix to save the plots and the L2 error. Default 'result_pinn'

    key: int

        Key for random samples in bolstering

    ec: float

        Stopping criteria error for EM algorithm in bolstering

    lamb: float

        Hyperparameter of EM algorithm in bolstering

    Returns
    -------
    pandas data frame with training results
    """
    #Config
    with open(file_name + '_config.pickle', 'rb') as file:
        config = pickle.load(file)
    epochs = config['epochs']
    train_data = config['train_data']
    forward = fconNN(config['width'],config['activation'],config['key'])['forward']

    #Generate keys
    if bolstering:
        keys = jax.random.randint(random.PRNGKey(key),(epochs,),0,1e9)

    #Data
    xdata = None
    ydata = None
    xydata = None
    if train_data['sensor'] is not None:
        sensor_sample = train_data['sensor'].shape[0]
        xdata = train_data['sensor']
        ydata = train_data['usensor']
        xydata = jnp.column_stack((train_data['sensor'],train_data['usensor']))
    else:
        sensor_sample = 0
    if train_data['boundary'] is not None:
        boundary_sample = train_data['boundary'].shape[0]
        if xdata is not None:
            xdata = jnp.vstack((xdata,train_data['boundary']))
            ydata = jnp.vstack((ydata,train_data['uboundary']))
            xydata = jnp.vstack((xydata,jnp.column_stack((train_data['boundary'],train_data['uboundary']))))
        else:
            xdata = train_data['boundary']
            ydata = train_data['uboundary']
            xydata = jnp.column_stack((train_data['boundary'],train_data['uboundary']))
    else:
        boundary_sample = 0
    if train_data['initial'] is not None:
        initial_sample = train_data['initial'].shape[0]
        if xdata is not None:
            xdata = jnp.vstack((xdata,train_data['initial']))
            ydata = jnp.vstack((ydata,train_data['uinitial']))
            xydata = jnp.vstack((xydata,jnp.column_stack((train_data['initial'],train_data['uinitial']))))
        else:
            xdata = train_data['initial']
            ydata = train_data['uinitial']
            xydata = jnp.column_stack((train_data['initial'],train_data['uinitial']))
    else:
        initial_sample = 0
    if train_data['collocation'] is not None:
        collocation_sample = train_data['collocation'].shape[0]
    else:
        collocation_sample = 0

    #Initialize loss
    train_mse = []
    test_mse = []
    train_L2 = []
    test_L2 = []
    bolstX = []
    bolstXY = []
    loss = []
    time = []
    ep = []

    #Process training
    with alive_bar(epochs) as bar:
        for e in range(epochs):
            if e % at_each == 0 or e == epochs - 1:
                ep = ep + [e]

                #Read parameters
                params = pd.read_pickle(file_name + '_epoch' + str(e).rjust(6, '0') + '.pickle')

                #Time
                time = time + [params['time']]

                #Define learned function
                def psi(x):
                    return forward(x,params['params'])

                #Train MSE and L2
                if xdata is not None:
                    train_mse = train_mse + [MSE(psi(xdata),ydata).tolist()]
                    train_L2 = train_L2 + [L2error(psi(xdata),ydata).tolist()]
                else:
                    train_mse = train_mse + [None]
                    train_L2 = train_L2 + [None]

                #Test MSE and L2
                test_mse = test_mse + [MSE(psi(test_data['xt']),test_data['u']).tolist()]
                test_L2 = test_L2 + [L2error(psi(test_data['xt']),test_data['u']).tolist()]

                #Bolstering
                if bolstering:
                    kx = gk.kernel_estimator(xdata,random.PRNGKey(keys[e]),method = "mpe",lamb = lamb,ec = ec)
                    kxy = gk.kernel_estimator(xydata,random.PRNGKey(keys[e]),method = "mpe",lamb = lamb,ec = ec)
                    bolstX = bolstX + [gb.bolstering(psi,xdata,ydata,kx,random.PRNGKey(keys[e]),mc_sample = mc_sample).tolist()]
                    bolstXY = bolstXY + [gb.bolstering(psi,xdata,ydata,kxy,random.PRNGKey(keys[e]),mc_sample = mc_sample).tolist()]
                else:
                    bolst = bolst + [None]

                #Loss
                loss = loss + [params['loss'].tolist()]
            #Update alive_bar
            bar()

    #Create data frame
    df = pd.DataFrame(np.column_stack([ep,time,[sensor_sample] * len(ep),[boundary_sample] * len(ep),[initial_sample] * len(ep),[collocation_sample] * len(ep),loss,
    train_mse,test_mse,train_L2,test_L2,bolstX,bolstXY]),
        columns=['epoch','training_time','sensor_sample','boundary_sample','initial_sample','collocation_sample','loss','train_mse','test_mse','train_L2','test_L2','bolstX','bolstXY'])
    if save:
        df.to_csv(file_name_save + '.csv',index = False)

    return df
