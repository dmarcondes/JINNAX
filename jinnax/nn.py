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
import os

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
    return (true - pred) ** 2

#MSE self-adaptative
@jax.jit
def MSE_SA(pred,true,wheight,c = 100):
    """
    Selft-adaptative mean square error
    ----------

    Parameters
    ----------
    pred : jax.numpy.array

        A JAX numpy array with the predicted values

    true : jax.numpy.array

        A JAX numpy array with the true values

    wheight : jax.numpy.array

        A JAX numpy array with the weights

    c : float

        Hyperparameter

    Returns
    -------
    self-adaptative mean square error with sigmoid mask
    """
    return c * jax.nn.sigmoid(wheight) * (true - pred) ** 2

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

#Get activation from string
def get_activation(act):
    """
    Return activation function from string
    ----------

    Parameters
    ----------
    act : str

        Name of the activation function. Default 'tanh'

    Returns
    -------
    jax.nn activation function
    """
    if act == 'tanh':
        return jax.nn.tanh
    elif act == 'relu':
        return jax.nn.relu
    elif act == 'relu6':
        return jax.nn.relu6
    elif act == 'sigmoid':
        return jax.nn.sigmoid
    elif act == 'softplus':
        return jax.nn.softplus
    elif act == 'sparse_plus':
        return jx.nn.sparse_plus
    elif act == 'soft_sign':
        return jax.nn.soft_sign
    elif act == 'silu':
        return jax.nn.silu
    elif act == 'swish':
        return jax.nn.swish
    elif act == 'log_sigmoid':
        return jax.nn.log_sigmoid
    elif act == 'leaky_relu':
        return jax.xx.leaky_relu
    elif act == 'hard_sigmoid':
        return jax.nn.hard_sigmoid
    elif act == 'hard_silu':
        return jax.nn.hard_silu
    elif act == 'hard_swish':
        return jax.nn.hard_swish
    elif act == 'hard_tanh':
        return jax.nn.hard_tanh
    elif act == 'elu':
        return jax.nn.elu
    elif act == 'celu':
        return jax.nn.celu
    elif act == 'selu':
        return jax.nn.selu
    elif act == 'gelu':
        return jax.nn.gelu
    elif act == 'glu':
        return jax.nn.glu
    elif act == 'squareplus':
        return  jax.nn.squareplus
    elif act == 'mish':
        return jax.nn.mish

#Training PINN
def train_PINN(data,width,pde,test_data = None,epochs = 100,at_each = 10,activation = 'tanh',neumann = False,oper_neumann = False,sa = False,c = {'ws': 1,'wr': 1,'w0': 100},inverse = False,initial_par = None,lr = 0.001,b1 = 0.9,b2 = 0.999,eps = 1e-08,eps_root = 0.0,key = 0,epoch_print = 100,save = False,file_name = 'result_pinn'):
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

    at_each : int

        Save results for epochs multiple of at_each. Default 10

    activation : str

        The name of the activation function of the neural network. Default 'tanh'

    neumann : logical

        Whether to consider Neumann boundary conditions

    oper_neumann : function

        Penalization of Neumann boundary conditions

    sa : logical

        Whether to consider self-adaptative PINN

    c : dict

        Dictionary with the hyperparameters of the self-adaptative sigmoid mask for the initial (w0), sensor (ws) and collocation (wr) points. The weights of the boundary points is fixed to 1

    inverse : logical

        Whether to estimate parameters of the PDE

    initial_par : jax.numpy.array

        Initial value of the parameters of the PDE in an inverse problem

    lr,b1,b2,eps,eps_root: float

        Hyperparameters of the Adam algorithm. Default lr = 0.001, b1 = 0.9, b2 = 0.999, eps = 1e-08, eps_root = 0.0

    key : int

        Seed for parameters initialization. Default 0

    epoch_print : int

        Number of epochs to calculate and print test errors. Default 100

    save : logical

        Whether to save the current parameters. Default False

    file_name : str

        File prefix to save the current parameters. Default 'result_pinn'

    Returns
    -------
    dict-like object with the estimated function, the estimated parameters, the neural network function for the forward pass and the training time
    """

    #Initialize architecture
    nnet = fconNN(width,get_activation(activation),key)
    params = nnet['params']

    #Initialize parameters in an inverse problem
    if inverse:
        params.append(initial_par)

    #Initialize self adaptative weights
    params.append({})
    if sa:
        #Initialize wheights close to zero
        ksa = jax.random.randint(jax.random.PRNGKey(key),(4,),1,1000000)
        if data['sensor'] is not None:
            params[-1].update({'ws': (1/10)*jax.random.normal(key = jax.random.PRNGKey(ksa[0]),shape = (data['sensor'].shape[0],1))})
        if data['initial'] is not None:
            params[-1].update({'w0': (1/10)*jax.random.normal(key = jax.random.PRNGKey(ksa[2]),shape = (data['initial'].shape[0],1))})
        if data['collocation'] is not None:
            params[-1].update({'wr': (1/10)*jax.random.normal(key = jax.random.PRNGKey(ksa[3]),shape = (data['collocation'].shape[0],1))})

    #Define forward function
    if inverse:
        forward = jax.jit(lambda x,params: nnet['forward'](x,params[:-2]))
    else:
        forward = jax.jit(lambda x,params: nnet['forward'](x,params[:-1]))

    #Save config file
    if save:
        pickle.dump({'train_data': data,'epochs': epochs,'activation': activation,'init_params': params,'forward': forward,'width': width,'pde': pde,'lr': lr,'b1': b1,'b2': b2,'eps': eps,'eps_root': eps_root,'key': key,'inverse': inverse,'sa': sa},open(file_name + '_config.pickle','wb'), protocol = pickle.HIGHEST_PROTOCOL)

    #Define loss function
    if sa:
        #Define loss function
        @jax.jit
        def lf(params,x):
            loss = 0
            if x['sensor'] is not None:
                #Term that refers to sensor data
                loss = loss + jnp.mean(MSE_SA(forward(x['sensor'],params),x['usensor'],params[-1]['ws'],c['ws']))
            if x['boundary'] is not None:
                if neumann:
                    #Neumann coditions
                    xb = x['boundary'][:,:-1].reshape((x['boundary'].shape[0],x['boundary'].shape[1] - 1))
                    tb = x['boundary'][:,-1].reshape((x['boundary'].shape[0],1))
                    loss = loss + jnp.mean(oper_neumann(lambda x,t: forward(jnp.append(x,t,1),params),xb,tb))
                else:
                    #Term that refers to boundary data
                    loss = loss + jnp.mean(MSE(forward(x['boundary'],params),x['uboundary']))
            if x['initial'] is not None:
                #Term that refers to initial data
                loss = loss + jnp.mean(MSE_SA(forward(x['initial'],params),x['uinitial'],params[-1]['w0'],c['w0']))
            if x['collocation'] is not None:
                #Term that refers to collocation points
                x_col = x['collocation'][:,:-1].reshape((x['collocation'].shape[0],x['collocation'].shape[1] - 1))
                t_col = x['collocation'][:,-1].reshape((x['collocation'].shape[0],1))
                if inverse:
                    loss = loss + jnp.mean(MSE_SA(pde(lambda x,t: forward(jnp.append(x,t,1),params),x_col,t_col,params[-2]),0,params[-1]['wr'],c['wr']))
                else:
                    loss = loss + jnp.mean(MSE_SA(pde(lambda x,t: forward(jnp.append(x,t,1),params),x_col,t_col),0,params[-1]['wr'],c['wr']))
            return loss
    else:
        @jax.jit
        def lf(params,x):
            loss = 0
            if x['sensor'] is not None:
                #Term that refers to sensor data
                loss = loss + jnp.mean(MSE(forward(x['sensor'],params),x['usensor']))
            if x['boundary'] is not None:
                if neumann:
                    #Neumann coditions
                    xb = x['boundary'][:,:-1].reshape((x['boundary'].shape[0],x['boundary'].shape[1] - 1))
                    tb = x['boundary'][:,-1].reshape((x['boundary'].shape[0],1))
                    loss = loss + jnp.mean(oper_neumann(lambda x,t: forward(jnp.append(x,t,1),params),xb,tb))
                else:
                    #Term that refers to boundary data
                    loss = loss + jnp.mean(MSE(forward(x['boundary'],params),x['uboundary']))
            if x['initial'] is not None:
                #Term that refers to initial data
                loss = loss + jnp.mean(MSE(forward(x['initial'],params),x['uinitial']))
            if x['collocation'] is not None:
                #Term that refers to collocation points
                x_col = x['collocation'][:,:-1].reshape((x['collocation'].shape[0],x['collocation'].shape[1] - 1))
                t_col = x['collocation'][:,-1].reshape((x['collocation'].shape[0],1))
                if inverse:
                    loss = loss + jnp.mean(MSE(pde(lambda x,t: forward(jnp.append(x,t,1),params),x_col,t_col,params[-2]),0))
                else:
                    loss = loss + jnp.mean(MSE(pde(lambda x,t: forward(jnp.append(x,t,1),params),x_col,t_col),0))
            return loss

    #Initialize Adam Optmizer
    optimizer = optax.adam(lr,b1,b2,eps,eps_root)
    opt_state = optimizer.init(params)

    #Define the gradient function
    grad_loss = jax.jit(jax.grad(lf,0))

    #Define update function
    @jax.jit
    def update(opt_state,params,x):
        #Compute gradient
        grads = grad_loss(params,x)
        #Invert gradient of self-adaptative wheights
        if sa:
            for w in grads[-1]:
                grads[-1][w] = - grads[-1][w]
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
            if (e % at_each == 0 or e == epochs - 1) and save:
                #Save current parameters
                pickle.dump({'params': params,'width': width,'time': time.time() - t0,'loss': lf(params,data)},open(file_name + '_epoch' + str(e).rjust(6, '0') + '.pickle','wb'), protocol = pickle.HIGHEST_PROTOCOL)
            #Update alive_bar
            bar()
    #Define estimated function
    def u(xt):
        return forward(xt,params)

    return {'u': u,'params': params,'forward': forward,'time': time.time() - t0}

#Process result
def process_result(test_data,fit,train_data,plot = True,times = 5,d2 = True,save = False,show = True,file_name = 'result_pinn',print_res = True):
    """
    Process the results of a Physics-informed Neural Network
    ----------

    Parameters
    ----------
    test_data : dict

        A dictionay with test data for L2 error calculation generated by the jinnax.data.generate_PINNdata function

    fit : function

        The fitted function

    train_data : dict

        Training data generated by the jinnax.data.generate_PINNdata

    plot : logical

        Whether to generate plots comparing the exact and estimated solutions when the spatial dimension is one. Default True

    times : int

        Number of points along the time interval to plot. Default 5

    d2 : logical

        Whether to plot 2D plot when the spatial dimension is one. Default True

    save : logical

        Whether to save the plots. Default False

    show : logical

        Whether to show the plots. Default True

    file_name : str

        File prefix to save the plots. Default 'result_pinn'

    print_res : logical

        Whether to print the L2 error. Default True

    Returns
    -------
    pandas data frame with L2 and MSE errors
    """

    #Dimension
    d = test_data['xt'].shape[1] - 1

    #Number of plots multiple of 5
    times = 5 * round(times/5.0)

    #Data
    td = get_train_data(train_data)
    xt_train = td['x']
    u_train = td['y']
    upred_train = fit(xt_train)
    upred_test = fit(test_data['xt'])

    #Results
    l2_error_test = L2error(upred_test,test_data['u']).tolist()
    MSE_test = jnp.mean(MSE(upred_test,test_data['u'])).tolist()
    l2_error_train = L2error(upred_train,u_train).tolist()
    MSE_train = jnp.mean(MSE(upred_train,u_train)).tolist()

    df = pd.DataFrame(np.array([l2_error_test,MSE_test,l2_error_train,MSE_train]).reshape((1,4)),
        columns=['l2_error_test','MSE_test','l2_error_train','MSE_train'])
    if print_res:
        print('L2 error test: ' + str(jnp.round(l2_error_test,6)) + ' L2 error train: ' + str(jnp.round(l2_error_train,6)) + ' MSE error test: ' + str(jnp.round(MSE_test,6)) + ' MSE error train: ' + str(jnp.round(MSE_train,6)) )

    #Plots
    if d == 1 and plot:
        plot_pinn1D(times,test_data['xt'],test_data['u'],upred_test,d2,save,show,file_name)

    return df

#Plot results for d = 1
def plot_pinn1D(times,xt,u,upred,d2 = True,save = False,show = True,file_name = 'result_pinn',title_1d = '',title_2d = ''):
    """
    Plot the prediction of a 1D PINN
    ----------

    Parameters
    ----------
    times : int

        Number of points along the time interval to plot. Default 5

    xt : jax.numpy.array

        Test data xt array

    u : jax.numpy.array

        Test data u(x,t) array

    upred : jax.numpy.array

        Predicted upred(x,t) array on test data

    d2 : logical

        Whether to plot 2D plot. Default True

    save : logical

        Whether to save the plots. Default False

    show : logical

        Whether to show the plots. Default True

    file_name : str

        File prefix to save the plots. Default 'result_pinn'

    title_1d : str

        Title of 1D plot

    title_2d : str

        Title of 2D plot

    Returns
    -------
    pandas data frame with L2 error
    """
    #Initialize
    fig, ax = plt.subplots(int(times/5),5,figsize = (10*int(times/5),3*int(times/5)))
    tlo = jnp.min(xt[:,-1])
    tup = jnp.max(xt[:,-1])
    ylo = jnp.min(u)
    ylo = ylo - 0.1*jnp.abs(ylo)
    yup = jnp.max(u)
    yup = yup + 0.1*jnp.abs(yup)
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

    #Title
    fig.suptitle(title_1d)
    fig.tight_layout()

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

        #Title
        fig.suptitle(title_2d)
        fig.tight_layout()

        #Show and save
        fig = plt.gcf()
        if show:
            plt.show()
        if save:
            fig.savefig(file_name + '_2d.png')
        plt.close()

#Get train data in one array
def get_train_data(train_data):
    """
    Process training sample
    ----------

    Parameters
    ----------
    train_data : dict

        A dictionay with train data generated by the jinnax.data.generate_PINNdata function

    Returns
    -------
    dict with the processed training data
    """
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

    return {'xy': xydata,'x': xdata,'y': ydata,'sensor_sample': sensor_sample,'boundary_sample': boundary_sample,'initial_sample': initial_sample,'collocation_sample': collocation_sample}

#Process training
def process_training(test_data,file_name,at_each = 100,bolstering = True,bias = None,mc_sample = 10000,save = False,file_name_save = 'result_pinn',key = 0,ec = 1e-6,lamb = 1):
    """
    Process the training of a Physics-informed Neural Network
    ----------

    Parameters
    ----------
    test_data : dict

        A dictionay with test data for L2 error calculation generated by the jinnax.data.generate_PINNdata function

    file_name : str

        Name of the files saved during training

    at_each : int

        Compute results for epochs multiple of at_each. Default 100

    bolstering : logical

        Whether to compute bolstering mean square error. Default True

    bias: float

        Bias for kernel estimation in bolstering via the Hessian method

    mc_sample : int

        Number of sample for Monte Carlo integration in bolstering. Default 10000

    save : logical

        Whether to save the training results. Default False

    file_name_save : str

        File prefix to save the plots and the L2 error. Default 'result_pinn'

    key : int

        Key for random samples in bolstering. Default 0

    ec : float

        Stopping criteria error for EM algorithm in bolstering. Default 1e-6

    lamb : float

        Hyperparameter of EM algorithm in bolstering. Default 1

    Returns
    -------
    pandas data frame with training results
    """
    #Config
    config = pickle.load(open(file_name + '_config.pickle', 'rb'))
    epochs = config['epochs']
    train_data = config['train_data']
    forward = config['forward']

    #Generate keys
    if bolstering:
        keys = jax.random.randint(random.PRNGKey(key),(epochs,),0,1e9)

    #Get train data
    td = get_train_data(train_data)
    xydata = td['xy']
    xdata = td['x']
    ydata = td['y']
    sensor_sample = td['sensor_sample']
    boundary_sample = td['boundary_sample']
    initial_sample = td['initial_sample']
    collocation_sample = td['collocation_sample']

    #Bias in bolstering
    if bias is None and bolstering:
        bias = 1/jnp.sqrt(xdata.shape[0])

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
                params = pickle.load(open(file_name + '_epoch' + str(e).rjust(6, '0') + '.pickle','rb'))

                #Time
                time = time + [params['time']]

                #Define learned function
                def psi(x):
                    return forward(x,params['params'])

                #Train MSE and L2
                if xdata is not None:
                    train_mse = train_mse + [jnp.mean(MSE(psi(xdata),ydata)).tolist()]
                    train_L2 = train_L2 + [L2error(psi(xdata),ydata).tolist()]
                else:
                    train_mse = train_mse + [None]
                    train_L2 = train_L2 + [None]

                #Test MSE and L2
                test_mse = test_mse + [jnp.mean(MSE(psi(test_data['xt']),test_data['u'])).tolist()]
                test_L2 = test_L2 + [L2error(psi(test_data['xt']),test_data['u']).tolist()]

                #Bolstering
                if bolstering:
                    kxy = gk.kernel_estimator(xydata,random.PRNGKey(keys[e]),method = "hessian",lamb = lamb,ec = ec,psi = psi,bias = bias)
                    kx = kxy[:,:-1,:-1]
                    bolstX = bolstX + [gb.bolstering(psi,xdata,ydata,kx,random.PRNGKey(keys[e]),mc_sample = mc_sample).tolist()]
                    bolstXY = bolstXY + [gb.bolstering(psi,xdata,ydata,kxy,random.PRNGKey(keys[e]),mc_sample = mc_sample).tolist()]
                else:
                    bolstX = bolstX + [None]
                    bolstXY = bolstXY + [None]

                #Loss
                loss = loss + [params['loss'].tolist()]

                #Delete
                del params, psi
            #Update alive_bar
            bar()

    #Create data frame
    df = pd.DataFrame(np.column_stack([ep,time,[sensor_sample] * len(ep),[boundary_sample] * len(ep),[initial_sample] * len(ep),[collocation_sample] * len(ep),loss,
    train_mse,test_mse,train_L2,test_L2,bolstX,bolstXY]),
        columns=['epoch','training_time','sensor_sample','boundary_sample','initial_sample','collocation_sample','loss','train_mse','test_mse','train_L2','test_L2','bolstX','bolstXY'])
    if save:
        df.to_csv(file_name_save + '.csv',index = False)

    return df

#Demo video for training1D PINN
def demo_train_pinn1D(test_data,file_name,at_each = 100,times = 5,d2 = True,file_name_save = 'result_pinn_demo',title = '',framerate = 10):
    """
    Demo video with the training of a 1D PINN
    ----------

    Parameters
    ----------
    test_data : dict

        A dictionay with test data for L2 error calculation generated by the jinnax.data.generate_PINNdata function

    file_name : str

        Name of the files saved during training

    at_each : int

        Compute results for epochs multiple of at_each. Default 100

    times : int

        Number of points along the time interval to plot. Default 5

    d2 : logical

        Whether to make video demo of 2D plot. Default True

    file_name_save : str

        File prefix to save the plots and videos. Default 'result_pinn_demo'

    title : str

        Title for plots

    framerate : int

        Framerate for video. Default 10

    Returns
    -------
    None
    """
    #Config
    with open(file_name + '_config.pickle', 'rb') as file:
        config = pickle.load(file)
    epochs = config['epochs']
    train_data = config['train_data']
    forward = config['forward']

    #Get train data
    td = get_train_data(train_data)
    xt = td['x']
    u = td['y']

    #Create folder to save plots
    os.system('mkdir ' + file_name_save)

    #Create images
    k = 1
    with alive_bar(epochs) as bar:
        for e in range(epochs):
            if e % at_each == 0 or e == epochs - 1:
                #Read parameters
                params = pd.read_pickle(file_name + '_epoch' + str(e).rjust(6, '0') + '.pickle')

                #Define learned function
                def psi(x):
                    return forward(x,params['params'])

                #Compute L2 train, L2 test and loss
                loss = params['loss']
                L2_train = L2error(psi(xt),u)
                L2_test = L2error(psi(test_data['xt']),test_data['u'])
                title_epoch = title + ' Epoch = ' + str(e) + ' L2 train = ' + str(round(L2_train,6)) + ' L2 test = ' + str(round(L2_test,6))

                #Save plot
                plot_pinn1D(times,test_data['xt'],test_data['u'],psi(test_data['xt']),d2,save = True,show = False,file_name = file_name_save + '/' + str(k),title_1d = title_epoch,title_2d = title_epoch)
                k = k + 1

                #Delete
                del params, psi, loss, L2_train, L2_test, title_epoch
            #Update alive_bar
            bar()
    #Create demo video
    os.system('ffmpeg -framerate ' + str(framerate) + ' -i ' + file_name_save + '/' + '%00d_slices.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p ' + file_name_save + '/' + file_name_save + '_slices.mp4')
    if d2:
        os.system('ffmpeg -framerate ' + str(framerate) + ' -i ' + file_name_save + '/' + '%00d_2d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p ' + file_name_save + '/' + file_name_save + '_2d.mp4')

#Demo in time for 1D PINN
def demo_time_pinn1D(test_data,file_name,epochs,file_name_save = 'result_pinn_time_demo',title = '',framerate = 10):
    """
    Demo video with the time evolution of a 1D PINN
    ----------

    Parameters
    ----------
    test_data : dict

        A dictionay with test data for L2 error calculation generated by the jinnax.data.generate_PINNdata function

    file_name : str

        Name of the files saved during training

    epochs : list

        Which training epochs to plot

    file_name_save : str

        File prefix to save the plots and video. Default 'result_pinn_time_demo'

    title : str

        Title for plots

    framerate : int

        Framerate for video. Default 10

    Returns
    -------
    None
    """
    #Config
    with open(file_name + '_config.pickle', 'rb') as file:
        config = pickle.load(file)
    train_data = config['train_data']
    forward = config['forward']

    #Create folder to save plots
    os.system('mkdir ' + file_name_save)

    #Plot parameters
    tdom = jnp.unique(test_data['xt'][:,-1])
    ylo = jnp.min(test_data['u'])
    ylo = ylo - 0.1*jnp.abs(ylo)
    yup = jnp.max(test_data['u'])
    yup = yup + 0.1*jnp.abs(yup)

    #Open PINN for each epoch
    results = []
    upred = []
    for e in epochs:
        tmp = pd.read_pickle(file_name + '_epoch' + str(e).rjust(6, '0') + '.pickle')
        results = results + [tmp]
        upred = upred + [forward(test_data['xt'],tmp['params'])]

    #Create images
    k = 1
    with alive_bar(len(tdom)) as bar:
        for t in tdom:
            #Test data
            xt_step = test_data['xt'][test_data['xt'][:,-1] == t]
            u_step = test_data['u'][test_data['xt'][:,-1] == t]
            #Initialize plot
            if len(epochs) > 1:
                fig, ax = plt.subplots(int(len(epochs)/2),2,figsize = (10,5*len(epochs)/2))
            else:
                fig, ax = plt.subplots(1,1,figsize = (10,5))
            #Create
            index = 0
            if int(len(epochs)/2) > 1:
                for i in range(int(len(epochs)/2)):
                    for j in range(min(2,len(epochs))):
                        upred_step = upred[index][test_data['xt'][:,-1] == t]
                        ax[i,j].plot(xt_step[:,0],u_step[:,0],'b-',linewidth=2,label='Exact')
                        ax[i,j].plot(xt_step[:,0],upred_step[:,0],'r--',linewidth=2,label='Prediction')
                        ax[i,j].set_title('Epoch = ' + str(epochs[index]),fontsize=10)
                        ax[i,j].set_xlabel(' ')
                        ax[i,j].set_ylim([1.3 * ylo.tolist(),1.3 * yup.tolist()])
                        index = index + 1
            elif len(epochs) > 1:
                for j in range(2):
                    upred_step = upred[index][test_data['xt'][:,-1] == t]
                    ax[j].plot(xt_step[:,0],u_step[:,0],'b-',linewidth=2,label='Exact')
                    ax[j].plot(xt_step[:,0],upred_step[:,0],'r--',linewidth=2,label='Prediction')
                    ax[j].set_title('Epoch = ' + str(epochs[index]),fontsize=10)
                    ax[j].set_xlabel(' ')
                    ax[j].set_ylim([1.3 * ylo.tolist(),1.3 * yup.tolist()])
                    index = index + 1
            else:
                upred_step = upred[index][test_data['xt'][:,-1] == t]
                ax.plot(xt_step[:,0],u_step[:,0],'b-',linewidth=2,label='Exact')
                ax.plot(xt_step[:,0],upred_step[:,0],'r--',linewidth=2,label='Prediction')
                ax.set_title('Epoch = ' + str(epochs[index]),fontsize=10)
                ax.set_xlabel(' ')
                ax.set_ylim([1.3 * ylo.tolist(),1.3 * yup.tolist()])
                index = index + 1


            #Title
            fig.suptitle(title + 't = ' + str(round(t,4)))
            fig.tight_layout()

            #Show and save
            fig = plt.gcf()
            fig.savefig(file_name_save + '/' + str(k) + '.png')
            k = k + 1
            plt.close()
            bar()

    #Create demo video
    os.system('ffmpeg -framerate ' + str(framerate) + ' -i ' + file_name_save + '/' + '%00d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p ' + file_name_save + '/' + file_name_save + '_time_demo.mp4')
