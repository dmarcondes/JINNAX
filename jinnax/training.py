#Functions to train NN
import jax
import jax.numpy as jnp
import optax
from alive_progress import alive_bar
import math
import time

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
def train_pinn(data,width,pde,test_data = None,epochs = 100,activation = jax.nn.tanh,lr = 0.001,b1 = 0.9,b2 = 0.999,eps = 1e-08,eps_root = 0.0,key = 0,epoch_print = 100):
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
    with alive_bar(epochs) as bar:
        for e in range(epochs):
            opt_state,params = update(opt_state,params,data)
            if e % epoch_print == 0:
                l = 'Loss: ' + str(jnp.round(lf(params,data),6))
                if test_data is not None:
                    l = l + ' L2 error: ' + str(jnp.round(L2error(forward(test_data['xt'],params),test_data['u']),6))
                print(l)
            bar()

    return params
