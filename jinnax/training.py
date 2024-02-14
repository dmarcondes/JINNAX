#Functions to train NN
import jax
import jax.numpy as jnp
import optax
from alive_progress import alive_bar
import math

#MSE
@jax.jit
def MSE(true,pred):
  return jnp.mean((true - pred)**2)

#MSE 255
@jax.jit
def MSE255(true,pred):
  return jnp.mean(((true - pred)/255) ** 2)

#Training function
def train_morph(x,y,forward,params,loss,epochs = 1,batches = 1,lr = 0.001,b1 = 0.9,b2 = 0.999,eps = 1e-08,eps_root = 0.0,key = 0):
    #Key
    key = jax.random.split(jax.random.PRNGKey(key),epochs)

    #Batch size
    bsize = int(math.floor(x.shape[0]/batches))

    #Optmizer NN
    optimizer = optax.adam(lr,b1,b2,eps,eps_root)
    opt_state = optimizer.init(params)

    #Loss function
    @jax.jit
    def lf(params,x,y):
        return loss(forward(x,params),y)

    #Training function
    grad_loss = jax.jit(jax.grad(lf,0))
    @jax.jit
    def update(opt_state,params,x,y):
      grads = grad_loss(params,x,y)
      updates, opt_state = optimizer.update(grads, opt_state)
      params = optax.apply_updates(params, updates)
      return opt_state,params

    #Train
    with alive_bar(epochs) as bar:
        for e in range(epochs):
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
            l = str(jnp.round(lf(params,x,y),10))
            bar.title("Loss: " + l)
            bar()

    return params
