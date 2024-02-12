#Functions to train NN
import jax
import optax
from alive_progress import alive_bar

#MSE
@jax.jit
def MSE(true,pred):
  return jnp.mean((true - pred)**2)

#Training function
def train_morph(x,y,forward,params,loss,epochs = 1,batches = 1,lr = 0.1,b1 = 0.9,b2 = 0.999,eps = 1e-08,eps_root = 0.0):
    #Isolate parameters and redefine forward function
    allp = params
    params = [allp[i]['params'] for i in range(len(params))]
    @jax.jit
    def new_forward(x,params):
        p = [{'params': params[i],'forward': allp[i]['forward']} for i in range(len(params))]
        return forward(x,p)

    #Optmizer NN
    optimizer = optax.adam(lr,b1,b2,eps,eps_root)
    opt_state = optimizer.init(params)

    #Loss function
    @jax.jit
    def lf(params,x,y):
        return loss(new_forward(x,params),y)

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
            opt_state,params = update(opt_state,params,x,y)
            bar()

    return [{'params': params[i],'forward': allp[i]['forward']} for i in range(len(params))]
