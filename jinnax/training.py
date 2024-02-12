#Functions to train NN
import optax

#MSE
@jax.jit
def MSE(true,pred):
  return jnp.mean((true - pred)**2)

#Morphology MSE
@jax.jit
def mse_morph(params,x,y,forward):
    fx = forward(x,params)
    return MSE(y,fx)

#Training function
def train(x,y,forward,params,loss,epochs,batches = 1,lr = 0.001,b1 = 0.9,b2 = 0.999,eps = 1e-08,eps_root = 0.0):
#Optmizer NN
optimizer = optax.adam(lr,b1,b2,eps,eps_root)
opt_state = optimizer.init(params)

#Loss function
loss = jax.jit(loss)

#Training function
grad_loss = jax.jit(jax.grad(loss,0))
@jax.jit
def update(opt_state,params):
  grads = grad_loss(params,xt_sensor,u_sensor,xt_collocation,xt_boundary,u_boundary,xt_initial,u_initial)
  updates, opt_state = optimizer.update(grads, opt_state)
  params = optax.apply_updates(params, updates)
  return opt_state,params

#Train
init = time.time()
for epoch in range(num_epochs):
  opt_state,params = update(opt_state,params)
  #Print
  if(epoch % 10000 == 0):
    print("Case: "+str(case)+" Epoch: "+str(epoch)+" NN Loss: "+str(round(loss(params,xt_sensor,u_sensor,xt_collocation,xt_boundary,u_boundary,xt_initial,u_initial),5))+" Time: "+str(round((time.time()-init),5)))
