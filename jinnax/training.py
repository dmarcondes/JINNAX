#Functions to train NN
import optax

#MSE
@jit
def MSE(true,pred):
  return jnp.mean((true - pred)**2)

#Training function
def train(x,forward,params,loss,epochs,batches = 1):
