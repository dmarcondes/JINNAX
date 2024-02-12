#Functions to define NN architectures
import jax
import jax.numpy as jnp
import jinnax.morph as mp
import math

#Simple fully connected architecture. Return the function for the forward pass
def fconNN(width,activation = jax.nn.tanh,key = 0):
    #Initialize parameters with Glorot initialization
    initializer = jax.nn.initializers.glorot_normal()
    key = jax.random.split(jax.random.PRNGKey(key),len(width)-1) #Seed for initialization
    params = list()
    for key,lin,lout in zip(key,width[:-1],width[1:]):
        W = initializer(key,(lin,lout),jnp.float32)
        B = initializer(key,(1,lout),jnp.float32)
        params.append({'W':W,'B':B})

    #Define function for forward pass
    def forward(params,x):
      *hidden,output = params
      for layer in hidden:
        x = activation(x @ layer['W'] + layer['B'])
      return x @ output['W'] + output['B']

    #Return initial parameters and forward function
    return {'params': params,'forward': forward}

#Fully connected architecture for structural element
def fconNN_str(width,activation = jax.nn.tanh,key = 0):
    #Add first and last layer
    width = [2] + width + [1]

    #Initialize parameters with Glorot initialization
    initializer = jax.nn.initializers.glorot_normal()
    key = jax.random.split(jax.random.PRNGKey(key),len(width)-1) #Seed for initialization
    params = list()
    for key,lin,lout in zip(key,width[:-1],width[1:]):
        W = initializer(key,(lin,lout),jnp.float32)
        B = initializer(key,(1,lout),jnp.float32)
        params.append({'W':W,'B':B})

    #Define function for forward pass
    def forward(params,x):
      *hidden,output = params
      for layer in hidden:
        x = activation(x @ layer['W'] + layer['B'])
      return jax.nn.sigmoid(x @ output['W'] + output['B'])

    #Return initial parameters and forward function
    return {'params': params,'forward': forward}

#Canonical Morphological NN
def cmnn(type,width,width_str,size,activation = jax.nn.tanh,key = 0):
#Initialize parameters with Glorot initialization
initializer = jax.nn.initializers.glorot_normal()
k = jax.random.split(jax.random.PRNGKey(key),(len(type),max(width))) #Seed for initialization
params = list()
for i in range(len(type)):
    if type[i] != 'sup' and type[i] != 'inf':
        for j in range(width[i]):
            params.append(fconNN_str(width_str,activation,k[i,j,0]))
            if type[i] == 'supgen' or type[i] == 'infgen':
                params.append(fconNN_str(width_str,activation,k[i,j,1]))
    elif type[i] == 'sup':
        params.append({'params': None,'forward': mp.sup})
    elif type[i] == 'inf':
        params.append({'params': None,'forward': mp.inf})


for key,lin,lout in zip(key,width[:-1],width[1:]):
    W = initializer(key,(lin,lout),jnp.float32)
    B = initializer(key,(1,lout),jnp.float32)
    params.append({'W':W,'B':B})
