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
    @jax.jit
    def forward(x,params):
      *hidden,output = params
      for layer in hidden:
        x = activation(x @ layer['W'] + layer['B'])
      return x @ output['W'] + output['B']

    #Return initial parameters and forward function
    return {'params': params,'forward': forward}

#Apply a morphological layer
def apply_morph_layer(x,type,params,index_x):
    #Define which operator will be applied
    oper = mp.operator(type)
    oper = jax.vmap(oper,in_axes = (None,None,0),out_axes = 0)
    fx = oper(x,index_x,jax.nn.sigmoid(params))
    return fx

#Canonical Morphological NN
def cmnn(type,width,size,shape_x,key = 0):
    #Index window
    index_x = mp.index_array(shape_x)

    #Initialize parameters with Glorot initialization
    initializer = jax.nn.initializers.glorot_normal()
    k = jax.random.split(jax.random.PRNGKey(key),(max(width))) #Seed for initialization
    params = list()
    for i in range(len(width)):
        if type[i] == 'supgen' or type[i] == 'infgen':
            params.append(initializer(k[i,:],(width[i],2,size[i],size[i]),jnp.float32))
        else:
            params.append(initializer(k[i,:],(width[i],1,size[i],size[i]),jnp.float32))

    #Forward pass
    @jax.jit
    def forward(x,params):
        x = x.reshape((1,x.shape[0],x.shape[1],x.shape[2]))
        for i in range(len(type)):
            #Apply sup and inf
            if type[i] == 'sup':
                x = mp.vmap_sup(x)
            elif type[i] == 'inf':
                x = mp.vmap_inf(x)
            elif type[i] == 'complement':
                x = 1 - x
            else:
                #Apply other layer
                x = apply_morph_layer(x[0,:,:,:],type[i],params[i],index_x)
        return x[0,:,:,:]

    #Return initial parameters and forward function
    return {'params': params,'forward': forward}
