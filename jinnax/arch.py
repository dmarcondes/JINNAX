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
def apply_morph_layer(x,type[i],params[i],index_x):
#Define which operator will be applied
oper = jmp.operator(type)
if type == 'supgen' or type == 'ingen':
oper = jax.vmap(lambda k: oper(x,index_x,k),in_axes = (0),out_axes = 0)
else:
    oper = jax.vmap(lambda k: oper(f,index_f,k[0,:,:],None),in_axes = (0),out_axis = 0)
params.shape
oper(params)
k1 = params[0,0,:,:]
k2 = params[0,1,:,:]
oper(x,index_x,k)
fx = oper(x,index_x,jax.nn.sigmoid(k[0,:,:]),jax.nn.sigmoid(k[1,:,:]))
k = params[0,:,:,:]
f(k)
#AQUI!

#Reshape x
x = x[0,:,:,:]
#List nodes
array_width = jnp.array(range(width),dtype = jnp.int8).reshape((width,1))
#Function to apply operator to all nodes
if type == 'supgen' or type == 'ingen':
apply_layer = jax.vmap(lambda f,k: oper(f,index_f,k),in_axes = (0,0),out_axes = 0)
else:
    apply_layer = jax.vmap(lambda i: apply_node(x,forward[p + i],None,params[p + i],None,w,d,index_x,oper,type),in_axes = (0),out_axes = 0)

#Apply
fx = apply_layer(array_width)
#Update p
if type == 'supgen' or type == 'ingen':
    p = p + 2*len(width)
else:
    p = p + len(width)
return {'x': fx,'p': p}

#Canonical Morphological NN
def cmnn(type,width,width_str,size,shape_x,activation = jax.nn.tanh,key = 0):
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
def forward(x,params):
x = x.reshape((1,x.shape[0],x.shape[1],x.shape[2]))
for i in range(len(type)):
    #Apply sup and inf
    if type == 'sup':
        x = mp.vmap_sup(x)
    elif type == 'inf':
        x = mp.vmap_inf(x)
    elif type == 'complement':
        x = 1 - x
    else:
        #Apply other layer
        x = jax.vmap(lambda x: apply_morph_layer(x,type[i],params[i],index_x),in_axes = (0),out_axes = 0)(x[0,:,:,:])
return x[0,:,:,:]

#Return initial parameters and forward function
return {'params': params,'forward': forward}
