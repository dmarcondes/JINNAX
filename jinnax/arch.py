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

#Apply a morphological layer
def apply_morph_layer(x,type,width,params,p,w):
#Define which operator will be applied
if type == 'erosion':
    oper = jax.jit(mp.erosion)
elif type == 'dilation':
    oper = jax.jit(mp.dilation)
elif type == 'opening':
    oper = jax.jit(mp.opening)
elif type == 'closing':
    oper = jax.jit(mp.closing)
elif type == 'asf':
    oper = jax.jit(mp.asf)
elif type == 'complement':
    oper = jax.jit(mp.complement)
elif type == 'supgen':
    oper = jax.jit(mp.supgen)
elif type == 'infgen':
    oper = jax.jit(mp.infgen)
elif type == 'sup':
    oper = jax.jit(mp.sup)
elif type == 'inf':
    oper = jax.jit(mp.inf)
else:
    print('Type of layer ' + type + 'is wrong!')
    return 1

#Apply sup or inf
if type == 'inf' or type == 'sup':
    x = jax.numpy.apply_along_axis(oper,0,x).reshape((1,x.shape[1],x.shape[2],x.shape[3])
#Apply each operator

for i in range(width):

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
        params.append({'params': jnp.array([0.0],dtype = jnp.float32),'forward': None})
    elif type[i] == 'inf':
        params.append({'params': jnp.array([0.0],dtype = jnp.float32),'forward': None})

#Create w to apply str NN
unique_size = set(size)
w = {}
for d in unique_size:
    if d != 1:
        w[str(d)] = jnp.array([[x1.tolist(),x2.tolist()] for x1 in jnp.linspace(-jnp.floor(d/2),jnp.floor(d/2),d) for x2 in jnp.linspace(jnp.floor(d/2),-jnp.floor(d/2),d)])

#Forward pass
def forward(params,x):
p = 0
x = x.reshape((1,x.shape[0],x.shape[1],x.shape[2]))
for i in range(len(type)):
    #Apply layer
    x = apply_morph_layer(x,type[i],width[i],params,p,w)
    #Update counter
    if type[i] == 'supgen' or type[i] == 'infgen':
        p = p + 2
    else:
        p = p + 1
return x
