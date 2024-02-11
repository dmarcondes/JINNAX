#Functions to define NN architectures
import jax
import jax.numpy as jnp
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

#Structuring element from function
def struct_function(k,d):
    w = jnp.array([[x1.tolist(),x2.tolist()] for x1 in jnp.linspace(-jnp.floor(d/2),jnp.floor(d/2),d) for x2 in jnp.linspace(jnp.floor(d/2),-jnp.floor(d/2),d)])
    k = k(w)
    return jnp.transpose(k.reshape((d,d)))

#Local erosion of f by k for pixel (i,j)
def local_erosion(f,k,l):
    def jit_local_erosion(index):
        fw = jax.lax.dynamic_slice(f, (index[0] - l, index[1] - l), (2*l + 1, 2*l + 1))
        return jnp.min(fw - k)
    return jit_local_erosion

#Erosion of f by k
def erosion(f,index_f,k):
    l = math.floor(k.shape[0]/2)
    jit_local_erosion = local_erosion(f,k,l)
    return jax.numpy.apply_along_axis(jit_local_erosion,1,index_f).reshape(f.shape)

#Local dilation of f by k for pixel (i,j)
def local_dilation(f,k,l):
    def jit_local_dilation(index):
        fw = jax.lax.dynamic_slice(f, (index[0] - l, index[1] - l), (2*l + 1, 2*l + 1))
        return jnp.max(fw + k)
    return jit_local_dilation

#Dilation of f by k
def dilation(f,index_f,k):
    l = math.floor(k.shape[0]/2)
    jit_local_dilation = local_dilation(f,k,l)
    return jax.numpy.apply_along_axis(jit_local_dilation,1,index_f).reshape(f.shape)

#Opening of f by k
def opening(f,index_f,k):
    return dilation(erosion(f,index_f,k),index_f,k)

#Coling of f by k
def closing(f,index_f,k):
    return erosion(dilation(f,index_f,k),index_f,k)

#Alternate-sequential filter of f by k
def asf(f,index_f,k):
    return closing(opening(f,index_f,k),index_f,k)

#Complement
def complement(f,m = 1):
    return m - f

#Sup-generating with interval [k1,k2]
def supgen(f,index_f,k1,k2,m):
    return jnp.minimum(erosion(f,index_f,k1),complement(dilation(f,index_f,complement(k2.transpose(),m)),m))

#Inf-generating with interval [k1,k2]
def infgen(f,index_f,k1,k2,m):
    return jnp.maximum(dilation(f,index_f,k1),complement(erosion(f,index_f,complement(k2.transpose(),m),m)))
