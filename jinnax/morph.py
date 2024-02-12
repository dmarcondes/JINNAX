#Morphology module
import jax
import jax.numpy as jnp
import math


#Structuring element from function
def struct_function(k,d):
    w = jnp.array([[x1.tolist(),x2.tolist()] for x1 in jnp.linspace(-jnp.floor(d/2),jnp.floor(d/2),d) for x2 in jnp.linspace(jnp.floor(d/2),-jnp.floor(d/2),d)])
    k = jnp.array(k(w))
    return jnp.transpose(k.reshape((d,d)))

#Create an index array for an array
def index_array(shape):
    return jnp.array([[x,y] for x in range(shape[0]) for y in range(shape[1])])

#Local erosion of f by k for pixel (i,j)
def local_erosion(f,k,l):
    def jit_local_erosion(index):
        fw = jax.lax.dynamic_slice(f, (index[0] - l, index[1] - l), (2*l + 1, 2*l + 1))
        return jnp.minimum(jnp.maximum(jnp.min(fw - k),0.0),1.0)
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
        return jnp.minimum(jnp.maximum(jnp.max(fw + k),0.0),1.0)
    return jit_local_dilation

#Dilation of f by k
def dilation(f,index_f,k):
    l = math.floor(k.shape[0]/2)
    jit_local_dilation = local_dilation(f,k,l)
    return jax.numpy.apply_along_axis(jit_local_dilation,1,index_f).reshape(f.shape)

#Opening of f by k
def opening(f,index_f,k):
    l = math.floor(k.shape[0]/2)
    jit_local_erosion = local_erosion(f,k,l)
    fe = jax.numpy.apply_along_axis(jit_local_erosion,1,index_f).reshape(f.shape)
    jit_local_dilation = local_dilation(fe,k,l)
    fed = jax.numpy.apply_along_axis(jit_local_dilation,1,index_f).reshape(f.shape)
    return fed

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
