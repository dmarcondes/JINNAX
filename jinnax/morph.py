#Morphology module
import jax
import jax.numpy as jnp
import math


#Structuring element from function
def struct_function(k,d):
    w = jnp.array([[x1.tolist(),x2.tolist()] for x1 in jnp.linspace(-jnp.floor(d/2),jnp.floor(d/2),d) for x2 in jnp.linspace(jnp.floor(d/2),-jnp.floor(d/2),d)])
    k = jnp.array(k(w))
    return jnp.transpose(k.reshape((d,d)))

def struct_function_w(k,w,d):
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
    return jax.vmap(jit_local_erosion,in_axes = (0),out_axes = 0)

#Erosion of f by k
@jax.jit
def erosion_2D(f,index_f,k):
    l = math.floor(k.shape[0]/2)
    jit_local_erosion = local_erosion(f,k,l)
    return jit_local_erosion(index_f).reshape(f.shape)

#Erosion in batches
@jax.jit
def erosion(f,index_f,k):
    eb = jax.vmap(lambda f: erosion_2D(f,index_f,k),in_axes = (0),out_axes = 0)
    return eb(f)

#Local dilation of f by k for pixel (i,j)
def local_dilation(f,k,l):
    def jit_local_dilation(index):
        fw = jax.lax.dynamic_slice(f, (index[0] - l, index[1] - l), (2*l + 1, 2*l + 1))
        return jnp.minimum(jnp.maximum(jnp.max(fw + k),0.0),1.0)
    return jax.vmap(jit_local_dilation,in_axes = (0),out_axes = 0)

#Dilation of f by k
@jax.jit
def dilation_2D(f,index_f,k):
    l = math.floor(k.shape[0]/2)
    jit_local_dilation = local_dilation(f,k,l)
    return jit_local_dilation(index_f).reshape(f.shape)

#Dilation in batches
@jax.jit
def dilation(f,index_f,k):
    db = jax.vmap(lambda f: dilation_2D(f,index_f,k),in_axes = (0),out_axes = 0)
    return db(f)

#Opening of f by k
def opening(f,index_f,k):
    eb = jax.vmap(lambda f: erosion_2D(f,index_f,k),in_axes = (0),out_axes = 0)
    db = jax.vmap(lambda f: dilation_2D(f,index_f,k),in_axes = (0),out_axes = 0)
    f = eb(f)
    return db(f)

#Colosing of f by k
def closing(f,index_f,k):
    fd = dilation(f,index_f,k)
    return erosion(fd,index_f,k)

#Alternate-sequential filter of f by k
def asf(f,index_f,k):
    fo = opening(f,index_f,k)
    return closing(fo,index_f,k)

#Complement
@jax.jit
def complement(f):
    return 1 - f

#Sup-generating with interval [k1,k2]
def supgen(f,index_f,k1,k2):
    K1 = jnp.minimum(k1,k2)
    K2 = jnp.maximum(k1,k2)
    return jnp.minimum(erosion(f,index_f,K1),1 - dilation(f,index_f,1 - K2.transpose()))

#Inf-generating with interval [k1,k2]
def infgen(f,index_f,k1,k2):
    K1 = jnp.minimum(k1,k2)
    K2 = jnp.maximum(k1,k2)
    return jnp.maximum(dilation(f,index_f,K1),1 - erosion(f,index_f,1 - K2.transpose()))

#Sup of array of images
@jax.jit
def sup(f):
    return jnp.max(f,axis = 0).reshape((1,f.shape[1],f.shape[2]))

#Sup vmap for arch
vmap_sup = jax.jit(jax.vmap(sup,in_axes = (1),out_axes = 1))

#Inf of array of images
@jax.jit
def inf(f):
    return jnp.min(f,axis = 0).reshape((1,f.shape[1],f.shape[2]))

#Inf vmap for arch
vmap_inf = jax.jit(jax.vmap(inf,in_axes = (1),out_axes = 1))

#Return operator by name
def operator(type):
    if type == 'erosion':
        oper = mp.erosion
    elif type == 'dilation':
        oper = mp.dilation
    elif type == 'opening':
        oper = mp.opening
    elif type == 'closing':
        oper = mp.closing
    elif type == 'asf':
        oper = mp.asf
    elif type == 'supgen':
        oper = mp.supgen
    elif type == 'infgen':
        oper = mp.infgen
    else:
        print('Type of layer ' + type + 'is wrong!')
        return 1
    return oper
