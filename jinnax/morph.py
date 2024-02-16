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
        fw = jax.lax.dynamic_slice(f, (index[0], index[1]), (2*l + 1, 2*l + 1))
        return jnp.minimum(jnp.maximum(jnp.min(fw - k),0.0),1.0)
    return jit_local_erosion

#Erosion of f by k
@jax.jit
def erosion_2D(f,index_f,k):
    l = math.floor(k.shape[0]/2)
    jit_local_erosion = local_erosion(f,k,l)
    return jnp.apply_along_axis(jit_local_erosion,1,index_f).reshape((f.shape[0] - 2*l,f.shape[1] - 2*l))

#Erosion in batches
@jax.jit
def erosion(f,index_f,k):
    l = math.floor(k.shape[0]/2)
    f = jax.lax.pad(f,0.0,((0,0,0),(l,l,0),(l,l,0)))
    eb = jax.vmap(lambda f: erosion_2D(f,index_f,k),in_axes = (0),out_axes = 0)(f)
    return eb

#Local dilation of f by k for pixel (i,j)
def local_dilation(f,k,l):
    def jit_local_dilation(index):
        fw = jax.lax.dynamic_slice(f, (index[0], index[1]), (2*l + 1, 2*l + 1))
        return jnp.minimum(jnp.maximum(jnp.max(fw + k),0.0),1.0)
    return jit_local_dilation

#Dilation of f by k
@jax.jit
def dilation_2D(f,index_f,k):
    l = math.floor(k.shape[0]/2)
    jit_local_dilation = local_dilation(f,k,l)
    return jnp.apply_along_axis(jit_local_dilation,1,index_f).reshape((f.shape[0] - 2*l,f.shape[1] - 2*l))

#Dilation in batches
@jax.jit
def dilation(f,index_f,k):
    l = math.floor(k.shape[0]/2)
    f = jax.lax.pad(f,0.0,((0,0,0),(l,l,0),(l,l,0)))
    db = jax.vmap(lambda f: dilation_2D(f,index_f,k),in_axes = (0),out_axes = 0)(f)
    return db

#Opening of f by k
def opening(f,index_f,k):
    l = math.floor(k.shape[0]/2)
    f = jax.lax.pad(f,0.0,((0,0,0),(l,l,0),(l,l,0)))
    eb = jax.vmap(lambda f: erosion_2D(f,index_f,k),in_axes = (0),out_axes = 0)
    db = jax.vmap(lambda f: dilation_2D(f,index_f,k),in_axes = (0),out_axes = 0)
    f = eb(f)
    f = jax.lax.pad(f,0.0,((0,0,0),(l,l,0),(l,l,0)))
    return db(f)

#Colosing of f by k
def closing(f,index_f,k):
    l = math.floor(k.shape[0]/2)
    f = jax.lax.pad(f,0.0,((0,0,0),(l,l,0),(l,l,0)))
    eb = jax.vmap(lambda f: erosion_2D(f,index_f,k),in_axes = (0),out_axes = 0)
    db = jax.vmap(lambda f: dilation_2D(f,index_f,k),in_axes = (0),out_axes = 0)
    f = db(f)
    f = jax.lax.pad(f,0.0,((0,0,0),(l,l,0),(l,l,0)))
    return eb(f)

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
def sup(f,leak = (1/255) ** 2):
    fs = f[0,:,:]
    for i in range(f.shape[0] - 1):
        fs = 0.5 * (fs + f[i,:,:] + jnp.sqrt((fs - f[i,:,:]) ** 2 + leak))
    return fs.reshape((1,f.shape[1],f.shape[2]))

#Sup vmap for arch
vmap_sup = jax.jit(jax.vmap(sup,in_axes = (1),out_axes = 1))

#Inf of array of images
@jax.jit
def inf(f):
    return - sup(-f)

#Inf vmap for arch
vmap_inf = jax.jit(jax.vmap(inf,in_axes = (1),out_axes = 1))

#Return operator by name
def operator(type):
    if type == 'erosion':
        oper = lambda x,index_x,k: erosion(x,index_x,jax.lax.slice_in_dim(k,0,1).reshape((k.shape[1],k.shape[2])))
    elif type == 'dilation':
        oper = lambda x,index_x,k: dilation(x,index_x,jax.lax.slice_in_dim(k,0,1).reshape((k.shape[1],k.shape[2])))
    elif type == 'opening':
        oper = lambda x,index_x,k: opening(x,index_x,jax.lax.slice_in_dim(k,0,1).reshape((k.shape[1],k.shape[2])))
    elif type == 'closing':
        oper = lambda x,index_x,k: closing(x,index_x,jax.lax.slice_in_dim(k,0,1).reshape((k.shape[1],k.shape[2])))
    elif type == 'asf':
        oper = lambda x,index_x,k: asf(x,index_x,jax.lax.slice_in_dim(k,0,1).reshape((k.shape[1],k.shape[2])))
    elif type == 'supgen':
        oper = lambda x,index_x,k: supgen(x,index_x,jax.lax.slice_in_dim(k,0,1).reshape((k.shape[1],k.shape[2])),jax.lax.slice_in_dim(k,1,2).reshape((k.shape[1],k.shape[2])))
    elif type == 'infgen':
        oper = lambda x,index_x,k: infgen(x,index_x,jax.lax.slice_in_dim(k,0,1).reshape((k.shape[1],k.shape[2])),jax.lax.slice_in_dim(k,1,2).reshape((k.shape[1],k.shape[2])))
    else:
        print('Type of layer ' + type + 'is wrong!')
        return 1
    return oper

#Structuring element of the approximate identity operator in a sample
def struct_lower(x,d):
    #Function to apply to each index
    l = math.floor(d/2)
    x = jax.lax.pad(x,0.0,((0,0,0),(l,l,0),(l,l,0)))
    index_x = index_array((x.shape[1],x.shape[2]))
    def struct_lower(index,x):
        fw = jax.lax.dynamic_slice(x, (index[0] - l, index[1] - l), (2*l + 1, 2*l + 1))
        return fw - x[index[0],index[1]]
    k = jax.vmap(lambda x: jnp.apply_along_axis(lambda index: struct_lower(index,x),1,index_x))(x).reshape((x.shape[0],x.shape[1],x.shape[2],3,3))
    k = k.reshape((k.shape[0]*k.shape[1]*k.shape[2],d,d))
    k = jnp.apply_along_axis(lambda k: jnp.percentile(k,10),0,k)
    return k

#Structuring element of upper limit of interval of supgen approximating identity operator
def struct_upper(x,d):
    #Function to apply to each index
    l = math.floor(d/2)
    x = jax.lax.pad(x,0.0,((0,0,0),(l,l,0),(l,l,0)))
    index_x = index_array((x.shape[1],x.shape[2]))
    def struct_lower(index,x):
        fw = jax.lax.dynamic_slice(x, (index[0] - l, index[1] - l), (2*l + 1, 2*l + 1))
        return fw + x[index[0],index[1]]
    k = jax.vmap(lambda x: jnp.apply_along_axis(lambda index: struct_lower(index,x),1,index_x))(x).reshape((x.shape[0],x.shape[1],x.shape[2],3,3))
    k = k.reshape((k.shape[0]*k.shape[1]*k.shape[2],d,d))
    k = jnp.apply_along_axis(lambda k: jnp.percentile(k,90),0,k)
    return k
