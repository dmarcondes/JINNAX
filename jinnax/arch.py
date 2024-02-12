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
    @jax.jit
    def forward(params,x):
      *hidden,output = params
      for layer in hidden:
        x = activation(x @ layer['W'] + layer['B'])
      return jax.nn.sigmoid(x @ output['W'] + output['B'])

    #Return initial parameters and forward function
    return {'params': params,'forward': forward}

#Apply a morphological layer
@jax.jit
def apply_morph_layer(x,type,width,params,p,w,index_x,d):
    #Define which operator will be applied
    if type == 'erosion':
        oper = jax.jit(lambda f,index_f,k1,k2: mp.erosion(f,index_f,k1))
    elif type == 'dilation':
        oper = jax.jit(lambda f,index_f,k1,k2: mp.dilation(f,index_f,k1))
    elif type == 'opening':
        oper = jax.jit(lambda f,index_f,k1,k2: mp.opening(f,index_f,k1))
    elif type == 'closing':
        oper = jax.jit(lambda f,index_f,k1,k2: mp.closing(f,index_f,k1))
    elif type == 'asf':
        oper = jax.jit(lambda f,index_f,k1,k2: mp.asf(f,index_f,k1))
    elif type == 'complement':
        oper = jax.jit(lambda f,index_f,k1,k2: mp.complement(f))
    elif type == 'supgen':
        oper = jax.jit(lambda f,index_f,k1,k2: mp.supgen(f,index_f,k1,k2))
    elif type == 'infgen':
        oper = jax.jit(lambda f,index_f,k1,k2: mp.infgen(f,index_f,k1,k2))
    elif type == 'sup':
        oper = jax.jit(mp.sup)
    elif type == 'inf':
        oper = jax.jit(mp.inf)
    else:
        print('Type of layer ' + type + 'is wrong!')
        return 1

    #Apply sup or inf
    if type == 'inf' or type == 'sup':
        fx = oper(x[:,0,:,:]).reshape((1,x.shape[2],x.shape[3]))
        for j in range(x.shape[1] - 1):
            fx = jnp.append(fx,oper(x[:,j + 1,:,:]).reshape((1,x.shape[2],x.shape[3])),axis = 0)
        fx = fx.reshape((1,fx.shape[0],fx.shape[1],fx.shape[2]))
        p = p + 1
    else:
        x = x[0,:,:,:]
        #Apply each operator
        fx = None
        for i in range(width):
            k1 = None
            k2 = None
            #Calculate kernel
            if type != 'complement':
                k1 = mp.struct_function_w(lambda w: params[p]['forward'](params[p]['params'],w),w,d)
                p = p + 1
                if type == 'supgen' or type == 'infgen':
                    k2 = mp.struct_function_w(lambda w: params[p]['forward'](params[p]['params'],w),w,d)
                    p = p + 1
            tmp = oper(x[0,:,:],index_x,k1,k2).reshape((1,x.shape[1],x.shape[2]))
            for j in range(x.shape[0] - 1):
                tmp = jnp.append(tmp,oper(x[j + 1,:,:],index_x,k1,k2).reshape((1,x.shape[1],x.shape[2])),axis = 0)
            tmp = tmp.reshape((1,tmp.shape[0],tmp.shape[1],tmp.shape[2]))
            if fx is None:
                fx = tmp
            else:
                fx = jnp.append(fx,tmp,axis = 0)
    return {'x': fx,'p': p}

#Canonical Morphological NN
@jax.jit
def cmnn(type,width,width_str,size,shape_x,activation = jax.nn.tanh,key = 0):
    #Index window
    index_x = mp.index_array(shape_x)

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
        w[str(d)] = jnp.array([[x1.tolist(),x2.tolist()] for x1 in jnp.linspace(-jnp.floor(d/2),jnp.floor(d/2),d) for x2 in jnp.linspace(jnp.floor(d/2),-jnp.floor(d/2),d)])

    #Forward pass
    def forward(x,params):
        p = 0
        x = x.reshape((1,x.shape[0],x.shape[1],x.shape[2]))
        for i in range(len(type)):
            #Apply layer
            x = apply_morph_layer(x,type[i],width[i],params,p,w[str(size[i])],index_x,size[i])
            #Update counter
            p = x['p']
            x = x['x']
        return x[0,:,:,:]

    #Return initial parameters and forward function
    return {'params': params,'forward': forward}
