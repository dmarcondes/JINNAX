#Functions to define NN architectures
import jax
import jax.numpy as jnp
import jinnax.morph as mp
import jinnax.training as jtr
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
    def forward(x,params):
      *hidden,output = params
      for layer in hidden:
        x = activation(x @ layer['W'] + layer['B'])
      return jax.nn.sigmoid(x @ output['W'] + output['B'])

    #Return initial parameters and forward function
    return {'params': params,'forward': forward}

#Apply a morphological layer
def apply_morph_layer(x,type,params,index_x):
    #Apply each operator
    params = jnp.minimum(jax.nn.relu(params),1.0)
    oper = mp.operator(type)
    fx = oper(x,index_x,params[0,:,:,:]).reshape((1,x.shape[0],x.shape[1],x.shape[2]))
    for i in range(1,params.shape[0]):
        fx = jnp.append(fx,oper(x,index_x,params[i,:,:,:]).reshape((1,x.shape[0],x.shape[1],x.shape[2])),0)
    return fx

#Apply a morphological layer in iterated NN
def apply_morph_layer_iter(x,type,params,index_x,w,forward_inner,d):
    #Compute structural elements
    k = None
    if type == 'supgen' or type == 'infgen':
        for i in range(int(len(params)/2)):
            tmp = forward_inner(w,params[2*i]).reshape((1,d,d))
            tmp = jnp.append(tmp,forward_inner(w,params[2*i + 1]).reshape((1,d,d)),0).reshape((1,2,d,d))
            if k is None:
                k = tmp
            else:
                k = jnp.append(k,tmp,0)
    else:
        for i in range(len(params)):
            tmp = forward_inner(w,params[i]).reshape((1,1,d,d))
            if k is None:
                k = tmp
            else:
                k = jnp.append(k,tmp,0)
    params = k

    #Apply each operator
    oper = mp.operator(type)
    fx = oper(x,index_x,params[0,:,:,:]).reshape((1,x.shape[0],x.shape[1],x.shape[2]))
    for i in range(1,params.shape[0]):
        fx = jnp.append(fx,oper(x,index_x,params[i,:,:,:]).reshape((1,x.shape[0],x.shape[1],x.shape[2])),0)
    return fx

#Canonical Morphological NN
def cmnn(type,width,size,shape_x,key = 0):
    #Index window
    index_x = mp.index_array(shape_x)

    #Initialize parameters
    initializer = jax.nn.initializers.normal()
    k = jax.random.split(jax.random.PRNGKey(key),(max(width))) #Seed for initialization
    params = list()
    for i in range(len(width)):
        if type[i] == 'supgen' or type[i] == 'infgen':
            params.append(jnp.abs(initializer(k[i,:],(width[i],2,size[i],size[i]),jnp.float32)))
        else:
            params.append(jnp.abs(initializer(k[i,:],(width[i],1,size[i],size[i]),jnp.float32)))

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

#Canonical Morphological NN with iterated NN
def cmnn_iter(type,width,width_str,size,shape_x,activation = jax.nn.tanh,key = 0,init = 'identity'):
    #Index window
    index_x = mp.index_array(shape_x)

    #Create w to apply str NN
    unique_size = set(size)
    w = {}
    for d in unique_size:
        w[str(d)] = jnp.array([[x1.tolist(),x2.tolist()] for x1 in jnp.linspace(-jnp.floor(d/2),jnp.floor(d/2),d) for x2 in jnp.linspace(jnp.floor(d/2),-jnp.floor(d/2),d)])

    #Initialize parameters
    if init == 'identity':
        #Train inner NN to generate zero and one kernel
        max_size = max(size)
        w_max = w[str(max_size)]
        nn = fconNN_str(width_str,activation,key)
        forward_inner = nn['forward']
        params0 = train_fcnn(w_max,jnp.zeros((w_max.shape[0],1)),forward_inner,nn['params'],jtr.MSE,epochs = 10000)
        params1 = train_fcnn(w_max,jnp.zeros((w_max.shape[0],1)) + 1.0,forward_inner,nn['params'],jtr.MSE,epochs = 10000)

        #Assign trained parameters
        params = list()
        for i in range(len(width)):
            params.append(list())
            for j in range(width[i]):
                if type[i] ==  'sup' or type[i] ==  'inf' or type[i] ==  'complement':
                    params[i].append(jnp.array(0.0,dtype = jnp.float32))
                else:
                    params[i].append(params0)
                    if type[i] == 'supgen' or type[i] == 'infgen':
                        params[i].append(params1)
    elif init == 'random':
        initializer = jax.nn.initializers.normal()
        k = jax.random.split(jax.random.PRNGKey(key),(len(width)*max(width))) #Seed for initialization
        c = 0
        forward_inner = fconNN_str(width_str,activation = jax.nn.tanh,key = 0)['forward']
        params = list()
        for i in range(len(width)):
            params.append(list())
            for j in range(width[i]):
                if type[i] ==  'sup' or type[i] ==  'inf' or type[i] ==  'complement':
                    params[i].append(jnp.array(0.0,dtype = jnp.float32))
                else:
                    tmp = fconNN_str(width_str,activation = jax.nn.tanh,key = k[c,0])
                    params[i].append(tmp['params'])
                    if type[i] == 'supgen' or type[i] == 'infgen':
                        tmp2 = fconNN_str(width_str,activation = jax.nn.tanh,key = k[c,1])
                        params[i].append(tmp2['params'])
                    c = c + 1

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
                x = apply_morph_layer_iter(x[0,:,:,:],type[i],params[i],index_x,w[str(size[i])],forward_inner,size[i])
        return x[0,:,:,:]

    #Return initial parameters and forward function
    return {'params': params,'forward': forward}
