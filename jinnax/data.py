#Functions to process data for training
import pandas
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import random
import sys
from PIL import Image
from IPython.display import display
__docformat__ = "numpy"

#Read and organize a data.frame
def read_data_frame(file,sep = None,header = None,sheet = 0):
    #Find out data extension
    ext = file.split('.')[1]

    #Read data frame
    if ext == 'csv':
        if sep is None:
            sep = ','
        dat = pandas.read_csv(file,sep = sep,header = header)
    elif ext == 'txt':
        if sep is None:
            sep = ' '
        dat = pandas.read_table(file,sep = sep,header = header)
    elif ext == 'xls' or ext == 'xlsx':
        dat = pandas.read_excel(file,header = header,sheet_name = sheet)

    #Convert to JAX data structure
    dat = jnp.array(dat,dtype = jnp.float32)

    return dat

#Generate d-dimensional data for PINN training
def generate_PINNdata(u,xlo,xup,tup,Ns,Nt,Nb = None,Nc = None,Ntc = None,train = True,tlo = 0,d = 1,poss = 'grid',post = 'grid',posc = 'grid',posct = 'grid',sigmas = 0,sigmab = 0,sigmai = 0):
    """Generate spatio-temporal data in a d-dimension cube for PINN simulation.

    Parameters
    ----------
    u : function

        Solution of the PDE

    xlo : float

        Lower bound of each x coordinate

    xup : float

        Upper bound of each x coordinate

    tlo : float

        Lower bound of the time interval. Default 0

    tup :

        Upper bound of the time interval

    Ns : int

        Number of points along each x coordinate for sensor data

    Nt : int

        Number of points along the time axis for sensor data

    Nb : int

        Number of points along each x coordinate for boundary data

    Nc : int

        Number of points along each x coordinate for collocation points

    Ntc : int

        Number of points along the time axis for collocation points

    train : logical

        Wheter to generate train (True) or test (False) data. Default True

    d : int

        Domain dimension. Default 1

    poss : str

        Position of sensor data in spatial domain. Either 'grid' or 'random' for uniform sampling. Default 'grid'

    post : str

        Position of points in the time interval. Either 'grid' or 'random' for uniform sampling. Default 'grid'

    posc : str

        Position of the collocation points in the x domain. Either 'grid' or 'random' for uniform sampling. Default 'grid'

    posct : str

        Position of the collocation points in the time interval. Either 'grid' or 'random' for uniform sampling. Default 'grid'

    sigmas : str

        Standard deviation of the Gaussian noise of sensor data (x inside the domain). Default 0

    sigmab : str

        Standard deviation of the Gaussian noise of boundary data. Default 0

    sigmai : str

        Standard deviation of the Gaussian noise of initial data. Default 0

    Returns
    -------

    dict-like object with generated data

    """

    #Repeat x limits
    xlo = [xlo for i in range(d)]
    xup = [xup for i in range(d)]

    #Sample sensor data
    if poss == 'grid':
        x_sensor = [[x.tolist()] for x in jnp.linspace(xlo[0],xup[0],Ns + 2)[1:-1]]
        for i in range(d-1):
            x_sensor =  [x1 + [x2.tolist()] for x1 in x_sensor for x2 in jnp.linspace(xlo[i+1],xup[i+1],Ns + 2)[1:-1]]
    else:
        x_sensor = jax.random.uniform(key = jax.random.PRNGKey(random.randint(0,sys.maxsize)),minval = xlo[0],maxval = xup[0],shape = (Ns ** d,1))
        for i in range(d-1):
            x_sensor =  jnp.append(x_sensor,jax.random.uniform(key = jax.random.PRNGKey(random.randint(0,sys.maxsize)),minval = xlo[i+1],maxval = xup[i+1],shape = (Ns ** d,1)),1)
    x_sensor = jnp.array(x_sensor,dtype = jnp.float32)

    if post == 'grid':
        t_sensor = jnp.linspace(tlo,tup,Nt + 1)[1:]
    else:
        t_sensor = jax.random.uniform(key = jax.random.PRNGKey(random.randint(0,sys.maxsize)),minval = tlo,maxval = tup,shape = (Nt,))
    xt_sensor = jnp.array([x.tolist() + [t.tolist()] for x in x_sensor for t in t_sensor],dtype = jnp.float32)
    u_sensor = jnp.array([u(x,t) + sigmas*jax.random.normal(key = jax.random.PRNGKey(random.randint(0,sys.maxsize))) for x in x_sensor for t in t_sensor],dtype = jnp.float32)
    u_sensor = u_sensor.reshape((u_sensor.shape[0],1))

    #Set collocation points (always in an interior grid)
    if train:
        if posct == 'grid':
            t_collocation = jnp.linspace(tlo,tup,Ntc)
        else:
            t_collocation = jax.random.uniform(key = jax.random.PRNGKey(random.randint(0,sys.maxsize)),minval = tlo,maxval = tup,shape = (Ntc,))

        if posc == 'grid':
            x_collocation = [[x.tolist()] for x in jnp.linspace(xlo[0],xup[0],Nc + 2)[1:-1]]
            for i in range(d-1):
                x_collocation =  [x1 + [x2.tolist()] for x1 in x_collocation for x2 in jnp.linspace(xlo[i+1],xup[i+1],Nc + 2)[1:-1]]
            xt_collocation = jnp.array([x + [t.tolist()] for x in x_collocation for t in t_collocation],dtype = jnp.float32)
        else:
            x_collocation = jax.random.uniform(key = jax.random.PRNGKey(random.randint(0,sys.maxsize)),minval = xlo[0],maxval = xup[0],shape = (Nc ** d,1))
            for i in range(d-1):
                x_collocation =  jnp.append(x_collocation,jax.random.uniform(key = jax.random.PRNGKey(random.randint(0,sys.maxsize)),minval = xlo[i+1],maxval = xup[i+1],shape = (Nc ** d,1)),1)
            xt_collocation = jnp.array([x.tolist() + [t.tolist()] for x in x_collocation for t in t_collocation],dtype = jnp.float32)

    #Sample boundary data
    if train:
        t_boundary = jnp.append(t_sensor,0)
        x_boundary = [[x.tolist()] for x in jnp.linspace(xlo[0],xup[0],Nb)]
        for i in range(d-1):
            x_boundary =  [x1 + [x2.tolist()] for x1 in x_boundary for x2 in jnp.linspace(xlo[i+1],xup[i+1],Nb)]
        x_boundary = jnp.array(x_boundary,dtype = jnp.float32)
        new_xb = x_boundary[0,:].reshape((1,d))
        x_max = jnp.max(x_boundary)
        x_min = jnp.min(x_boundary)
        for i in range(x_boundary.shape[0]):
            if x_min in x_boundary[i,:].tolist() or x_max in x_boundary[i,:].tolist():
                new_xb = jnp.append(new_xb,x_boundary[i,:].reshape((1,d)),0)
        x_boundary = jnp.unique(new_xb,axis = 0)
        xt_boundary = jnp.array([x.tolist() + [t.tolist()] for x in x_boundary for t in t_boundary],dtype = jnp.float32)
        u_boundary = jnp.array([[u(x,t) + sigmab*jax.random.normal(key = jax.random.PRNGKey(random.randint(0,sys.maxsize)))] for x in x_boundary for t in t_boundary],dtype = jnp.float32)
        u_boundary = u_boundary.reshape((u_boundary.shape[0],1))

    #Sample initial data
    if train:
        x_initial = x_sensor
        xt_initial = jnp.array([x.tolist() + [t] for x in x_initial for t in [0.0]],dtype = jnp.float32)
        u_initial = jnp.array([[u(x,t) + sigmai*jax.random.normal(key = jax.random.PRNGKey(random.randint(0,sys.maxsize)))] for x in x_initial for t in jnp.array([0.0])],dtype = jnp.float32)
        u_initial = u_initial.reshape((u_initial.shape[0],1))

    #Create data structure
    if train:
        dat = {'sensor': xt_sensor,'usensor': u_sensor,'boundary': xt_boundary,'uboundary': u_boundary,'initial': xt_initial,'uinitial': u_initial,'collocation': xt_collocation}
    else:
        dat = {'xt': xt_sensor,'u': u_sensor}

    return dat

#Read images into an array
def image_to_jnp(files_path):
    dat = None
    for f in files_path:
        img = Image.open(f)
        img = jnp.array(img,dtype = jnp.float32)/255
        if len(img.shape) == 3:
            img = img.reshape((1,img.shape[0],img.shape[1],img.shape[2]))
        else:
            img = img.reshape((1,img.shape[0],img.shape[1]))
        if dat is None:
            dat = img
        else:
            dat = jnp.append(dat,img,0)
    return dat

#Save images
def save_images(images,files_path):
    if len(files_path) > 1:
        for i in range(len(files_path)):
            if len(images.shape) == 4:
                tmp = Image.fromarray(np.uint8(jnp.round(255*images[i,:,:,:]))).convert('RGB')
            else:
                tmp = Image.fromarray(np.uint8(jnp.round(255*images[i,:,:]))).convert('RGB')
            tmp.save(files_path[i])
    else:
        if len(images.shape) == 4:
            tmp = Image.fromarray(np.uint8(jnp.round(255*images[0,:,:,:]))).convert('RGB')
        else:
            tmp = Image.fromarray(np.uint8(jnp.round(255*images[0,:,:]))).convert('RGB')
        tmp.save(files_path[0])

#Print images
def print_images(images):
    for i in range(images.shape[0]):
        if len(images.shape) == 4:
            tmp = Image.fromarray(np.uint8(jnp.round(255*images[i,:,:,:]))).convert('RGB')
        else:
            tmp = Image.fromarray(np.uint8(jnp.round(255*images[i,:,:]))).convert('RGB')
        display(tmp)
