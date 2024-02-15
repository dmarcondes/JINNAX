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
def generate_dDimdataPINN(u,xlo,xhi,tlo,thi,Nx,Nt,Nc,d = 1,posx = 'grid',post = 'grid',posc = 'grid',sigmaB = 0,sigmaI = 0,sigmaS = 0):
    #Repeat x limits
    xlo = [xlo for i in range(d)]
    xhi = [xhi for i in range(d)]

    #Sample sensor data
    if posx == 'grid':
        x_sensor = [[x.tolist()] for x in jnp.linspace(xlo[0],xhi[0],Nx)[1:-1]]
        for i in range(d-1):
            x_sensor =  [x1 + [x2.tolist()] for x1 in x_sensor for x2 in jnp.linspace(xlo[i+1],xhi[i+1],Nx)[1:-1]]
    elif posx == 'uniform':
        x_sensor = jax.random.uniform(key = jax.random.PRNGKey(random.randint(0,sys.maxsize)),minval = xlo[0],maxval = xhi[0],shape = ((Nx - 2) ** d,1))
        for i in range(d-1):
            x_sensor =  jnp.append(x_sensor,jax.random.uniform(key = jax.random.PRNGKey(random.randint(0,sys.maxsize)),minval = xlo[i+1],maxval = xhi[i+1],shape = ((Nx - 2) ** d,1)),1)
    x_sensor = jnp.array(x_sensor,dtype = jnp.float32)
    if post == 'grid':
        t_sensor = jnp.linspace(tlo,thi,Nt)[1:]
    elif post == 'uniform':
        t_sensor = jax.random.uniform(key = jax.random.PRNGKey(random.randint(0,sys.maxsize)),minval = tlo,maxval = thi,shape = (Nt - 1,))
    xt_sensor = jnp.array([x.tolist() + [t.tolist()] for x in x_sensor for t in t_sensor],dtype = jnp.float32)
    u_sensor = jnp.array([[u(x,t) + sigmaS*jax.random.normal(key = jax.random.PRNGKey(random.randint(0,sys.maxsize)))] for x in x_sensor for t in t_sensor],dtype = jnp.float32)

    #Set collocation points (always in an interior grid)
    if posc == 'grid':
        x_collocation = [[x.tolist()] for x in jnp.linspace(xlo[0],xhi[0],Nc + 2)[1:-1]]
        for i in range(d-1):
            x_collocation =  [x1 + [x2.tolist()] for x1 in x_collocation for x2 in jnp.linspace(xlo[i+1],xhi[i+1],Nc + 2)[1:-1]]
        t_collocation = jnp.linspace(tlo,thi,Nc + 1)[1:]
        xt_collocation = jnp.array([x + [t.tolist()] for x in x_collocation for t in t_collocation],dtype = jnp.float32)
    else:
        x_collocation = jax.random.uniform(key = jax.random.PRNGKey(random.randint(0,sys.maxsize)),minval = xlo[0],maxval = xhi[0],shape = (Nc ** d,1))
        for i in range(d-1):
            x_collocation =  jnp.append(x_collocation,jax.random.uniform(key = jax.random.PRNGKey(random.randint(0,sys.maxsize)),minval = xlo[i+1],maxval = xhi[i+1],shape = (Nc ** d,1)),1)
        t_collocation = jax.random.uniform(key = jax.random.PRNGKey(random.randint(0,sys.maxsize)),minval = tlo,maxval = thi,shape = (Nc,))
        xt_collocation = jnp.array([x.tolist() + [t.tolist()] for x in x_collocation for t in t_collocation],dtype = jnp.float32)

    #Sample boundary data
    t_boundary = jnp.append(t_sensor,0)
    x_boundary = jnp.array([[xlo[0]] + x.tolist() for x in jnp.delete(x_sensor,0,1)],dtype = jnp.float32)
    x_boundary = jnp.append(x_boundary,jnp.array([[xhi[0]] + x.tolist() for x in jnp.delete(x_sensor,0,1)],dtype = jnp.float32),0)
    for i in range(d-1):
        x_boundary = jnp.append(x_boundary,jnp.append(jnp.append(x_sensor[:,0:(i+1)],jnp.repeat(xlo[i+1],x_sensor.shape[0]).reshape(x_sensor.shape[0],1),1),x_sensor[:,(i+2):],1),0)
        x_boundary = jnp.append(x_boundary,jnp.append(jnp.append(x_sensor[:,0:(i+1)],jnp.repeat(xhi[i+1],x_sensor.shape[0]).reshape(x_sensor.shape[0],1),1),x_sensor[:,(i+2):],1),0)
    x_boundary = jnp.unique(x_boundary,axis = 0)
    xt_boundary = jnp.array([x.tolist() + [t.tolist()] for x in x_boundary for t in t_boundary],dtype = jnp.float32)
    u_boundary = jnp.array([[u(x,t) + sigmaB*jax.random.normal(key = jax.random.PRNGKey(random.randint(0,sys.maxsize)))] for x in x_boundary for t in t_boundary],dtype = jnp.float32)
    print('c')
    #Sample initial data
    x_initial = x_sensor
    xt_initial = jnp.array([x.tolist() + [t] for x in x_initial for t in [0.0]],dtype = jnp.float32)
    u_initial = jnp.array([[u(x,t) + sigmaI*jax.random.normal(key = jax.random.PRNGKey(random.randint(0,sys.maxsize)))] for x in x_initial for t in jnp.array([0.0])],dtype = jnp.float32)

    #Create data structure
    dat = {'sensor': xt_sensor,'usensor': u_sensor,'boundary': xt_boundary,'uboundary': u_boundary,'initial': xt_initial,'uinitial': u_initial,'collocation': xt_collocation}

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
