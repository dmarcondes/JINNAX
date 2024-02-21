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

#Generate d-dimensional data for PINN training
def generate_PINNdata(u,xl,xu,tl,tu,Ns = None,Nts = None,Nb = None,Ntb = None,N0 = None,Nc = None,Ntc = None,train = True,d = 1,poss = 'grid',posts = 'grid',pos0 = 'grid',posb = 'grid',postb = 'grid',posc = 'grid',postc = 'grid',sigmas = 0,sigmab = 0,sigma0 = 0):
    """
    generate_PINNdata: Generate spatio-temporal data in a d-dimension cube for PINN simulation
    ----------

    Parameters
    ----------
    u : function

        The function u(x,t) solution of the PDE

    xl : float

        Lower bound of each x coordinate

    xu : float

        Upper bound of each x coordinate

    tl : float

        Lower bound of the time interval.

    tu : float

        Upper bound of the time interval

    Ns : int, None

        Number of points along each x coordinate for sensor data. None for not generating sensor data

    Nts : int, None

        Number of points along the time axis for sensor data. None for not generating sensor data

    Nb : int, None

        Number of points along each x coordinate for boundary data. None for not generating boundary data

    Ntb : int, None

        Number of points along the time axis for boundary data. None for not generating boundary data

    N0 : int, None

        Number of points along each x coordinate for initial data. None for not generating initial data

    Nc : int, None

        Number of points along each x coordinate for collocation points. None for not generating collocation points

    Ntc : int, None

        Number of points along the time axis for collocation points. None for not generating collocation points

    train : logical

        Wheter to generate train (True) or test (False) data. Only sensor data is generated for test data. Default True

    d : int

        Domain dimension. Default 1

    poss : str

        Position of sensor data in spatial domain. Either 'grid' or 'random' for uniform sampling. Default 'grid'

    posts : str

        Position of sensor data in the time interval. Either 'grid' or 'random' for uniform sampling. Default 'grid'

    posb : int

        Position of boundary data in spatial domain. Either 'grid' or 'random' for uniform sampling. Default 'grid'

    postb : int

        Position of boundary data in the time interval. Either 'grid' or 'random' for uniform sampling. Default 'grid'

    pos0 : int

        Position of initial data in spatial domain. Either 'grid' or 'random' for uniform sampling. Default 'grid'

    posc : str

        Position of the collocation points in the x domain. Either 'grid' or 'random' for uniform sampling. Default 'grid'

    postc : str

        Position of the collocation points in the time interval. Either 'grid' or 'random' for uniform sampling. Default 'grid'

    sigmas : str

        Standard deviation of the Gaussian noise of sensor data. Default 0

    sigmab : str

        Standard deviation of the Gaussian noise of boundary data. Default 0

    sigma0 : str

        Standard deviation of the Gaussian noise of initial data. Default 0

    Returns
    -------

    dict-like object with generated data

    """

    #Repeat x limits
    if isinstance(xl,int) or isinstance(xl,float):
        xl = [xl for i in range(d)]
    if isinstance(xu,int) or isinstance(xu,float):
        xu = [xu for i in range(d)]

    #Sensor data
    if Ns is not None and Nts is not None:
        if poss == 'grid':
            #Create the grid for the first coordinate
            x_sensor = [[x.tolist()] for x in jnp.linspace(xl[0],xu[0],Ns + 2)[1:-1]]
            for i in range(d-1):
                #Product with the grid of the i-th coordinate
                x_sensor =  [x1 + [x2.tolist()] for x1 in x_sensor for x2 in jnp.linspace(xl[i+1],xu[i+1],Ns + 2)[1:-1]]
        else:
            #Sample Ns^d points for the first coordinate
            x_sensor = jax.random.uniform(key = jax.random.PRNGKey(random.randint(0,sys.maxsize)),minval = xl[0],maxval = xu[0],shape = (Ns ** d,1))
            for i in range(d-1):
                #Sample Ns^d points for the i-th coordinate and append collumn-wise
                x_sensor =  jnp.append(x_sensor,jax.random.uniform(key = jax.random.PRNGKey(random.randint(0,sys.maxsize)),minval = xl[i+1],maxval = xu[i+1],shape = (Ns ** d,1)),1)
        x_sensor = jnp.array(x_sensor,dtype = jnp.float32)

        if posts == 'grid':
            #Create the Nt grid of (tl,tu]
            t_sensor = jnp.linspace(tl,tu,Nts + 1)[1:]
        else:
            #Sample Nt points from (tl,tu)
            t_sensor = jax.random.uniform(key = jax.random.PRNGKey(random.randint(0,sys.maxsize)),minval = tl,maxval = tu,shape = (Nts,))
        #Product of x and t
        xt_sensor = jnp.array([x.tolist() + [t.tolist()] for x in x_sensor for t in t_sensor],dtype = jnp.float32)
        #Calculate u at each point
        u_sensor = jnp.array([u(x,t) + sigmas*jax.random.normal(key = jax.random.PRNGKey(random.randint(0,sys.maxsize))) for x in x_sensor for t in t_sensor],dtype = jnp.float32)
        u_sensor = u_sensor.reshape((u_sensor.shape[0],1))
    else:
        #Return None if sensor data should not be generated
        xt_sensor = None
        u_sensor = None

    #Set collocation points (always in an interior grid)
    if train and Ntc is not None and Nc is not None:
        if postc == 'grid':
            #Create the Ntc grid of (tl,tu]
            t_collocation = jnp.linspace(tl,tu,Ntc + 1)[1:]
        else:
            #Sample Ntc points from (tl,tu)
            t_collocation = jax.random.uniform(key = jax.random.PRNGKey(random.randint(0,sys.maxsize)),minval = tl,maxval = tu,shape = (Ntc,))

        if posc == 'grid':
            #Create the grid for the first coordinate
            x_collocation = [[x.tolist()] for x in jnp.linspace(xl[0],xu[0],Nc + 2)[1:-1]]
            for i in range(d-1):
                #Product with the grid of the i-th coordinate
                x_collocation =  [x1 + [x2.tolist()] for x1 in x_collocation for x2 in jnp.linspace(xl[i+1],xu[i+1],Nc + 2)[1:-1]]
            #Product of x and t
            xt_collocation = jnp.array([x + [t.tolist()] for x in x_collocation for t in t_collocation],dtype = jnp.float32)
        else:
            #Sample Nc^d points for the first coordinate
            x_collocation = jax.random.uniform(key = jax.random.PRNGKey(random.randint(0,sys.maxsize)),minval = xl[0],maxval = xu[0],shape = (Nc ** d,1))
            for i in range(d-1):
                #Sample Nc^d points for the i-th coordinate and append collumn-wise
                x_collocation =  jnp.append(x_collocation,jax.random.uniform(key = jax.random.PRNGKey(random.randint(0,sys.maxsize)),minval = xl[i+1],maxval = xu[i+1],shape = (Nc ** d,1)),1)
            #Product of x and t
            xt_collocation = jnp.array([x.tolist() + [t.tolist()] for x in x_collocation for t in t_collocation],dtype = jnp.float32)
    else:
        #Return None if collocation data should not be generated
        xt_collocation = None

    #Boundary data
    if train and Ntb is not None and Nb is not None:
        if postb == 'grid':
            #Create the Ntb grid of (tl,tu]
            t_boundary = jnp.linspace(tl,tu,Ntb + 1)[1:]
        else:
            #Sample Ntb points from (tl,tu)
            t_boundary = jax.random.uniform(key = jax.random.PRNGKey(random.randint(0,sys.maxsize)),minval = tl,maxval = tu,shape = (Ntb,))

        #An array in which each line represents an edge of the n-cube
        pre_grid = [[xl[0]],[xu[0]],[jnp.inf]]
        for i in range(d - 1):
            pre_grid = [x1 + [x2] for x1 in pre_grid for x2 in [xl[i + 1],xu[i + 1],jnp.inf]]
        #Exclude last row
        pre_grid = pre_grid[:-1]
        #Create array with vertex (xl,...,xl)
        x_boundary = jnp.array(pre_grid[0],dtype = jnp.float32).reshape((1,d))
        if posb == 'grid':
            #Create a grid over each edge of the n-cube
            for i in range(len(pre_grid) - 1):
                if jnp.inf in pre_grid[i + 1]:
                    #Create a list of the grid values along each coordinate in the edge i + 1
                    grid_points = list()
                    for j in range(len(pre_grid[i + 1])):
                        #If the coordinate is free, create grid
                        if pre_grid[i + 1][j] == jnp.inf:
                            grid_points.append(jnp.linspace(xl[j],xu[j],Nb + 2)[1:-1].tolist())
                        else:
                            #If the coordinate is fixed, store its value
                            grid_points.append([pre_grid[i + 1][j]])
                    #Product of these values
                    grid_values = [[x] for x in grid_points[0]]
                    for j in range(len(grid_points) - 1):
                        grid_values = [x1 + [x2] for x1 in grid_values for x2 in grid_points[j + 1]]
                    #Append to data
                    x_boundary = jnp.append(x_boundary,jnp.array(grid_values,dtype = jnp.float32).reshape((len(grid_values),d)),0)
                else:
                    #If the point is a vertex, append it to data
                    x_boundary = jnp.append(x_boundary,jnp.array(pre_grid[i + 1],dtype = jnp.float32).reshape((1,d)),0)
        else:
            #Sample points over each edge of the n-cube
            for i in range(len(pre_grid) - 1):
                if jnp.inf in pre_grid[i + 1]:
                    #Product of the fixed and sampled values
                    if jnp.inf == pre_grid[i + 1][0]:
                        grid_values = [[x] for x in jax.random.uniform(key = jax.random.PRNGKey(random.randint(0,sys.maxsize)),minval = xl[0],maxval = xu[0],shape = (Nb,)).tolist()]
                    else:
                        grid_values = [[pre_grid[i + 1][0]]]
                    for j in range(len(pre_grid[i + 1]) - 1):
                        if jnp.inf == pre_grid[i + 1][j + 1]:
                            grid_values = [x1 + [x2] for x1 in grid_values for x2 in jax.random.uniform(key = jax.random.PRNGKey(random.randint(0,sys.maxsize)),minval = xl[j + 1],maxval = xu[j + 1],shape = (Nb,)).tolist()]
                        else:
                            grid_values = [x1 + [pre_grid[i + 1][j + 1]] for x1 in grid_values]
                    #Append to data
                    x_boundary = jnp.append(x_boundary,jnp.array(grid_values,dtype = jnp.float32).reshape((len(grid_values),d)),0)
                else:
                    #If the point is a vertex, append it to data
                    x_boundary = jnp.append(x_boundary,jnp.array(pre_grid[i + 1],dtype = jnp.float32).reshape((1,d)),0)
        #Product of x and t
        xt_boundary = jnp.array([x.tolist() + [t.tolist()] for x in x_boundary for t in t_boundary],dtype = jnp.float32)
        #Calculate u at each point
        u_boundary = jnp.array([[u(x,t) + sigmab*jax.random.normal(key = jax.random.PRNGKey(random.randint(0,sys.maxsize)))] for x in x_boundary for t in t_boundary],dtype = jnp.float32)
        u_boundary = u_boundary.reshape((u_boundary.shape[0],1))
    else:
        #Return None if boundary data should not be generated
        xt_boundary = None
        u_boundary = None

    #Initial data
    if train and N0 is not None:
        if pos0 == 'grid':
            #Create the grid for the first coordinate
            x_initial = [[x.tolist()] for x in jnp.linspace(xl[0],xu[0],N0)]
            for i in range(d-1):
                #Product with the grid of the i-th coordinate
                x_initial =  [x1 + [x2.tolist()] for x1 in x_initial for x2 in jnp.linspace(xl[i+1],xu[i+1],N0)]
        else:
            #Sample N0^d points for the first coordinate
            x_initial = jax.random.uniform(key = jax.random.PRNGKey(random.randint(0,sys.maxsize)),minval = xl[0],maxval = xu[0],shape = (N0 ** d,1))
            for i in range(d-1):
                #Sample N0^d points for the i-th coordinate and append collumn-wise
                x_initial =  jnp.append(x_initial,jax.random.uniform(key = jax.random.PRNGKey(random.randint(0,sys.maxsize)),minval = xl[i+1],maxval = xu[i+1],shape = (N0 ** d,1)),1)
        x_initial = jnp.array(x_initial,dtype = jnp.float32)

        #Product of x and t
        xt_initial = jnp.array([x.tolist() + [t] for x in x_initial for t in [0.0]],dtype = jnp.float32)
        #Calculate u at each point
        u_initial = jnp.array([[u(x,t) + sigma0*jax.random.normal(key = jax.random.PRNGKey(random.randint(0,sys.maxsize)))] for x in x_initial for t in jnp.array([0.0])],dtype = jnp.float32)
        u_initial = u_initial.reshape((u_initial.shape[0],1))
    else:
        #Return None if initial data should not be generated
        xt_initial = None
        u_initial = None

    #Create data structure
    if train:
        dat = {'sensor': xt_sensor,'usensor': u_sensor,'boundary': xt_boundary,'uboundary': u_boundary,'initial': xt_initial,'uinitial': u_initial,'collocation': xt_collocation}
    else:
        dat = {'xt': xt_sensor,'u': u_sensor}

    return dat

#Read and organize a data.frame
def read_data_frame(file,sep = None,header = 'infer',sheet = 0):
    """
    Read a data file and convert to JAX array.
    -------

    Parameters
    ----------
    file : str

        File name with extension .csv, .txt, .xls or .xlsx

    sep : str

        Separation character for .csv and .txt files. Default ',' for .csv and ' ' for .txt

    header : int, Sequence of int, ‘infer’ or None

        See pandas.read_csv documentation. Default 'infer'

    sheet : int

        Sheet number for .xls and .xlsx files. Default 0

    Returns
    -------

    a JAX numpy array

    """

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
