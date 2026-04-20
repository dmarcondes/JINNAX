#Functions to train NN
import jax
import jax.numpy as jnp
import optax
from alive_progress import alive_bar
import math
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dill as pickle
from genree import bolstering as gb
from genree import kernel as gk
from jax import random
from jinnax import data as jd
import os
import numpy as np
from scipy.fft import dst, idst
from itertools import product
from functools import partial
import orthax
from jax import lax
from jaxopt import LBFGS
from jax.tree_util import tree_flatten

__docformat__ = "numpy"

def assert_tree_float64(tree, name="object"):
    """
    Raises an error if any leaf in a pytree is not float64.
    """
    leaves, _ = tree_flatten(tree)
    bad = [
        (i, x.dtype)
        for i, x in enumerate(leaves)
        if hasattr(x, "dtype") and x.dtype != jnp.float64
    ]
    if bad:
        raise RuntimeError(
            f"[FLOAT64 CHECK FAILED] {name} contains non-float64 dtypes:\n"
            + "\n".join([f"  leaf {i}: {dtype}" for i, dtype in bad])
        )


def warn_tree_float64(tree, name="object"):
    """
    Same as assert_tree_float64, but prints a warning instead of raising.
    """
    leaves, _ = tree_flatten(tree)
    dtypes = {x.dtype for x in leaves if hasattr(x, "dtype")}
    if dtypes != {jnp.float64}:
        print(f"[WARNING] {name} dtypes = {dtypes}")


def check_grads_float64(grads):
    leaves, _ = tree_flatten(grads)
    bad = [
        (i, x.dtype)
        for i, x in enumerate(leaves)
        if hasattr(x, "dtype") and x.dtype != jnp.float64
    ]
    if bad:
        raise RuntimeError(
            "[FLOAT64 CHECK FAILED] Gradients are not float64:\n"
            + "\n".join([f"  grad {i}: {dtype}" for i, dtype in bad])
        )


def assert_lbfgs_state_float64(state):
    """
    Ensures that all *floating-point* values inside the LBFGS state
    are float64. Integers and booleans are allowed.
    """
    leaves, _ = tree_flatten(state)
    bad = []
    for i, x in enumerate(leaves):
        if isinstance(x, jnp.ndarray):
            if jnp.issubdtype(x.dtype, jnp.floating):
                if x.dtype != jnp.float64:
                    bad.append((i, x.dtype))
    if bad:
        raise RuntimeError(
            "[FLOAT64 CHECK FAILED] LBFGS floating-point buffers are not float64:\n"
            + "\n".join([f"  leaf {i}: {dtype}" for i, dtype in bad])
        )


#Change to float64
def to_float64(tree):
    def cast(x):
        if isinstance(x, (jax.Array, np.ndarray, float, int)):
            return x.astype(jnp.float64)
        return x
    return jax.tree_util.tree_map(cast, tree)


#MSE
@jax.jit
def MSE(pred,true):
    """
    Squared error
    ----------
    Parameters
    ----------
    pred : jax.numpy.array

        A JAX numpy array with the predicted values

    true : jax.numpy.array

        A JAX numpy array with the true values

    Returns
    -------
    squared error
    """
    return (true - pred) ** 2

#MSE self-adaptative
@jax.jit
def MSE_SA(pred,true,w,q = 2):
    """
    Self-adaptative squared error
    ----------
    Parameters
    ----------
    pred : jax.numpy.array

        A JAX numpy array with the predicted values

    true : jax.numpy.array

        A JAX numpy array with the true values

    weight : jax.numpy.array

        A JAX numpy array with the weights

    q : float

        Power for the weights mask

    Returns
    -------
    self-adaptative squared error with polynomial mask
    """
    return (w ** q) * ((true - pred) ** 2)

#L2 error
@jax.jit
def L2error(pred,true):
    """
    L2-error in percentage (%)
    ----------
    Parameters
    ----------
    pred : jax.numpy.array

        A JAX numpy array with the predicted values

    true : jax.numpy.array

        A JAX numpy array with the true values

    Returns
    -------
    L2-error
    """
    return 100*jnp.sqrt(jnp.sum((true - pred)**2))/jnp.sqrt(jnp.sum(true ** 2))

#Auxialiry functions to sample singular Matern
def idst1(x,axis = -1):
    """
    Inverse Discrete Sine Transform of type I with orthonormal scaling
    ----------
    Parameters
    ----------
    x : jax.numpy.array

        Array to apply the transformation

    axis : int

        Axis to apply the transformation over

    Returns
    -------
    jax.numpy.array
    """
    return idst(x,type = 1,axis = axis,norm = 'ortho')

def dstn(x,axes = None):
    """
    Discrete Sine Transform of type I with orthonormal scaling over many axes
    ----------
    Parameters
    ----------
    x : jax.numpy.array

        Array to apply the transformation

    axes : int

        Axes to apply the transformation over

    Returns
    -------
    jax.numpy.array
    """
    if axes is None:
        axes = tuple(range(x.ndim))
    y = x
    for ax in axes:
        y = dst(y,type = 1,axis = ax,norm = 'ortho')
    return y

def idstn(x,axes = None):
    """
    Inverse Discrete Sine Transform of type I with orthonormal scaling over many axes
    ----------
    Parameters
    ----------
    x : jax.numpy.array

        Array to apply the transformation

    axes : tuple

        Axes to apply the transformation over

    Returns
    -------
    jax.numpy.array
    """
    if axes is None:
        axes = tuple(range(x.ndim))
    y = x
    for ax in axes:
        y = idst1(y,axis = ax)
    return y

def dirichlet_eigs_nd(n,L):
    """
    Eigenvalues of the discrete Dirichlet-Laplace operator in a rectangle
    ----------
    Parameters
    ----------
    n : list

        List with the number of points in the grid in each dimension

    L : list

        List with the upper limit of the interval of the domain in each dimension. Assumed the lower limit is zero

    Returns
    -------
    jax.numpy.array
    """
    #Unidimensional eigenvalues
    lam_axes = []
    for ni, Li in zip(n,L):
        h = Li / (ni + 1)
        k = jnp.arange(1,ni + 1)
        ln = (2 / (h*h)) * (1 - jnp.cos(jnp.pi * k / (ni + 1)))
        lam_axes.append(ln)
    grids = jnp.meshgrid(*lam_axes, indexing='ij')
    Lam = jnp.zeros_like(grids[0])
    for g in grids:
        Lam += g
    return Lam


#Sample from d-dimensional Matern process
def generate_matern_sample(key,d = 2,N = 128,L = 1.0,kappa = 1,alpha = 1,sigma = 1,periodic = False):
    """
    Sample d-dimensional Matern process
    ----------
    Parameters
    ----------
    key : int

        Seed for randomization

    d : int

        Dimension. Default 2

    N : int

        Size of grid in each dimension. Default 128

    L : list of float

        The domain of the function in each coordinate is [0,L[1]]. If a float, repeat the same interval for all coordinates. Default 1

    kappa,alpha,sigma : float

        Parameters of the Matern process

    periodic : logical

        Whether to sample with periodic boundary conditions. Periodic = False is not JAX native and does not work with JIT

    Returns
    -------
    jax.numpy.array
    """
    if periodic:
        #Shape and key
        key = jax.random.PRNGKey(key)
        shape = (N,) * d
        if isinstance(L,float) or isinstance(L,int):
            L = d*[L]
        if isinstance(N,float) or isinstance(N,int):
            N = d*[N]

        #Setup Frequency Grid (2D)
        freq = [jnp.fft.fftfreq(N[j],d = L[j]/N[j]) * 2 * jnp.pi for j in range(d)]
        grids = jnp.meshgrid(*freq, indexing='ij')
        sq_norm_xi = sum(g**2 for g in grids)

        #Generate White Noise in Fourier Space
        key_re, key_im = jax.random.split(key)
        white_noise_f = (jax.random.normal(key_re, shape) +
                         1j * jax.random.normal(key_im, shape))

        #Apply the Whittle Filter
        amplitude_filter = (kappa ** 2 + sq_norm_xi) ** (-alpha / 2)
        field_f = white_noise_f * amplitude_filter

        #Transform back to Physical Space
        sample = jnp.real(jnp.fft.ifftn(field_f))
        return sigma*sample
    else: #NOT JAX
        #Shape and key
        rng = np.random.default_rng(seed = key)
        if isinstance(L,float) or isinstance(L,int):
            L = d*[L]
        if isinstance(N,float) or isinstance(N,int):
            N = d*[N]
        shape = tuple(N)

        #White noise in real space
        W = rng.standard_normal(size = shape)

        #To Dirichlet eigenbasis via separable DST-I (orthonormal)
        W_hat = dstn(W)

        #Discrete Dirichlet Laplacian eigenvalues
        lam = dirichlet_eigs_nd(N, L)

        #Spectral filter
        filt = ((kappa + lam) ** (-alpha/2))
        psi_hat = filt * W_hat

        #Back to real space
        psi = idstn(psi_hat)
        return jnp.array(sigma*psi)

#Vectorized generate_matern_sample
def generate_matern_sample_batch(d = 2,N = 512,L = 1.0,kappa = 10.0,alpha = 1,sigma = 10,periodic = False):
    """
    Create function to sample d-dimensional Matern process
    ----------
    Parameters
    ----------
    d : int

        Dimension. Default 2

    N : int

        Size of grid in each dimension. Default 128

    L : list of float

        The domain of the function in each coordinate is [0,L[1]]. If a float, repeat the same interval for all coordinates. Default 1

    kappa,alpha,sigma : float

        Parameters of the Matern process

    periodic : logical

        Whether to sample with periodic boundary conditions. Periodic = False is not JAX native and does not work with JIT

    Returns
    -------
    function
    """
    if periodic:
        return jax.vmap(lambda k: generate_matern_sample(k,d = d,N = N,L = L,kappa = kappa,alpha = alpha,sigma = sigma,periodic = periodic))
    else:
        return lambda keys: jnp.array(np.apply_along_axis(lambda k: generate_matern_sample(k,d = d,N = N,L = L,kappa = kappa,alpha = alpha,sigma = sigma,periodic = periodic),1,keys.reshape((keys.shape[0],1))))

#Build function to compute the eigenfunctions of Laplacian
def eigenf_laplace(L_vec,kmax_per_axis = None,bc = "dirichlet",max_ef = None):
    """
    Create function to compute in batches the eigenfunctions of the Dirichlet-Laplace or Neumann-Laplace.
    ----------
    Parameters
    ----------
    L_vec : list of float

        The domain of the function in each coordinate is [0,L[1]]

    kmax_per_axis : list

        List with the maximum number of eigenfunctions per dimension. Consider d * max(kmax_per_axis) eigenfunctions

    bc : str

        Boundary condition. 'dirichlet' or 'neumann'

    max_ef : int

        Maximum number of eigenfunctions to consider among the ones with greatest eigenvalues. If None, considers d * max(kmax_per_axis) eigenfunctions

    Returns
    -------
    function to compute eigenfunctions,eigenvalues of the eigenfunctions considered
    """
    #Parameters
    L_vec = jnp.asarray(L_vec)
    d = L_vec.shape[0]
    bc = bc.lower()

    #Maximum number of functions
    if max_ef is None:
        if d == 1:
            max_ef = jnp.max(jnp.array(kmax_per_axis))
        else:
            max_ef = jnp.max(d * jnp.array(kmax_per_axis))

    #Build the candidate multi-indices per axis
    kmax_per_axis = list(map(int, kmax_per_axis))
    if bc.startswith("d"):
        axis_ranges = [range(1, km + 1) for km in kmax_per_axis]
    elif bc.startswith("n"):
        axis_ranges = [range(0, km + 1) for km in kmax_per_axis]

    #Get all multi-indices
    Ks_list = list(product(*axis_ranges))
    Ks = jnp.array(Ks_list)

    #Eigenvalues of the continuous Laplacian
    pi_over_L = jnp.pi / L_vec
    lambdas_all = jnp.sum((Ks * pi_over_L) ** 2, axis=1)

    #Sort by eigenvalue
    order = jnp.argsort(lambdas_all)
    Ks = Ks[order]
    lambdas_all = lambdas_all[order]

    #Keep first max_ef
    Ks = Ks[:max_ef]
    lambdas = lambdas_all[:max_ef]
    m = Ks.shape[0]

    #Precompute per-feature normalization factor (closed form)
    def per_axis_norm_factor(k_i, L_i, is_dirichlet):
        if is_dirichlet:
            return jnp.sqrt(2 / L_i)
        else:
            return jnp.where(k_i == 0, jnp.sqrt(1 / L_i), jnp.sqrt(2 / L_i))
    if bc.startswith("d"):
        nf = jnp.prod(jnp.sqrt(2 / L_vec)[None, :],axis = 1)
        norm_factors = jnp.ones((m,)) * nf
    else:
        # per-mode product across axes
        def nf_row(k_row):
            return jnp.prod(per_axis_norm_factor(k_row, L_vec, False))
        norm_factors = jax.vmap(nf_row)(Ks)

    #Build the callable function
    Ks_int = Ks  # float array, but only integer values
    L_vec_f = L_vec
    @jax.jit
    def phi(x):
        x = jnp.asarray(x)
        #Initialize with ones
        vals = jnp.ones(x.shape[:-1] + (m,))
        #Compute eigenfunction
        for i in range(d):
            ang = (jnp.pi / L_vec_f[i]) * x[..., i][..., None] * Ks_int[:, i]
            if bc.startswith("d"):
                comp = jnp.sin(ang)
            else:
                comp = jnp.cos(ang)
            vals = vals * comp
        #Apply L2-normalizing constants
        vals = vals * norm_factors[None, ...] if vals.ndim > 1 else vals * norm_factors
        return vals
    return phi, lambdas

#Compute multiple frequences of domain aware fourrier fesatures
def multiple_daff(L_vec,kmax_per_axis = None,bc = "dirichlet",max_ef = None):
    """
    Create function to compute multiple frequences of the eigenfunctions of the Dirichlet-Laplace or Neumann-Laplace. Each frequences is a different domain.
    ----------
    Parameters
    ----------
    L_vec : list of lists of float

        List with the domain of each frequence of the eigenfunctions in the form [0,L[i][1]]

    kmax_per_axis : list

        List with the maximum number of eigenfunctions per dimension.

    bc : str

        Boundary condition. 'dirichlet' or 'neumann'

    max_ef : int

        Maximum number of eigenfunctions to consider among the ones with greatest eigenvalues. If None, considers d * max(kmax_per_axis) eigenfunctions

    Returns
    -------
    function to compute daff,eigenvalues of the eigenfunctions considered
    """
    psi = []
    lamb = []
    for L in L_vec:
        tmp,l = eigenf_laplace(L,kmax_per_axis,bc,max_ef) #Get function
        lamb.append(l)
        psi.append(tmp)
        del tmp
    #Create function to compute features
    @jax.jit
    def mff(x):
        y = []
        for i in range(len(psi)):
            y.append(psi[i](x))
        if len(psi) == 1:
            return y[0]
        else:
            return jnp.concatenate(y,1)
    return mff,jnp.concatenate(lamb)

#Code for chebyshev polynomials writeen by AI (deprecated)
def _chebyshev_T_all(t, K: int):
    """
    Compute T_0..T_K(t) with the standard recurrence.
    t shape should be (..., d). We DO NOT squeeze any axis to preserve 'd'
    even when d == 1.
    Returns: array of shape (K+1, ...) matching t's batch dims, including d.
    """
    # Expect t to have last axis = d (keep it, even if d == 1)
    T0 = jnp.ones_like(t)           # (..., d)
    if K == 0:
        return T0[None, ...]        # (1, ..., d)

    T1 = t                          # (..., d)
    if K == 1:
        return jnp.stack([T0, T1], axis=0)  # (2, ..., d)

    def body(carry, _):
        Tkm1, Tk = carry            # each (..., d)
        Tkp1 = 2 * t * Tk - Tkm1  # (..., d)
        return (Tk, Tkp1), Tkp1

    # K >= 2: produce T_2..T_K
    (_, _), T2_to_TK = lax.scan(body, (T0, T1), jnp.arange(K - 1))  # (K-1, ..., d)
    return jnp.concatenate([T0[None, ...], T1[None, ...], T2_to_TK], axis=0)  # (K+1, ..., d)

@partial(jax.jit,static_argnums=(2,))  # n is static here; compile once per n
def multiple_cheb_fast(x, L_vec, n: int):
    """
    x: (N, d)
    L_vec: (L, d) containing 'b' endpoints (a is 0) for each dimension
    n: number of k terms (static)
    returns: (N, L*n)
    """
    N, d = x.shape
    L = L_vec.shape[0]

    a = 0
    b = L_vec                       # (L, d)
    # Map x to t in [-1, 1] for each l, j: shape (L, N, d)
    t = (2 * x[None, :, :] - (a + b)[:, None, :]) / (b - a)[:, None, :]

    # Chebyshev T_0..T_{n+2} for all (L, N, d): shape (n+3, L, N, d)
    T = _chebyshev_T_all(t, n + 2)

    # phi_k = T_{k+2} - T_k, k = 0..n-1  => shape (n, L, N, d)
    ks = jnp.arange(n)
    phi = T[ks + 2, ...] - T[ks, ...]

    # Multiply across dimensions (over the last axis = d) => (n, L, N)
    z = jnp.prod(phi, axis=-1)

    # Reorder to (N, L, n) then flatten to (N, L*n)
    z = jnp.transpose(z, (2, 1, 0)).reshape(N, L * n)
    return z

def multiple_cheb(L_vec, n: int):
    """
    Factory that closes over static n and L_vec (so shapes are constant).
    """
    L_vec = jnp.asarray(L_vec)
    @jax.jit  # optional; multiple_cheb_fast is already jitted
    def mcheb(x):
        x = jnp.asarray(x)
        return multiple_cheb_fast(x, L_vec, n)
    return mcheb


#Initialize fully connected neyral network Return the initial parameters and the function for the forward pass
def fconNN(width,activation = jax.nn.tanh,key = 0,mlp = False,ftype = None,fargs = None,static = None,daff = None):
    """
    Initialize fully connected neural network
    ----------
    Parameters
    ----------
    width : list

        List with the layers width

    activation : jax.nn activation

        The activation function. Default jax.nn.tanh

    key : int

        Seed for parameters initialization. Default 0

    mlp : logical

        Whether to consider a modified multilayer perceptron. Assumes all hidden layers have the same dimension.

    ftype : str

        Type of feature transformation to use: None, 'ff', 'daff','daff_bias', 'cheb', 'cheb_bias'.

    fargs : list

        Arguments for deature transformation:

        For 'ff': A list with the number of frequences and value of greatest frequence standard deviation.

        For 'daff' and 'daff' bias: A dicitionary with a list with the size of rectangles and the type of boundary condition. If its a list, than boundary conditions is dirichlet.

    static : function

        A static function to sum to the neural network output.

    daff : list

    List with function to compute daff and the number of daff. If None computes assuming rectangular domain.

    Returns
    -------
    dict with initial parameters and the function for the forward pass
    """
    #Initialize parameters with Glorot initialization
    initializer = jax.nn.initializers.glorot_normal()
    params = list()
    if static is None:
        static = lambda x: 0

    #Feature mapping
    if ftype == 'ff': #Fourrier features
        for s in range(fargs[0]):
            sd = fargs[1] ** ((s + 1)/fargs[0])
            if s == 0:
                Bff = sd*jax.random.normal(jax.random.PRNGKey(key + s + 1),(width[0],int(width[1]/2)))
            else:
                Bff = jnp.append(Bff,sd*jax.random.normal(jax.random.PRNGKey(key + s + 1),(width[0],int(width[1]/2))),1)
        @jax.jit
        def phi(x):
            x = x @ Bff
            return jnp.concatenate([jnp.sin(2 * jnp.pi * x),jnp.cos(2 * jnp.pi * x)],axis = -1)
        width = width[1:]
        width[0] = 2*Bff.shape[1]
    elif ftype == 'daff' or ftype == 'daff_bias':
        if not isinstance(fargs, dict):
            fargs = {'L': fargs,'bc': "dirichlet"}
        if daff is None:
            phi,lamb = multiple_daff(list(fargs.values())[0],kmax_per_axis = [width[1]] * width[0],bc = list(fargs.values())[1])
            width = width[1:]
            width[0] = lamb.shape[0]
        else:
            phi = daff[0]
            width = width[1:]
            width[0] = daff[1]
    elif ftype == 'cheb' or ftype == 'cheb_bias':
        phi = multiple_cheb(fargs,n = width[1])
        width = width[1:]
        width[0] = len(fargs)*width[0]
    else:
        @jax.jit
        def phi(x):
            return x

    #Initialize parameters
    if mlp:
        k = jax.random.split(jax.random.PRNGKey(key),4)
        WU = initializer(k[0],(width[0],width[1]))
        BU = initializer(k[1],(1,width[1]))
        WV = initializer(k[2],(width[0],width[1]))
        BV = initializer(k[3],(1,width[1]))
        params.append({'WU':WU,'BU':BU,'WV':WV,'BV':BV})
    key = jax.random.split(jax.random.PRNGKey(key + 1),len(width)-1) #Seed for initialization
    for key,lin,lout in zip(key,width[:-1],width[1:]):
        W = initializer(key,(lin,lout))
        B = initializer(key,(1,lout))
        params.append({'W':W,'B':B})

    #Define function for forward pass
    if mlp:
        if ftype != 'daff' and ftype != 'cheb':
            @jax.jit
            def forward(x,params):
                encode,*hidden,output = params
                sx = static(x)
                x = phi(x)
                U = activation(x @ encode['WU'] + encode['BU'])
                V = activation(x @ encode['WV'] + encode['BV'])
                for layer in hidden:
                    x = activation(x @ layer['W'] + layer['B'])
                    x = x * U + (1 - x) * V
                return x @ output['W'] + output['B'] + sx
        else:
            @jax.jit
            def forward(x,params):
                encode,*hidden,output = params
                sx = static(x)
                x = phi(x)
                U = activation(x @ encode['WU'])
                V = activation(x @ encode['WV'])
                for layer in hidden:
                    x = activation(x @ layer['W'])
                    x = x * U + (1 - x) * V
                return x @ output['W'] + sx
    else:
        if ftype != 'daff' and ftype != 'cheb':
            @jax.jit
            def forward(x,params):
                *hidden,output = params
                sx = static(x)
                x = phi(x)
                for layer in hidden:
                    x = activation(x @ layer['W'] + layer['B'])
                return x @ output['W'] + output['B'] + sx
        else:
            @jax.jit
            def forward(x,params):
                *hidden,output = params
                sx = static(x)
                x = phi(x)
                for layer in hidden:
                    x = activation(x @ layer['W'])
                return x @ output['W'] + sx

    #Return initial parameters and forward function
    return {'params': params,'forward': forward}

#Get activation from string
def get_activation(act):
    """
    Return activation function from string
    ----------
    Parameters
    ----------
    act : str

        Name of the activation function. Default 'tanh'

    Returns
    -------
    jax.nn activation function
    """
    if act == 'tanh':
        return jax.nn.tanh
    elif act == 'relu':
        return jax.nn.relu
    elif act == 'relu6':
        return jax.nn.relu6
    elif act == 'sigmoid':
        return jax.nn.sigmoid
    elif act == 'softplus':
        return jax.nn.softplus
    elif act == 'sparse_plus':
        return jax.nn.sparse_plus
    elif act == 'soft_sign':
        return jax.nn.soft_sign
    elif act == 'silu':
        return jax.nn.silu
    elif act == 'swish':
        return jax.nn.swish
    elif act == 'log_sigmoid':
        return jax.nn.log_sigmoid
    elif act == 'leaky_relu':
        return jax.nn.leaky_relu
    elif act == 'hard_sigmoid':
        return jax.nn.hard_sigmoid
    elif act == 'hard_silu':
        return jax.nn.hard_silu
    elif act == 'hard_swish':
        return jax.nn.hard_swish
    elif act == 'hard_tanh':
        return jax.nn.hard_tanh
    elif act == 'elu':
        return jax.nn.elu
    elif act == 'celu':
        return jax.nn.celu
    elif act == 'selu':
        return jax.nn.selu
    elif act == 'gelu':
        return jax.nn.gelu
    elif act == 'glu':
        return jax.nn.glu
    elif act == 'squareplus':
        return  jax.nn.squareplus
    elif act == 'mish':
        return jax.nn.mish

#Training PINN
def train_PINN(data,width,pde,test_data = None,epochs = 100,at_each = 10,activation = 'tanh',neumann = False,oper_neumann = False,sa = False,c = {'ws': 1,'wr': 1,'w0': 100,'wb': 1},inverse = False,initial_par = None,lr = 0.001,b1 = 0.9,b2 = 0.999,eps = 1e-08,eps_root = 0.0,key = 0,epoch_print = 100,save = False,file_name = 'result_pinn',exp_decay = False,transition_steps = 1000,decay_rate = 0.9,mlp = False):
    """
    Train a Physics-informed Neural Network
    ----------
    Parameters
    ----------
    data : dict

        Data generated by the jinnax.data.generate_PINNdata function

    width : list

        A list with the width of each layer

    pde : function

        The partial differential operator. Its arguments are u, x and t

    test_data : dict, None

        A dictionay with test data for L2 error calculation generated by the jinnax.data.generate_PINNdata function. Default None for not calculating L2 error

    epochs : int

        Number of training epochs. Default 100

    at_each : int

        Save results for epochs multiple of at_each. Default 10

    activation : str

        The name of the activation function of the neural network. Default 'tanh'

    neumann : logical

        Whether to consider Neumann boundary conditions

    oper_neumann : function

        Penalization of Neumann boundary conditions

    sa : logical

        Whether to consider self-adaptative PINN

    c : dict

        Dictionary with the hyperparameters of the self-adaptative sigmoid mask for the initial (w0), sensor (ws) and collocation (wr) points. The weights of the boundary points is fixed to 1

    inverse : logical

        Whether to estimate parameters of the PDE

    initial_par : jax.numpy.array

        Initial value of the parameters of the PDE in an inverse problem

    lr,b1,b2,eps,eps_root: float

        Hyperparameters of the Adam algorithm. Default lr = 0.001, b1 = 0.9, b2 = 0.999, eps = 1e-08, eps_root = 0.0

    key : int

        Seed for parameters initialization. Default 0

    epoch_print : int

        Number of epochs to calculate and print test errors. Default 100

    save : logical

        Whether to save the current parameters. Default False

    file_name : str

        File prefix to save the current parameters. Default 'result_pinn'

    exp_decay : logical

        Whether to consider exponential decay of learning rate. Default False

    transition_steps : int

        Number of steps for exponential decay. Default 1000

    decay_rate : float

        Rate of exponential decay. Default 0.9

    mlp : logical

        Whether to consider modifed multi-layer perceptron

    Returns
    -------
    dict-like object with the estimated function, the estimated parameters, the neural network function for the forward pass and the training time
    """

    #Initialize architecture
    nnet = fconNN(width,get_activation(activation),key,mlp)
    forward = nnet['forward']

    #Initialize self adaptative weights
    par_sa = {}
    if sa:
        #Initialize wheights close to zero
        ksa = jax.random.randint(jax.random.PRNGKey(key),(5,),1,1000000)
        if data['sensor'] is not None:
            par_sa.update({'ws': c['ws'] * jax.random.uniform(key = jax.random.PRNGKey(ksa[0]),shape = (data['sensor'].shape[0],1))})
        if data['initial'] is not None:
            par_sa.update({'w0': c['w0'] * jax.random.uniform(key = jax.random.PRNGKey(ksa[1]),shape = (data['initial'].shape[0],1))})
        if data['collocation'] is not None:
            par_sa.update({'wr': c['wr'] * jax.random.uniform(key = jax.random.PRNGKey(ksa[2]),shape = (data['collocation'].shape[0],1))})
        if data['boundary'] is not None:
            par_sa.update({'wb': c['wb'] * jax.random.uniform(key = jax.random.PRNGKey(ksa[3]),shape = (data['boundary'].shape[0],1))})

    #Store all parameters
    params = {'net': nnet['params'],'inverse': initial_par,'sa': par_sa}

    #Save config file
    if save:
        pickle.dump({'train_data': data,'epochs': epochs,'activation': activation,'init_params': params,'forward': forward,'width': width,'pde': pde,'lr': lr,'b1': b1,'b2': b2,'eps': eps,'eps_root': eps_root,'key': key,'inverse': inverse,'sa': sa},open(file_name + '_config.pickle','wb'), protocol = pickle.HIGHEST_PROTOCOL)

    #Define loss function
    if sa:
        #Define loss function
        @jax.jit
        def lf(params,x):
            loss = 0
            if x['sensor'] is not None:
                #Term that refers to sensor data
                loss = loss + jnp.mean(MSE_SA(forward(x['sensor'],params['net']),x['usensor'],params['sa']['ws']))
            if x['boundary'] is not None:
                if neumann:
                    #Neumann coditions
                    xb = x['boundary'][:,:-1].reshape((x['boundary'].shape[0],x['boundary'].shape[1] - 1))
                    tb = x['boundary'][:,-1].reshape((x['boundary'].shape[0],1))
                    loss = loss + jnp.mean(oper_neumann(lambda x,t: forward(jnp.append(x,t,1),params['net']),xb,tb,params['sa']['wb']))
                else:
                    #Term that refers to boundary data
                    loss = loss + jnp.mean(MSE_SA(forward(x['boundary'],params['net']),x['uboundary'],params['sa']['wb']))
            if x['initial'] is not None:
                #Term that refers to initial data
                loss = loss + jnp.mean(MSE_SA(forward(x['initial'],params['net']),x['uinitial'],params['sa']['w0']))
            if x['collocation'] is not None:
                #Term that refers to collocation points
                x_col = x['collocation'][:,:-1].reshape((x['collocation'].shape[0],x['collocation'].shape[1] - 1))
                t_col = x['collocation'][:,-1].reshape((x['collocation'].shape[0],1))
                if inverse:
                    loss = loss + jnp.mean(MSE_SA(pde(lambda x,t: forward(jnp.append(x,t,1),params['net']),x_col,t_col,params['inverse']),0,params['sa']['wr']))
                else:
                    loss = loss + jnp.mean(MSE_SA(pde(lambda x,t: forward(jnp.append(x,t,1),params['net']),x_col,t_col),0,params['sa']['wr']))
            return loss
    else:
        @jax.jit
        def lf(params,x):
            loss = 0
            if x['sensor'] is not None:
                #Term that refers to sensor data
                loss = loss + jnp.mean(MSE(forward(x['sensor'],params['net']),x['usensor']))
            if x['boundary'] is not None:
                if neumann:
                    #Neumann coditions
                    xb = x['boundary'][:,:-1].reshape((x['boundary'].shape[0],x['boundary'].shape[1] - 1))
                    tb = x['boundary'][:,-1].reshape((x['boundary'].shape[0],1))
                    loss = loss + jnp.mean(oper_neumann(lambda x,t: forward(jnp.append(x,t,1),params['net']),xb,tb))
                else:
                    #Term that refers to boundary data
                    loss = loss + jnp.mean(MSE(forward(x['boundary'],params['net']),x['uboundary']))
            if x['initial'] is not None:
                #Term that refers to initial data
                loss = loss + jnp.mean(MSE(forward(x['initial'],params['net']),x['uinitial']))
            if x['collocation'] is not None:
                #Term that refers to collocation points
                x_col = x['collocation'][:,:-1].reshape((x['collocation'].shape[0],x['collocation'].shape[1] - 1))
                t_col = x['collocation'][:,-1].reshape((x['collocation'].shape[0],1))
                if inverse:
                    loss = loss + jnp.mean(MSE(pde(lambda x,t: forward(jnp.append(x,t,1),params['net']),x_col,t_col,params['inverse']),0))
                else:
                    loss = loss + jnp.mean(MSE(pde(lambda x,t: forward(jnp.append(x,t,1),params['net']),x_col,t_col),0))
            return loss

    #Initialize Adam Optmizer
    if exp_decay:
        lr = optax.exponential_decay(lr,transition_steps,decay_rate)
    optimizer = optax.adam(lr,b1,b2,eps,eps_root)
    opt_state = optimizer.init(params)

    #Define the gradient function
    grad_loss = jax.jit(jax.grad(lf,0))

    #Define update function
    @jax.jit
    def update(opt_state,params,x):
        #Compute gradient
        grads = grad_loss(params,x)
        #Invert gradient of self-adaptative wheights
        if sa:
            for w in grads['sa']:
                grads['sa'][w] = - grads['sa'][w]
        #Calculate parameters updates
        updates, opt_state = optimizer.update(grads, opt_state)
        #Update parameters
        params = optax.apply_updates(params, updates)
        #Return state of optmizer and updated parameters
        return opt_state,params

    ###Training###
    t0 = time.time()
    #Initialize alive_bar for tracing in terminal
    with alive_bar(epochs) as bar:
        #For each epoch
        for e in range(epochs):
            #Update optimizer state and parameters
            opt_state,params = update(opt_state,params,data)
            #After epoch_print epochs
            if e % epoch_print == 0:
                #Compute elapsed time and current error
                l = 'Time: ' + str(round(time.time() - t0)) + ' s Loss: ' + str(jnp.round(lf(params,data),6))
                #If there is test data, compute current L2 error
                if test_data is not None:
                    #Compute L2 error
                    l2_test = L2error(forward(test_data['xt'],params['net']),test_data['u']).tolist()
                    l = l + ' L2 error: ' + str(jnp.round(l2_test,3))
                if inverse:
                    l = l + ' Parameter: ' + str(jnp.round(params['inverse'].tolist(),6))
                #Print
                print(l)
            if ((e % at_each == 0 and at_each != epochs) or e == epochs - 1) and save:
                #Save current parameters
                pickle.dump({'params': params,'width': width,'time': time.time() - t0,'loss': lf(params,data)},open(file_name + '_epoch' + str(e).rjust(6, '0') + '.pickle','wb'), protocol = pickle.HIGHEST_PROTOCOL)
            #Update alive_bar
            bar()
    #Define estimated function
    def u(xt):
        return forward(xt,params['net'])

    return {'u': u,'params': params,'forward': forward,'time': time.time() - t0}

#Training PINN
def train_Matern_PINN(data,width,pde,test_data = None,params = None,d = 2,N = 128,L = 1,alpha = 1,kappa = 1,sigma = 100,bsize = 1024,resample = False,epochs = 100,at_each = 10,activation = 'tanh',
    neumann = False,oper_neumann = None,inverse = False,initial_par = None,lr = 0.001,b1 = 0.9,b2 = 0.999,eps = 1e-08,eps_root = 0.0,key = 0,epoch_print = 1,save = False,file_name = 'result_pinn',
    exp_decay = True,transition_steps = 100,decay_rate = 0.9,mlp = True,ftype = None,fargs = None,q = 4,w = None,periodic = False,static = None,opt = 'LBFGS',float64 = False):
    """
    Train a Physics-informed Neural Network
    ----------
    Parameters
    ----------
    data : dict

        Data generated by the jinnax.data.generate_PINNdata function

    width : list

        A list with the width of each layer

    pde : function

        The partial differential operator. Its arguments are u, x and t

    test_data : dict, None

        A dictionay with test data for L2 error calculation generated by the jinnax.data.generate_PINNdata function. Default None for not calculating L2 error

    params : list

        Initial parameters for the neural network. Default None to initialize randomly

    d : int

        Dimension of the problem including the time variable if present. Default 2

    N : int

        Size of grid in each dimension. Default 128

    L : list of float

        The domain of the function in each coordinate is [0,L[1]]. If a float, repeat the same interval for all coordinates. Default 1

    kappa,alpha,sigma : float

        Parameters of the Matern process

    bsize : int

        Batch size for weak norm computation. Default 1024

    resample : logical

        Whether to resample the test functions at each epoch

    epochs : int

        Number of training epochs. Default 100

    at_each : int

        Save results for epochs multiple of at_each. Default 10

    activation : str

        The name of the activation function of the neural network. Default 'tanh'

    neumann : logical

        Whether to consider Neumann boundary conditions

    oper_neumann : function

        Penalization of Neumann boundary conditions

    inverse : logical

        Whether to estimate parameters of the PDE

    initial_par : jax.numpy.array

        Initial value of the parameters of the PDE in an inverse problem

    lr,b1,b2,eps,eps_root: float

        Hyperparameters of the Adam algorithm. Default lr = 0.001, b1 = 0.9, b2 = 0.999, eps = 1e-08, eps_root = 0.0

    key : int

        Seed for parameters initialization. Default 0

    epoch_print : int

        Number of epochs to calculate and print test errors. Default 1

    save : logical

        Whether to save the current parameters. Default False

    file_name : str

        File prefix to save the current parameters. Default 'result_pinn'

    exp_decay : logical

        Whether to consider exponential decay of learning rate. Default True

    transition_steps : int

        Number of steps for exponential decay. Default 100

    decay_rate : float

        Rate of exponential decay. Default 0.9

    mlp : logical

        Whether to consider modifed multilayer perceptron

    ftype : str

        Type of feature transformation to use: None, 'ff', 'daff','daff_bias', 'cheb', 'cheb_bias'.

    fargs : list

        Arguments for deature transformation:

        For 'ff': A list with the number of frequences and value of greatest frequence standard deviation.

        For 'daff' and 'daff' bias: A dicitionary with a list with the size of rectangles and the type of boundary condition. If its a list, than boundary conditions is dirichlet.

    q : int

        Power of weights mask. Default 4

    w : dict

        Initila weights for self-adaptive scheme.

    periodic : logical

        Whether to consider periodic test functions. Default False.

    static : function

        A static function to sum to the neural network output.

    opt : str

        Optimizer. Default LBFGS.

    float64 : logical

        Whether to train with float64

    Returns
    -------
    dict-like object with the estimated function, the estimated parameters, the neural network function for the forward pass and the loss, L2error and training time at each epoch
    """
    #Initialize architecture
    nnet = fconNN(width,get_activation(activation),key,mlp,ftype,fargs,static)
    if float64:
        forward = lambda x,params: nnet['forward'](x,params).astype(jnp.float64)
        assert jax.config.jax_enable_x64, "JAX is NOT running in float64 mode!"
    else:
        forward = nnet['forward']
    if params is not None:
        nnet['params'] = params
    if float64:
        data = to_float64(data)
        if test_data is not None:
            test_data = to_float64(test_data)

    #Generate from Matern process
    if sigma > 0:
        if isinstance(L,float) or isinstance(L,int):
            L = d*[L]
        #Grid for weak norm
        if float64:
            grid = [jnp.linspace(0,L[i],N,dtype = jnp.float64) for i in range(d)]
        else:
            grid = [jnp.linspace(0,L[i],N) for i in range(d)]
        grid = jnp.meshgrid(*grid, indexing='ij')
        grid = jnp.stack(grid, axis=-1).reshape((-1, d))
        #Set sigma
        if data['boundary'] is not None:
            gen = generate_matern_sample_batch(d = d,N = N,L = L,kappa = kappa,alpha = alpha,sigma = sigma)
            tf = gen(jax.random.split(jax.random.PRNGKey(key + 1),(bsize,))[:,0])
            if neumann:
                loss_boundary = oper_neumann(lambda x: forward(x,nnet['params']),data['boundary'])
            else:
                loss_boundary = jnp.mean(MSE(forward(data['boundary'],nnet['params']),data['uboundary']))
            output_w = pde(lambda x: forward(x,nnet['params']),grid)
            integralOmega = jax.vmap(lambda psi: jnp.mean(psi*output_w.reshape((N,) * d)))(tf)
            loss_res_weak = jnp.mean(integralOmega ** 2)
            sigma = float(jnp.sqrt(loss_boundary/loss_res_weak).tolist())
            del gen
            gen = generate_matern_sample_batch(d = d,N = N,L = L,kappa = kappa,alpha = alpha,sigma = sigma,periodic = periodic)
            tf = sigma*tf
        else:
            gen = generate_matern_sample_batch(d = d,N = N,L = L,kappa = kappa,alpha = alpha,sigma = sigma,periodic = periodic)
            tf = gen(jax.random.split(jax.random.PRNGKey(key + 1),(bsize,))[:,0])
        if float64 and tf is not None:
            tf = to_float64(tf)
            grid = to_float64(grid)

    #Define loss function
    @jax.jit
    def lf_each(params,x,k,tf,grid):
        if sigma > 0:
            #Term that refers to weak loss
            if resample:
                test_functions = to_float64(gen(jax.random.split(jax.random.PRNGKey(k[0]),(bsize,))[:,0]))
            else:
                test_functions = tf
        loss_sensor = loss_boundary = loss_initial = loss_res = loss_res_weak = 0
        if x['sensor'] is not None:
            #Term that refers to sensor data
            loss_sensor = MSE(forward(x['sensor'],params['net']),x['usensor'])
        if x['boundary'] is not None:
            if neumann:
                #Neumann coditions
                loss_boundary = oper_neumann(lambda x: forward(x,nnet['params']),x['boundary'])
            else:
                #Term that refers to boundary data
                loss_boundary = MSE(forward(x['boundary'],params['net']),x['uboundary'])
        if x['initial'] is not None:
            #Term that refers to initial data
            loss_initial = MSE(forward(x['initial'],params['net']),x['uinitial'])
        if x['collocation'] is not None and sigma == 0:
            if inverse:
                output = pde(lambda x: forward(x,params['net']),x['collocation'],params['inverse'])
                loss_res = MSE(output,0)
            else:
                output = pde(lambda x: forward(x,params['net']),x['collocation'])
                loss_res = MSE(output,0)
        if sigma > 0:
            #Term that refers to weak loss
            if inverse:
                output_w = pde(lambda x: forward(x,params['net']),grid,params['inverse'])
                integralOmega = jax.vmap(lambda psi: jnp.mean(psi*output_w.reshape((N,) * d)))(test_functions)
                loss_res_weak = jnp.mean(integralOmega ** 2)
            else:
                output_w = pde(lambda x: forward(x,params['net']),grid)
                integralOmega = jax.vmap(lambda psi: jnp.mean(psi*output_w.reshape((N,) * d)))(test_functions)
                loss_res_weak = jnp.mean(integralOmega ** 2)
        return {'ls': loss_sensor,'lb': loss_boundary,'li': loss_initial,'lc': loss_res,'lc_weak': loss_res_weak}

    @jax.jit
    def lf(params,x,k,tf,grid):
        l = lf_each(params,x,k,tf,grid)
        w = params['w']
        loss = jnp.mean((w['ws'] ** q)*l['ls']) + jnp.mean((w['wb'] ** q)*l['lb']) + jnp.mean((w['wi'] ** q)*l['li']) + jnp.mean((w['wc'] ** q)*l['lc']) + (w['wc_weak'] ** q)*l['lc_weak']
        if opt != 'LBFGS':
            return loss
        else:
            l2 = None
            if test_data is not None:
                l2 = L2error(forward(test_data['sensor'],params['net']),test_data['usensor'])
            return loss,{'loss': loss,'l2': l2}

    #Initialize self-adaptive weights
    if float64:
        typ = jnp.float64
    else:
        typ = jnp.float32
    if w is None:
        w = {'ws': jnp.array(1.0),'wb': jnp.array(1.0),'wi': jnp.array(1.0),'wc': jnp.array(1.0),'wc_weak': jnp.array(1.0)}
    if q != 0:
        if data['sensor'] is not None:
            w['ws'] = w['ws'] + 0.05*jax.random.normal(jax.random.PRNGKey(key+1),(data['sensor'].shape[0],1),dtype = typ)
        if data['boundary'] is not None:
            w['wb'] = w['wb'] + 0.05*jax.random.normal(jax.random.PRNGKey(key+2),(data['boundary'].shape[0],1),dtype = typ)
        if data['initial'] is not None:
            w['wi'] = w['wi'] + 0.05*jax.random.normal(jax.random.PRNGKey(key+3),(data['initial'].shape[0],1),dtype = typ)
        if data['collocation'] is not None:
            w['wc'] = w['wc'] + 0.05*jax.random.normal(jax.random.PRNGKey(key+4),(data['collocation'].shape[0],1),dtype = typ)

    #Store all parameters
    params = {'net': nnet['params'],'inverse': initial_par,'w': w}
    if float64:
        params = to_float64(params)

    #Save config file
    if save:
        pickle.dump({'train_data': data,'epochs': epochs,'activation': activation,'init_params': params,'forward': forward,'width': width,'pde': pde,'lr': lr,'b1': b1,'b2': b2,'eps': eps,'eps_root': eps_root,'key': key,'inverse': inverse},open(file_name + '_config.pickle','wb'), protocol = pickle.HIGHEST_PROTOCOL)

    #Initialize Adam Optmizer
    if opt != 'LBFGS':
        print('--------- GRADIENT DESCENT OPTIMIZER ---------')
        if exp_decay:
            lr = optax.exponential_decay(lr,transition_steps,decay_rate)
        optimizer = optax.adam(lr,b1,b2,eps,eps_root)
        opt_state = optimizer.init(params)

        #Define the gradient function
        grad_loss = jax.jit(jax.grad(lf,0))

        #Define update function
        @jax.jit
        def update(opt_state,params,x,k,tf,grid):
            #Compute gradient
            grads = grad_loss(params,x,k,tf,grid)
            #Calculate parameters updates
            updates, opt_state = optimizer.update(grads, opt_state)
            #Update parameters
            if q != 0:
                updates = {**updates, 'w': jax.tree_util.tree_map(lambda x: -x, updates['w'])} #Change signs of weights
            params = optax.apply_updates(params, updates)
            #Return state of optmizer and updated parameters
            return opt_state,params
    else:
        print('--------- LBFGS OPTIMIZER ---------')
        @jax.jit
        def loss_LBFGS(params):
            return lf(params,data,key + 234,tf,grid)
        solver = LBFGS(fun = loss_LBFGS,has_aux = True,maxiter = epochs,tol = 1e-9,verbose = False,linesearch = 'zoom',history_size = 10)  # linesearch='zoom' by default
        state = solver.init_state(params)
        if float64:
            assert_lbfgs_state_float64(state)

    ###Training###
    t0 = time.time()
    k = jax.random.split(jax.random.PRNGKey(key+234),(epochs,))
    sloss = []
    sL2 = []
    stime = []
    #Initialize alive_bar for tracing in terminal
    with alive_bar(epochs) as bar:
        #For each epoch
        for e in range(epochs):
            if opt != 'LBFGS':
                if float64 and e < 10:
                    assert_tree_float64(params, name="params (before update)")
                    grads = grad_loss(params, data, k[e,:], tf, grid)
                    check_grads_float64(grads)
                #Update optimizer state and parameters
                opt_state,params = update(opt_state,params,data,k[e,:],tf,grid)
                sloss.append(lf(params,data,k[e,:],tf,grid))
                if test_data is not None:
                    sL2.append(L2error(forward(test_data['sensor'],params['net']),test_data['usensor']))
                if float64 and e < 10:
                    assert_tree_float64(params, name="params (before update)")
                    grads = grad_loss(params, data, k[e,:], tf, grid)
                    check_grads_float64(grads)
            else:
                if float64 and e < 10:
                    assert_tree_float64(params, name="params (before LBFGS step)")
                params, state = solver.update(params, state)
                if float64 and e < 10:
                    assert_tree_float64(params, name="params (after LBFGS step)")
                    assert_lbfgs_state_float64(state)
                sL2.append(state.aux["l2"])
                sloss.append(state.aux["loss"])
                if float64 and e < 10:
                    assert_tree_float64(params, name="params (before update)")
            stime.append(time.time() - t0)
            #After epoch_print epochs
            if e % epoch_print == 0:
                #Compute elapsed time and current error
                l = 'Time: ' + str(round(time.time() - t0)) + ' s Loss: ' + str(jnp.round(sloss[-1],6))
                #If there is test data, compute current L2 error
                if test_data is not None:
                    #Compute L2 error
                    l = l + ' L2 error: ' + str(jnp.round(sL2[-1],6))
                if inverse:
                    l = l + ' Parameter: ' + str(jnp.round(params['inverse'].tolist(),6))
                #Print
                print(l)
            if ((e % at_each == 0 and at_each != epochs) or e == epochs - 1) and save:
                #Save current parameters
                pickle.dump({'params': params,'width': width,'time': stime,'loss': sloss,'L2error': sL2},open(file_name + '_epoch' + str(e).rjust(6, '0') + '.pickle','wb'), protocol = pickle.HIGHEST_PROTOCOL)
            #Update alive_bar
            bar()
    #Define estimated function
    def u(xt):
        return forward(xt,params['net'])

    return {'u': u,'params': params,'forward': forward,'time': time.time() - t0,'loss_each': lf_each(params,data,[key + 100],tf,grid),'loss': sloss,'L2error': sL2}


#Process result
def process_result(test_data,fit,train_data,plot = True,plot_test = True,times = 5,d2 = True,save = False,show = True,file_name = 'result_pinn',print_res = True,p = 1):
    """
    Process the results of a Physics-informed Neural Network
    ----------

    Parameters
    ----------
    test_data : dict

        A dictionay with test data for L2 error calculation generated by the jinnax.data.generate_PINNdata function

    fit : function

        The fitted function

    train_data : dict

        Training data generated by the jinnax.data.generate_PINNdata

    plot : logical

        Whether to generate plots comparing the exact and estimated solutions when the spatial dimension is one. Default True

    plot_test : logical

        Whether to plot the test data. Default True

    times : int

        Number of points along the time interval to plot. Default 5

    d2 : logical

        Whether to plot 2D plot when the spatial dimension is one. Default True

    save : logical

        Whether to save the plots. Default False

    show : logical

        Whether to show the plots. Default True

    file_name : str

        File prefix to save the plots. Default 'result_pinn'

    print_res : logical

        Whether to print the L2 error. Default True

    p : int

        Output dimension. Default 1

    Returns
    -------
    pandas data frame with L2 and MSE errors
    """

    #Dimension
    d = test_data['xt'].shape[1] - 1

    #Number of plots multiple of 5
    times = 5 * round(times/5)

    #Data
    td = get_train_data(train_data)
    xt_train = td['x']
    u_train = td['y']
    upred_train = fit(xt_train)
    upred_test = fit(test_data['xt'])

    #Results
    l2_error_test = L2error(upred_test,test_data['u']).tolist()
    MSE_test = jnp.mean(MSE(upred_test,test_data['u'])).tolist()
    l2_error_train = L2error(upred_train,u_train).tolist()
    MSE_train = jnp.mean(MSE(upred_train,u_train)).tolist()

    df = pd.DataFrame(np.array([l2_error_test,MSE_test,l2_error_train,MSE_train]).reshape((1,4)),
        columns=['l2_error_test','MSE_test','l2_error_train','MSE_train'])
    if print_res:
        print('L2 error test: ' + str(jnp.round(l2_error_test,6)) + ' L2 error train: ' + str(jnp.round(l2_error_train,6)) + ' MSE error test: ' + str(jnp.round(MSE_test,6)) + ' MSE error train: ' + str(jnp.round(MSE_train,6)) )

    #Plots
    if d == 1 and p ==1 and plot:
        plot_pinn1D(times,test_data['xt'],test_data['u'],upred_test,d2,save,show,file_name)
    elif p == 2 and plot:
        plot_pinn_out2D(times,test_data['xt'],test_data['u'],upred_test,save,show,file_name,plot_test)

    return df

#Plot results for d = 1
def plot_pinn1D(times,xt,u,upred,d2 = True,save = False,show = True,file_name = 'result_pinn',title_1d = '',title_2d = ''):
    """
    Plot the prediction of a 1D PINN
    ----------

    Parameters
    ----------
    times : int

        Number of points along the time interval to plot. Default 5

    xt : jax.numpy.array

        Test data xt array

    u : jax.numpy.array

        Test data u(x,t) array

    upred : jax.numpy.array

        Predicted upred(x,t) array on test data

    d2 : logical

        Whether to plot 2D plot. Default True

    save : logical

        Whether to save the plots. Default False

    show : logical

        Whether to show the plots. Default True

    file_name : str

        File prefix to save the plots. Default 'result_pinn'

    title_1d : str

        Title of 1D plot

    title_2d : str

        Title of 2D plot

    Returns
    -------
    None
    """
    #Initialize
    fig, ax = plt.subplots(int(times/5),5,figsize = (10*int(times/5),3*int(times/5)))
    tlo = jnp.min(xt[:,-1])
    tup = jnp.max(xt[:,-1])
    ylo = jnp.min(u)
    ylo = ylo - 0.1*jnp.abs(ylo)
    yup = jnp.max(u)
    yup = yup + 0.1*jnp.abs(yup)
    k = 0
    t_values = np.linspace(tlo,tup,times)

    #Create
    for i in range(int(times/5)):
        for j in range(5):
            if k < len(t_values):
                t = t_values[k]
                t = xt[jnp.abs(xt[:,-1] - t) == jnp.min(jnp.abs(xt[:,-1] - t)),-1][0].tolist()
                x_plot = xt[xt[:,-1] == t,:-1]
                y_plot = upred[xt[:,-1] == t,:]
                u_plot = u[xt[:,-1] == t,:]
                if int(times/5) > 1:
                    ax[i,j].plot(x_plot[:,0],u_plot[:,0],'b-',linewidth=2,label='Exact')
                    ax[i,j].plot(x_plot[:,0],y_plot,'r--',linewidth=2,label='Prediction')
                    ax[i,j].set_title('$t = %.2f$' % (t),fontsize=10)
                    ax[i,j].set_xlabel(' ')
                    ax[i,j].set_ylim([1.3 * ylo.tolist(),1.3 * yup.tolist()])
                else:
                    ax[j].plot(x_plot[:,0],u_plot[:,0],'b-',linewidth=2,label='Exact')
                    ax[j].plot(x_plot[:,0],y_plot,'r--',linewidth=2,label='Prediction')
                    ax[j].set_title('$t = %.2f$' % (t),fontsize=10)
                    ax[j].set_xlabel(' ')
                    ax[j].set_ylim([1.3 * ylo.tolist(),1.3 * yup.tolist()])
                k = k + 1

    #Title
    fig.suptitle(title_1d)
    fig.tight_layout()

    #Show and save
    fig = plt.gcf()
    if show:
        plt.show()
    if save:
        fig.savefig(file_name + '_slices.png')
    plt.close()

    #2d plot
    if d2:
        #Initialize
        fig, ax = plt.subplots(1,2)
        l1 = jnp.unique(xt[:,-1]).shape[0]
        l2 = jnp.unique(xt[:,0]).shape[0]

        #Create
        ax[0].pcolormesh(xt[:,-1].reshape((l2,l1)),xt[:,0].reshape((l2,l1)),u[:,0].reshape((l2,l1)),cmap = 'RdBu',vmin = ylo.tolist(),vmax = yup.tolist())
        ax[0].set_title('Exact')
        ax[1].pcolormesh(xt[:,-1].reshape((l2,l1)),xt[:,0].reshape((l2,l1)),upred[:,0].reshape((l2,l1)),cmap = 'RdBu',vmin = ylo.tolist(),vmax = yup.tolist())
        ax[1].set_title('Predicted')

        #Title
        fig.suptitle(title_2d)
        fig.tight_layout()

        #Show and save
        fig = plt.gcf()
        if show:
            plt.show()
        if save:
            fig.savefig(file_name + '_2d.png')
        plt.close()

#Plot results for d = 1
def plot_pinn_out2D(times,xt,u,upred,save = False,show = True,file_name = 'result_pinn',title = '',plot_test = True):
    """
    Plot the prediction of a PINN with 2D output
    ----------
    Parameters
    ----------
    times : int

        Number of points along the time interval to plot. Default 5

    xt : jax.numpy.array

        Test data xt array

    u : jax.numpy.array

        Test data u(x,t) array

    upred : jax.numpy.array

        Predicted upred(x,t) array on test data

    save : logical

        Whether to save the plots. Default False

    show : logical

        Whether to show the plots. Default True

    file_name : str

        File prefix to save the plots. Default 'result_pinn'

    title : str

        Title of plot

    plot_test : logical

        Whether to plot the test data. Default True

    Returns
    -------
    None
    """
    #Initialize
    fig, ax = plt.subplots(int(times/5),5,figsize = (10*int(times/5),3*int(times/5)))
    tlo = jnp.min(xt[:,-1])
    tup = jnp.max(xt[:,-1])
    xlo = jnp.min(u[:,0])
    xlo = xlo - 0.1*jnp.abs(xlo)
    xup = jnp.max(u[:,0])
    xup = xup + 0.1*jnp.abs(xup)
    ylo = jnp.min(u[:,1])
    ylo = ylo - 0.1*jnp.abs(ylo)
    yup = jnp.max(u[:,1])
    yup = yup + 0.1*jnp.abs(yup)
    k = 0
    t_values = np.linspace(tlo,tup,times)

    #Create
    for i in range(int(times/5)):
        for j in range(5):
            if k < len(t_values):
                t = t_values[k]
                t = xt[jnp.abs(xt[:,-1] - t) == jnp.min(jnp.abs(xt[:,-1] - t)),-1][0].tolist()
                xpred_plot = upred[xt[:,-1] == t,0]
                ypred_plot = upred[xt[:,-1] == t,1]
                if plot_test:
                    x_plot = u[xt[:,-1] == t,0]
                    y_plot = u[xt[:,-1] == t,1]
                if int(times/5) > 1:
                    if plot_test:
                        ax[i,j].plot(x_plot,y_plot,'b-',linewidth=2,label='Exact')
                    ax[i,j].plot(xpred_plot,ypred_plot,'r-',linewidth=2,label='Prediction')
                    ax[i,j].set_title('$t = %.2f$' % (t),fontsize=10)
                    ax[i,j].set_xlabel(' ')
                    ax[i,j].set_ylim([1.3 * ylo.tolist(),1.3 * yup.tolist()])
                else:
                    if plot_test:
                        ax[j].plot(x_plot,y_plot,'b-',linewidth=2,label='Exact')
                    ax[j].plot(xpred_plot,ypred_plot,'r-',linewidth=2,label='Prediction')
                    ax[j].set_title('$t = %.2f$' % (t),fontsize=10)
                    ax[j].set_xlabel(' ')
                    ax[j].set_ylim([1.3 * ylo.tolist(),1.3 * yup.tolist()])
                k = k + 1

    #Title
    fig.suptitle(title)
    fig.tight_layout()

    #Show and save
    fig = plt.gcf()
    if show:
        plt.show()
    if save:
        fig.savefig(file_name + '_slices.png')
    plt.close()

#Get train data in one array
def get_train_data(train_data):
    """
    Process training sample
    ----------

    Parameters
    ----------
    train_data : dict

        A dictionay with train data generated by the jinnax.data.generate_PINNdata function

    Returns
    -------
    dict with the processed training data
    """
    xdata = None
    ydata = None
    xydata = None
    if train_data['sensor'] is not None:
        sensor_sample = train_data['sensor'].shape[0]
        xdata = train_data['sensor']
        ydata = train_data['usensor']
        xydata = jnp.column_stack((train_data['sensor'],train_data['usensor']))
    else:
        sensor_sample = 0
    if train_data['boundary'] is not None:
        boundary_sample = train_data['boundary'].shape[0]
        if xdata is not None:
            xdata = jnp.vstack((xdata,train_data['boundary']))
            ydata = jnp.vstack((ydata,train_data['uboundary']))
            xydata = jnp.vstack((xydata,jnp.column_stack((train_data['boundary'],train_data['uboundary']))))
        else:
            xdata = train_data['boundary']
            ydata = train_data['uboundary']
            xydata = jnp.column_stack((train_data['boundary'],train_data['uboundary']))
    else:
        boundary_sample = 0
    if train_data['initial'] is not None:
        initial_sample = train_data['initial'].shape[0]
        if xdata is not None:
            xdata = jnp.vstack((xdata,train_data['initial']))
            ydata = jnp.vstack((ydata,train_data['uinitial']))
            xydata = jnp.vstack((xydata,jnp.column_stack((train_data['initial'],train_data['uinitial']))))
        else:
            xdata = train_data['initial']
            ydata = train_data['uinitial']
            xydata = jnp.column_stack((train_data['initial'],train_data['uinitial']))
    else:
        initial_sample = 0
    if train_data['collocation'] is not None:
        collocation_sample = train_data['collocation'].shape[0]
    else:
        collocation_sample = 0

    return {'xy': xydata,'x': xdata,'y': ydata,'sensor_sample': sensor_sample,'boundary_sample': boundary_sample,'initial_sample': initial_sample,'collocation_sample': collocation_sample}

#Process training
def process_training(test_data,file_name,at_each = 100,bolstering = True,mc_sample = 10000,save = False,file_name_save = 'result_pinn',key = 0,ec = 1e-6,lamb = 1):
    """
    Process the training of a Physics-informed Neural Network
    ----------

    Parameters
    ----------
    test_data : dict

        A dictionay with test data for L2 error calculation generated by the jinnax.data.generate_PINNdata function

    file_name : str

        Name of the files saved during training

    at_each : int

        Compute results for epochs multiple of at_each. Default 100

    bolstering : logical

        Whether to compute bolstering mean square error. Default True

    mc_sample : int

        Number of sample for Monte Carlo integration in bolstering. Default 10000

    save : logical

        Whether to save the training results. Default False

    file_name_save : str

        File prefix to save the plots and the L2 error. Default 'result_pinn'

    key : int

        Key for random samples in bolstering. Default 0

    ec : float

        Stopping criteria error for EM algorithm in bolstering. Default 1e-6

    lamb : float

        Hyperparameter of EM algorithm in bolstering. Default 1

    Returns
    -------
    pandas data frame with training results
    """
    #Config
    config = pickle.load(open(file_name + '_config.pickle', 'rb'))
    epochs = config['epochs']
    train_data = config['train_data']
    forward = config['forward']

    #Get train data
    td = get_train_data(train_data)
    xydata = td['xy']
    xdata = td['x']
    ydata = td['y']
    sensor_sample = td['sensor_sample']
    boundary_sample = td['boundary_sample']
    initial_sample = td['initial_sample']
    collocation_sample = td['collocation_sample']

    #Generate keys
    if bolstering:
        keys = jax.random.split(jax.random.PRNGKey(key),epochs)

    #Initialize loss
    train_mse = []
    test_mse = []
    train_L2 = []
    test_L2 = []
    bolstX = []
    bolstXY = []
    loss = []
    time = []
    ep = []

    #Process training
    with alive_bar(epochs) as bar:
        for e in range(epochs):
            if (e % at_each == 0 and at_each != epochs) or e == epochs - 1:
                ep = ep + [e]

                #Read parameters
                params = pickle.load(open(file_name + '_epoch' + str(e).rjust(6, '0') + '.pickle','rb'))

                #Time
                time = time + [params['time']]

                #Define learned function
                def psi(x):
                    return forward(x,params['params']['net'])

                #Train MSE and L2
                if xdata is not None:
                    train_mse = train_mse + [jnp.mean(MSE(psi(xdata),ydata)).tolist()]
                    train_L2 = train_L2 + [L2error(psi(xdata),ydata).tolist()]
                else:
                    train_mse = train_mse + [None]
                    train_L2 = train_L2 + [None]

                #Test MSE and L2
                test_mse = test_mse + [jnp.mean(MSE(psi(test_data['xt']),test_data['u'])).tolist()]
                test_L2 = test_L2 + [L2error(psi(test_data['xt']),test_data['u']).tolist()]

                #Bolstering
                if bolstering:
                    bX = []
                    bXY = []
                    for method in ['chi','mm','mpe']:
                        kxy = gk.kernel_estimator(data = xydata,key = keys[e,0],method = method,lamb = lamb,ec = ec,psi = psi)
                        kx = gk.kernel_estimator(data = xdata,key = keys[e,0],method = method,lamb = lamb,ec = ec,psi = psi)
                        bX = bX + [gb.bolstering(psi,xdata,ydata,kx,key = keys[e,0],mc_sample = mc_sample).tolist()]
                        bXY = bXY + [gb.bolstering(psi,xdata,ydata,kxy,key = keys[e,0],mc_sample = mc_sample).tolist()]
                    for bias in [1/jnp.sqrt(xdata.shape[0]),1/xdata.shape[0],1/(xdata.shape[0] ** 2),1/(xdata.shape[0] ** 3),1/(xdata.shape[0] ** 4)]:
                        kx = gk.kernel_estimator(data = xydata,key = keys[e,0],method = 'hessian',lamb = lamb,ec = ec,psi = psi,bias = bias)
                        bX = bX + [gb.bolstering(psi,xdata,ydata,kx,key = keys[e,0],mc_sample = mc_sample).tolist()]
                    bolstX = bolstX + [bX]
                    bolstXY = bolstXY + [bXY]
                else:
                    bolstX = bolstX + [None]
                    bolstXY = bolstXY + [None]

                #Loss
                loss = loss + [params['loss'].tolist()]

                #Delete
                del params, psi
            #Update alive_bar
            bar()

    #Bolstering results
    if bolstering:
        bolstX = jnp.array(bolstX)
        bolstXY = jnp.array(bolstXY)

    #Create data frame
    if bolstering:
        df = pd.DataFrame(np.column_stack([ep,time,[sensor_sample] * len(ep),[boundary_sample] * len(ep),[initial_sample] * len(ep),[collocation_sample] * len(ep),loss,
            train_mse,test_mse,train_L2,test_L2,bolstX[:,0],bolstXY[:,0],bolstX[:,1],bolstXY[:,1],bolstX[:,2],bolstXY[:,2],bolstX[:,3],bolstX[:,4],bolstX[:,5],bolstX[:,6],bolstX[:,7]]),
            columns=['epoch','training_time','sensor_sample','boundary_sample','initial_sample','collocation_sample','loss','train_mse','test_mse','train_L2','test_L2','bolstX_chi','bolstXY_chi','bolstX_mm','bolstXY_mm','bolstX_mpe','bolstXY_mpe','bolstHessian_sqrtn','bolstHessian_n','bolstHessian_n2','bolstHessian_n3','bolstHessian_n4'])
    else:
        df = pd.DataFrame(np.column_stack([ep,time,[sensor_sample] * len(ep),[boundary_sample] * len(ep),[initial_sample] * len(ep),[collocation_sample] * len(ep),loss,
            train_mse,test_mse,train_L2,test_L2]),
            columns=['epoch','training_time','sensor_sample','boundary_sample','initial_sample','collocation_sample','loss','train_mse','test_mse','train_L2','test_L2'])
    if save:
        df.to_csv(file_name_save + '.csv',index = False)

    return df

#Demo video for training1D PINN
def demo_train_pinn1D(test_data,file_name,at_each = 100,times = 5,d2 = True,file_name_save = 'result_pinn_demo',title = '',framerate = 10):
    """
    Demo video with the training of a 1D PINN
    ----------

    Parameters
    ----------
    test_data : dict

        A dictionay with test data for L2 error calculation generated by the jinnax.data.generate_PINNdata function

    file_name : str

        Name of the files saved during training

    at_each : int

        Compute results for epochs multiple of at_each. Default 100

    times : int

        Number of points along the time interval to plot. Default 5

    d2 : logical

        Whether to make video demo of 2D plot. Default True

    file_name_save : str

        File prefix to save the plots and videos. Default 'result_pinn_demo'

    title : str

        Title for plots

    framerate : int

        Framerate for video. Default 10

    Returns
    -------
    None
    """
    #Config
    with open(file_name + '_config.pickle', 'rb') as file:
        config = pickle.load(file)
    epochs = config['epochs']
    train_data = config['train_data']
    forward = config['forward']

    #Get train data
    td = get_train_data(train_data)
    xt = td['x']
    u = td['y']

    #Create folder to save plots
    os.system('mkdir ' + file_name_save)

    #Create images
    k = 1
    with alive_bar(epochs) as bar:
        for e in range(epochs):
            if e % at_each == 0 or e == epochs - 1:
                #Read parameters
                params = pd.read_pickle(file_name + '_epoch' + str(e).rjust(6, '0') + '.pickle')

                #Define learned function
                def psi(x):
                    return forward(x,params['params']['net'])

                #Compute L2 train, L2 test and loss
                loss = params['loss']
                L2_train = L2error(psi(xt),u)
                L2_test = L2error(psi(test_data['xt']),test_data['u'])
                title_epoch = title + ' Epoch = ' + str(e) + ' L2 train = ' + str(round(L2_train,6)) + ' L2 test = ' + str(round(L2_test,6))

                #Save plot
                plot_pinn1D(times,test_data['xt'],test_data['u'],psi(test_data['xt']),d2,save = True,show = False,file_name = file_name_save + '/' + str(k),title_1d = title_epoch,title_2d = title_epoch)
                k = k + 1

                #Delete
                del params, psi, loss, L2_train, L2_test, title_epoch
            #Update alive_bar
            bar()
    #Create demo video
    os.system('ffmpeg -framerate ' + str(framerate) + ' -i ' + file_name_save + '/' + '%00d_slices.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p ' + file_name_save + '/' + file_name_save + '_slices.mp4')
    if d2:
        os.system('ffmpeg -framerate ' + str(framerate) + ' -i ' + file_name_save + '/' + '%00d_2d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p ' + file_name_save + '/' + file_name_save + '_2d.mp4')

#Demo in time for 1D PINN
def demo_time_pinn1D(test_data,file_name,epochs,file_name_save = 'result_pinn_time_demo',title = '',framerate = 10):
    """
    Demo video with the time evolution of a 1D PINN
    ----------

    Parameters
    ----------
    test_data : dict

        A dictionay with test data for L2 error calculation generated by the jinnax.data.generate_PINNdata function

    file_name : str

        Name of the files saved during training

    epochs : list

        Which training epochs to plot

    file_name_save : str

        File prefix to save the plots and video. Default 'result_pinn_time_demo'

    title : str

        Title for plots

    framerate : int

        Framerate for video. Default 10

    Returns
    -------
    None
    """
    #Config
    with open(file_name + '_config.pickle', 'rb') as file:
        config = pickle.load(file)
    train_data = config['train_data']
    forward = config['forward']

    #Create folder to save plots
    os.system('mkdir ' + file_name_save)

    #Plot parameters
    tdom = jnp.unique(test_data['xt'][:,-1])
    ylo = jnp.min(test_data['u'])
    ylo = ylo - 0.1*jnp.abs(ylo)
    yup = jnp.max(test_data['u'])
    yup = yup + 0.1*jnp.abs(yup)

    #Open PINN for each epoch
    results = []
    upred = []
    for e in epochs:
        tmp = pd.read_pickle(file_name + '_epoch' + str(e).rjust(6, '0') + '.pickle')
        results = results + [tmp]
        upred = upred + [forward(test_data['xt'],tmp['params']['net'])]

    #Create images
    k = 1
    with alive_bar(len(tdom)) as bar:
        for t in tdom:
            #Test data
            xt_step = test_data['xt'][test_data['xt'][:,-1] == t]
            u_step = test_data['u'][test_data['xt'][:,-1] == t]
            #Initialize plot
            if len(epochs) > 1:
                fig, ax = plt.subplots(int(len(epochs)/2),2,figsize = (10,5*len(epochs)/2))
            else:
                fig, ax = plt.subplots(1,1,figsize = (10,5))
            #Create
            index = 0
            if int(len(epochs)/2) > 1:
                for i in range(int(len(epochs)/2)):
                    for j in range(min(2,len(epochs))):
                        upred_step = upred[index][test_data['xt'][:,-1] == t]
                        ax[i,j].plot(xt_step[:,0],u_step[:,0],'b-',linewidth=2,label='Exact')
                        ax[i,j].plot(xt_step[:,0],upred_step[:,0],'r--',linewidth=2,label='Prediction')
                        ax[i,j].set_title('Epoch = ' + str(epochs[index]),fontsize=10)
                        ax[i,j].set_xlabel(' ')
                        ax[i,j].set_ylim([1.3 * ylo.tolist(),1.3 * yup.tolist()])
                        index = index + 1
            elif len(epochs) > 1:
                for j in range(2):
                    upred_step = upred[index][test_data['xt'][:,-1] == t]
                    ax[j].plot(xt_step[:,0],u_step[:,0],'b-',linewidth=2,label='Exact')
                    ax[j].plot(xt_step[:,0],upred_step[:,0],'r--',linewidth=2,label='Prediction')
                    ax[j].set_title('Epoch = ' + str(epochs[index]),fontsize=10)
                    ax[j].set_xlabel(' ')
                    ax[j].set_ylim([1.3 * ylo.tolist(),1.3 * yup.tolist()])
                    index = index + 1
            else:
                upred_step = upred[index][test_data['xt'][:,-1] == t]
                ax.plot(xt_step[:,0],u_step[:,0],'b-',linewidth=2,label='Exact')
                ax.plot(xt_step[:,0],upred_step[:,0],'r--',linewidth=2,label='Prediction')
                ax.set_title('Epoch = ' + str(epochs[index]),fontsize=10)
                ax.set_xlabel(' ')
                ax.set_ylim([1.3 * ylo.tolist(),1.3 * yup.tolist()])
                index = index + 1


            #Title
            fig.suptitle(title + 't = ' + str(round(t,4)))
            fig.tight_layout()

            #Show and save
            fig = plt.gcf()
            fig.savefig(file_name_save + '/' + str(k) + '.png')
            k = k + 1
            plt.close()
            bar()

    #Create demo video
    os.system('ffmpeg -framerate ' + str(framerate) + ' -i ' + file_name_save + '/' + '%00d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p ' + file_name_save + '/' + file_name_save + '_time_demo.mp4')
