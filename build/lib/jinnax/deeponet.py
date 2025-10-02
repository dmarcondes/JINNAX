#Adapted from https://github.com/PredictiveIntelligenceLab/Physics-informed-DeepONets
import jax, os
import jax.numpy as np
from jax import random, grad, vmap, jit, hessian
from jax.example_libraries import optimizers
from jax.experimental.ode import odeint
from jax.nn import relu, elu
from jax.tree_util import tree_map
#from jax.config import config
#from jax.ops import index_update, index
#from jaxpi.utils import restore_checkpoint, save_checkpoint
from flax.training import checkpoints
from jax import lax, pmap
from jax.flatten_util import ravel_pytree
import ml_collections
import itertools
from functools import partial
from torch.utils import data
from tqdm import trange, tqdm
from jaxpi.samplers import BaseSampler, UniformSampler
from jaxpi.logging import Logger
import time
import wandb

def save_checkpoint(state, workdir, step, keep=5, name=None):
    # Create the workdir if it doesn't exist.
    if not os.path.isdir(workdir):
        os.makedirs(workdir)

    # Save the checkpoint.
    if jax.process_index() == 0:
        # Get the first replica's state and save it.
        state = jax.device_get(tree_map(lambda x: x[0], state))
        checkpoints.save_checkpoint(workdir, state, step=step, keep=keep)

def restore_checkpoint(state, workdir, step=None):
    # check if passed state is in a sharded state
    # if so, reduce to a single device sharding

    if isinstance(
        tree_map(lambda x: jnp.array(x).sharding, jax.tree.leaves(state.params))[0],
        jax.sharding.PmapSharding,
    ):
        state = tree_map(lambda x: x[0], state)

    # ensuring that we're in a single device setting
    assert isinstance(
        tree_map(lambda x: jnp.array(x).sharding, jax.tree.leaves(state.params))[0],
        jax.sharding.SingleDeviceSharding,
    )

    state = checkpoints.restore_checkpoint(workdir, state, step=step)
    return state

def get_base_config():
    """
    Base config file for training PINN for CSF in jaxpi

    Returns
    -------
    ml_collections config dictionary
    """
    #Get the default hyperparameter configuration.
    config = ml_collections.ConfigDict()
    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.tag = None
    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.branch_layers = [1024] + 4*[256]
    arch.trunk_layers = [2] + 4*[256]
    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.learning_rate = 1e-3
    optim.decay_rate = 0.9
    optim.decay_steps = 2000
    # Training
    config.training = training = ml_collections.ConfigDict()
    training.batch_size_per_device = 4096
    config.training.batch_size_train_data = 128
    # Weighting
    config.weights = {'b': 100,'res': 1,'data': 1,'ic' : 100}
    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 1000
    logging.log_errors = True
    logging.log_losses = True
    logging.log_weights = True
    logging.log_preds = False
    logging.log_grads = False
    logging.log_ntk = False
    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = 1000
    saving.num_keep_ckpts = 1000000
    #Seed
    config.seed = 10
    return config

#Periodic kernel
@jax.jit
def kernel_periodic(x1,x2,ls = 1,p = 1):
    return np.exp(-(np.sin(np.pi*np.abs(x1 - x2)/p) ** 2)/(2 * (ls ** 2)))

#Generate initial data
def generate_initial_data(N0,size,kernel = kernel_periodic,xl = 0,xu = 1,key = 0):
    x = np.linspace(xl,xu,N0)
    K = np.array([[kernel(x1,x2)] for x1 in x for x2 in x]).reshape((N0,N0))
    u = jax.random.multivariate_normal(key = jax.random.PRNGKey(key),mean = np.zeros((K.shape[0],)),cov = K,shape = (size,),method = 'svd')
    return u

class InitialDataSampler(BaseSampler):
    def __init__(self, data, batch_size,rng_key = jax.random.PRNGKey(1234)):
        super().__init__(batch_size, rng_key)
        self.data = data
        self.dim = data.shape[0]
    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key):
        "Generates data containing batch_size samples"
        idx = jax.random.choice(key, self.dim, (self.batch_size,), replace=False)
        return self.data[idx,:]

class DataSampler(BaseSampler):
    def __init__(self, data, batch_size,rng_key = jax.random.PRNGKey(1234)):
        super().__init__(batch_size, rng_key)
        self.data = data
        self.dim = data.shape[0]
    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key):
        "Generates data containing batch_size samples"
        idx = jax.random.choice(key, self.dim, (self.batch_size,), replace=False)
        return self.data[idx,:,:]

# Define MLP
def MLP(layers, activation=relu):
  ''' Vanilla MLP'''
  def init(rng_key):
      def init_layer(key, d_in, d_out):
          k1, k2 = random.split(key)
          glorot_stddev = 1. / np.sqrt((d_in + d_out) / 2.)
          W = glorot_stddev * random.normal(k1, (d_in, d_out))
          b = np.zeros(d_out)
          return W, b
      key, *keys = random.split(rng_key, len(layers))
      params = list(map(init_layer, keys, layers[:-1], layers[1:]))
      return params
  def apply(params, inputs):
      for W, b in params[:-1]:
          outputs = np.dot(inputs, W) + b
          inputs = activation(outputs)
      W, b = params[-1]
      outputs = np.dot(inputs, W) + b
      return outputs
  return init, apply

# Define modified MLP
def modified_MLP(layers, activation=relu):
  def xavier_init(key, d_in, d_out):
      glorot_stddev = 1. / np.sqrt((d_in + d_out) / 2.)
      W = glorot_stddev * random.normal(key, (d_in, d_out))
      b = np.zeros(d_out)
      return W, b
  def init(rng_key):
      U1, b1 =  xavier_init(random.PRNGKey(12345), layers[0], layers[1])
      U2, b2 =  xavier_init(random.PRNGKey(54321), layers[0], layers[1])
      def init_layer(key, d_in, d_out):
          k1, k2 = random.split(key)
          W, b = xavier_init(k1, d_in, d_out)
          return W, b
      key, *keys = random.split(rng_key, len(layers))
      params = list(map(init_layer, keys, layers[:-1], layers[1:]))
      return (params, U1, b1, U2, b2)
  def apply(params, inputs):
      params, U1, b1, U2, b2 = params
      U = activation(np.dot(inputs, U1) + b1)
      V = activation(np.dot(inputs, U2) + b2)
      for W, b in params[:-1]:
          outputs = activation(np.dot(inputs, W) + b)
          inputs = np.multiply(outputs, U) + np.multiply(1 - outputs, V)
      W, b = params[-1]
      outputs = np.dot(inputs, W) + b
      return outputs
  return init, apply

# Define the model
class PI_DeepONet:
    def __init__(self, config):
        # Network initialization and evaluation functions
        self.branch_init, self.branch_apply = modified_MLP(config.arch.branch_layers, activation = np.tanh)
        self.trunk_init, self.trunk_apply = modified_MLP(config.arch.trunk_layers, activation = np.tanh)

        # Initialize
        branch_params = self.branch_init(rng_key = random.PRNGKey(config.seed))
        trunk_params = self.trunk_init(rng_key = random.PRNGKey(config.seed + 1))
        params = (branch_params, trunk_params)

        # Use optimizers to set optimizer initialization and update functions
        self.opt_init, \
        self.opt_update, \
        self.get_params = optimizers.adam(optimizers.exponential_decay(config.optim.learning_rate,
                                                                      decay_steps = config.optim.decay_steps,
                                                                      decay_rate = config.optim.decay_rate))
        self.opt_state = self.opt_init(params)

        # Used to restore the trained model parameters
        _, self.unravel_params = ravel_pytree(params)

        self.itercount = itertools.count()

        # Residual net and boundary condition loss
        self.loss_bc = config.loss_bc
        self.residual_net = config.residual_net
        self.config = config

        # Limits domain
        self.xl = config.xl
        self.xu = config.xu

        #Vmap neural net
        self.pred_fn = vmap(self.operator_net, (None, 0, 0, 0))

        #Vmap residual operator
        if self.residual_net is not None:
            self.r_pred_fn = vmap(self.residual_net, (None, 0, 0, 0))

        #Vmap train and test data
        self.pred_batch = vmap(
            vmap(
                vmap(self.operator_net, (None, None, 0, None)),(None,None,None,0)
            ),(None,0,None,None)
        )

        self.pred_batch_xt = vmap(
                vmap(self.operator_net, (None, 0, None, None)),(None,None,0,0))
        if self.residual_net is not None:
            self.r_pred_batch = vmap(
                vmap(
                    vmap(self.residual_net, (None, None, 0, None)),(None,None,None,0)
                ),(None,0,None,None)
            )

        #Data
        self.u_test = config.u_test
        self.u_train = config.u_train
        self.x_mesh = config.x_mesh
        self.t_mesh = config.t_mesh

        #Weights
        self.w = config.weights

    # Define DeepONet architecture
    @partial(jit, static_argnums=(0,))
    def operator_net(self, params, u, x, t):
        branch_params, trunk_params = params
        y = np.stack([x,t])
        B = self.branch_apply(branch_params, u)
        T = self.trunk_apply(trunk_params, y)
        outputs = np.sum(B * T)
        return outputs

    # Define residual loss
    @partial(jit, static_argnums=(0,))
    def loss_res(self, params, batch):
        # Compute forward pass
        pred = self.residual_net(self.operator_net,params,batch)
        # Compute loss
        loss = np.mean((pred)**2)
        return loss

    #Data loss
    @partial(jit, static_argnums=(0,))
    def loss_data(self,params,batch_train):
        pred = self.pred_batch(params,batch_train['u0'],batch_train['x'],batch_train['t'])
        return np.mean((pred - batch_train['u']) ** 2)

    # Define initial condition loss
    @partial(jit, static_argnums=(0,))
    def loss_ic(self, params, batch):
        # Compute forward pass
        pred = self.pred_batch_xt(params,batch['u0'], self.x_mesh, np.zeros(self.x_mesh.shape[0]))
        # Compute loss
        loss = np.mean((pred - batch['u0'].transpose())**2)
        return loss

    # Define total loss
    @partial(jit, static_argnums=(0,))
    def loss(self, params, batch, batch_train):
        loss_bc = 0.0
        loss_data = 0.0
        loss_res = 0.0
        loss_ic = 0.0
        if self.loss_bc is not None:
            loss_bc = self.loss_bc(self.pred_batch,params,{'u0': batch['u0'],'t': batch['t_bc']},self.xl,self.xu)
        if self.residual_net is not None:
            loss_res = self.loss_res(params, batch)
        if batch_train is not None:
            loss_data = self.loss_data(params,batch_train)
        loss = self.w['b'] * loss_bc +  self.w['res'] * loss_res + self.w['data'] * loss_data + self.w['ic'] * self.loss_ic(params,batch)
        return loss

    # Define a compiled update step
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, batch, batch_train):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params, batch, batch_train)
        return self.opt_update(i, g, opt_state)

    def evaluator(self,batch,batch_train):
        log_dict = {}
        params = self.get_params(self.opt_state)
        #Test loss
        if self.u_test is not None:
            pred = self.pred_batch(params, self.u_test[:,0,:], self.x_mesh, self.t_mesh)
            log_dict['test_L2'] = np.mean(np.sqrt(np.mean((pred - self.u_test) ** 2,[1,2])/np.mean((self.u_test) ** 2,[1,2])))

        #Train
        if self.loss_bc is not None:
            log_dict['bc_loss'] = self.loss_bc(self.pred_batch,params,{'u0': batch['u0'],'t': batch['t_bc']},self.xl,self.xu)
        if self.residual_net is not None:
            log_dict['res_loss'] = self.loss_res(params,batch)
        log_dict['ic_loss'] = self.loss_ic(params,batch)
        if batch_train is not None:
            log_dict['data_loss'] = self.loss_data(params,batch_train)

        return log_dict

    # Optimize parameters in a loop
    def train(self):
        config = self.config
        if config.save_wandb:
            wandb_config = config.wandb
            wandb.init(project = wandb_config.project, name = wandb_config.name)

        #Initialize the initial data sampler
        if config.initial_data is None:
            initial_data = generate_initial_data(config.N0,int(config.size),kernel = config.kernel,xl = config.xl,xu = config.xu,key = 0)
        else:
            initial_data = config.initial_data
        initial_sampler = iter(InitialDataSampler(initial_data, config.N))

        # Initialize the residual sampler
        dom = np.array([[config.xl, config.xu],[config.tl, config.tu]])
        res_sampler = iter(UniformSampler(dom, config.Q))

        # Initialize the boundary condition sampler
        bc_sampler = iter(UniformSampler(np.array([[config.tl, config.tu]]), config.N0))

        #Initialize the training data sampler
        batch_train = None
        if config.u_train is not None:
            data_sampler = iter(DataSampler(config.u_train, config.training.batch_size_train_data))

        # Logger
        logger = Logger()
        batch = {'u0': None,'x': None,'t': None,'t_bc': None}

        #Train
        print("Training DeepONet...")
        start_time = time.time()
        t0 = start_time
        for step in range(config.training.max_steps):
            print(step)
            batch['u0'] = next(initial_sampler)[0,:,:]
            res_data_tmp = next(res_sampler)[0,:,:]
            batch['x'] = res_data_tmp[:,0]
            batch['t'] = res_data_tmp[:,1]
            batch['t_bc'] = next(bc_sampler)[0,:]
            if config.u_train is not None:
                u = next(data_sampler)
                u = u.reshape((u.shape[1],u.shape[2],u.shape[3]))
                batch_train = {'u0': u[:,0,:],'u': u,'t': self.t_mesh,'x': self.x_mesh}

            #Step
            self.opt_state = self.step(next(self.itercount), self.opt_state, batch, batch_train)

            # Log training metrics, only use host 0 to record results
            if step % config.logging.log_every_steps == 0:
                # Get the first replica of the state and batch
                log_dict = self.evaluator(batch, batch_train)
                if config.save_wandb:
                    wandb.log(log_dict, step)
                end_time = time.time()
                logger.log_iter(step, start_time, end_time, log_dict)
                start_time = end_time
                print(log_dict)

            # Saving
            if config.saving.save_every_steps is not None:
                if (step + 1) % config.saving.save_every_steps == 0 or (
                    step + 1
                ) == config.training.max_steps:
                    ckpt_path = os.path.join(os.getcwd(),"ckpt",config.wandb.name)
                    save_checkpoint(self.opt_state, ckpt_path, step, keep=config.saving.num_keep_ckpts)


        #Run summary
        log_dict['total_time'] = time.time() - t0
        log_dict['epochs'] = step + 1

        return log_dict

# Define PDE residual
def bc_loss_periodic(pred_batch,params,batch,xl,xu):
    pred_xl = pred_batch(
        params, batch['u0'], xl + np.zeros((1,)), batch['t'].reshape((batch['t'].shape[0],))
    )
    pred_xu = pred_batch(
        params, batch['u0'], xu + np.zeros((1,)), batch['t'].reshape((batch['t'].shape[0],))
    )
    return np.mean((pred_xl - pred_xu) ** 2)
