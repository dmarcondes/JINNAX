#Funtions to train PINNs for csf
import jax
import jax.numpy as jnp
import time
import os
from absl import logging
from jax.tree_util import tree_map
import numpy as np
import scipy.io
import ml_collections
import wandb
from jaxpi.samplers import UniformSampler
from jaxpi.logging import Logger
from jaxpi.utils import save_checkpoint
import ml_collections
from functools import partial
from jax import lax, jit, grad, vmap, jacrev, hessian
from jax.tree_util import tree_map
import optax
from jaxpi import archs
from jaxpi.models import ForwardIVP
from jaxpi.evaluator import BaseEvaluator
from jaxpi.utils import ntk_fn
from jinnax import class_csf

def get_base_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.mode = "train"

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.tag = None

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "ModifiedMlp"
    arch.num_layers = 4
    arch.hidden_dim = 256
    arch.out_dim = 2
    arch.activation = "tanh"
    arch.reparam = ml_collections.ConfigDict(
        {"type": "weight_fact", "mean": 1.0, "stddev": 0.1}
    )

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "Adam"
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.learning_rate = 1e-3
    optim.decay_rate = 0.9
    optim.decay_steps = 2000
    optim.grad_accum_steps = 0
    optim.warmup_steps = 0
    optim.grad_accum_steps = 0

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 150000
    training.batch_size_per_device = 4096

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"
    weighting.init_weights = ml_collections.ConfigDict(
        {"ic": 1.0,
        "res1": 1.0,
        "res2": 1.0,
        'rd': 1.0,
        'ld': 1.0,
        'ln': 1.0
        }
    )
    weighting.momentum = 0.9
    weighting.update_every_steps = 1000

    weighting.use_causal = True
    weighting.causal_tol = 1.0
    weighting.num_chunks = 16
    optim.staircase = False

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 100
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

    # # Input shape for initializing Flax models
    config.input_dim = 2

    return config


#Demo in time for 2D PINN
def demo_time_CSF(data,radius,file_name_save = 'result_pinn_CSF_demo',title = '',framerate = 10,ffmpeg = 'ffmpeg'):
    """
    Demo video with the time evolution of a 2D PINN
    ----------
    Parameters
    ----------
    data : jax.array

        Data with the predict values

    radius : float

        Radius of circle

    file_name_save : str

        File prefix to save the plots and video. Default 'result_pinn_CSF_demo'

    title : str

        Title for plots

    framerate : int

        Framerate for video. Default 10

    ffmpeg : str

        Path to ffmpeg

    Returns
    -------
    None
    """
    #Create folder to save plots
    os.system('mkdir ' + file_name_save)

    #Plot parameters
    tdom = jnp.unique(data[:,0])
    ylo = jnp.min(data[:,-2:])
    ylo = ylo - 0.1*jnp.abs(ylo)
    yup = jnp.max(data[:,-2:])
    yup = yup + 0.1*jnp.abs(yup)

    #Circle data
    circle = jnp.array([[radius*jnp.sin(t),radius*jnp.cos(t)] for t in jnp.linspace(0,2*jnp.pi,1000)])

    #Create images
    k = 1
    with alive_bar(len(tdom)) as bar:
        for t in tdom:
            #Test data
            x_step = data[data[:,0] == t,1]
            ux_step = data[data[:,0] == t,2]
            uy_step = data[data[:,0] == t,3]
            #Initialize plot
            fig, ax = plt.subplots(1,1,figsize = (10,10))
            #Create
            ax.plot(ux_step,uy_step,'b-',linewidth=2,label='Exact')
            ax.plot(circle[:,0],circle[:,1],'r-',linewidth=2,label='Prediction')
            ax.set_xlabel(' ')
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
    os.system(ffmpeg + ' -framerate ' + str(framerate) + ' -i ' + file_name_save + '/' + '%00d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p ' + file_name_save + '/' + file_name_save + '_time_demo.mp4')


def train_csf(config: ml_collections.ConfigDict, workdir: str):
    if config.save_wandb:
        wandb_config = config.wandb
        wandb.init(project = wandb_config.project, name = wandb_config.name)

    # Define the time and space domain
    t = jnp.linspace(config.tl, config.tu, config.Nt)
    t0 = t[0]
    t1 = t[-1]
    dom = jnp.array([[t0, t1], [xl, xu]])

    # Initialize the residual sampler
    res_sampler = iter(UniformSampler(dom, config.training.batch_size_per_device))

    # Initialize the model
    if config.type_csf == 'DN':
        model = class_csf.DN_csf(config, uinitial, t)

    # Logger
    logger = Logger()

    # Initialize evaluator
    evaluator = CSF_DN_Evaluator(config, model,x0_test, tb_test, xc_test, tc_test, u1_0_test, u2_0_test)

    # jit warm up
    print("Waiting for JIT...")
    start_time = time.time()
    for step in range(config.training.max_steps):
        batch = next(res_sampler)
        model.state = model.step(model.state, batch)

        # Update weights if necessary
        if config.weighting.scheme in ["grad_norm", "ntk"]:
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch)

        # Log training metrics, only use host 0 to record results
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                # Get the first replica of the state and batch
                state = jax.device_get(tree_map(lambda x: x[0], model.state))
                batch = jax.device_get(tree_map(lambda x: x[0], batch))
                log_dict = evaluator(state, batch)
                wandb.log(log_dict, step)
                end_time = time.time()
                logger.log_iter(step, start_time, end_time, log_dict)
                start_time = end_time

        # Saving
        if config.saving.save_every_steps is not None:
            if (step + 1) % config.saving.save_every_steps == 0 or (
                step + 1
            ) == config.training.max_steps:
                ckpt_path = os.path.join(os.getcwd(),"ckpt",config.wandb.name)
                save_checkpoint(model.state, ckpt_path, keep=config.saving.num_keep_ckpts)

    return model

def uinitial(x):
  b = jnp.log(jnp.pi)/3 + 3*jnp.pi
  u1 = jnp.where(x <= -3*jnp.pi,x,-x*jnp.cos(-x))
  u2 = jnp.where(x <= -3*jnp.pi,-1*jnp.sin(jnp.exp(3 * (b + x))),-x*jnp.sin(-x))
  return u1,u2

def csf_circle(uinitial,xl,xu,tl,radius,file_name,Nt = 1000,N0 = 10000,Nc = 10000,Nb = 10000,type = 'DN',config = None,save_wandb = False,wandb_project = 'CSF_project',seed = 3284,workdir = '.'):
    #Set config file
    if config is None:
        config = get_base_config()

    wandb.project = wandb_project
    wandb.name = file_name
    config.seed = seed
    config.type_csf = type
    config.save_wandb = save_wandb
    config.xl = xl
    config.xu = xu
    config.tl = tl
    config.tu = tu
    config.radius = radius
    config.rd = jnp.append(uinitial(xu)[0],uinitial(xu)[1])
    config.Nt = Nt
    config.N0 = N0
    config.Nc = Nc
    config.Nb = Nb

    #Define model
    model = train_csf(config,workdir,unitial)
