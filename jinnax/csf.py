#Funtions to train PINNs for csf
from absl import logging
from alive_progress import alive_bar
from functools import partial
import jax
from jax import grad, hessian, jit, lax, vmap, jacrev
import jax.numpy as jnp
from jax.tree_util import tree_map
from jaxpi import archs
from jaxpi.evaluator import BaseEvaluator
from jaxpi.logging import Logger
from jaxpi.models import ForwardIVP
from jaxpi.samplers import UniformSampler
from jaxpi.utils import ntk_fn, restore_checkpoint, save_checkpoint
from jinnax import class_csf
import matplotlib.pyplot as plt
import ml_collections
import numpy as np
import optax
import os
import pandas as pd
import random
import scipy.io
import sys
import time
import wandb

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
    training.batch_size_per_device = 4096

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"
    weighting.momentum = 0.9
    weighting.update_every_steps = 1000

    weighting.use_causal = True
    weighting.causal_tol = 1.0
    weighting.num_chunks = 16
    optim.staircase = False

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

    # # Input shape for initializing Flax models
    config.input_dim = 2

    return config


#Demo in time for 2D PINN
def demo_time_CSF(data,type = 'DN',radius = None,file_name_save = 'result_pinn_CSF_demo',title = '',framerate = 10,ffmpeg = 'ffmpeg'):
    """
    Demo video with the time evolution of a CSF in a circle
    ----------
    Parameters
    ----------
    data : jax.array

        Data with the predicted values

    type : str

        Type of problem

    radius : float

        Radius of circle for types 'DN' and 'NN'

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
    if type == 'DN' or type == 'NN':
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
            ax.plot(ux_step,uy_step,'b-',linewidth=2)
            if type == 'DN' or type == 'NN':
                ax.plot(circle[:,0],circle[:,1],'r-',linewidth=2)
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
    os.system(ffmpeg + ' -framerate ' + str(framerate) + ' -i ' + file_name_save + '/' + '%00d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p ' + file_name_save + '_time_demo.mp4')


def train_csf(config: ml_collections.ConfigDict):
    """
    Train PINN for CSF in jaxpi
    ----------
    Parameters
    ----------
    config : ml_collections.ConfigDict

        Dictionary for training PINN in jaxpi

    uninitial : function

        Function that computes the initial condition


    Returns
    -------
    model, log_dict
    """
    if config.save_wandb:
        wandb_config = config.wandb
        wandb.init(project = wandb_config.project, name = wandb_config.name)

    # Define the time and space domain
    dom = jnp.array([[config.tl, config.tu], [config.xl, config.xu]])

    # Initialize the residual sampler
    res_sampler = iter(UniformSampler(dom, config.training.batch_size_per_device))

    # Initialize the model
    if config.type_csf == 'DN':
        model = class_csf.DN_csf(config)
    elif config.type_csf == 'NN':
        model = class_csf.NN_csf(config)
    elif config.type_csf == 'DD':
        model = class_csf.DD_csf(config)
    elif config.type_csf == 'closed':
        model = class_csf.closed_csf(config)

    # Logger
    logger = Logger()

    # Initialize evaluator
    key = jax.random.split(jax.random.PRNGKey(config.seed),4)
    x0_test = jax.random.uniform(key = jax.random.PRNGKey(key[0]),minval = config.xl,maxval = config.xu,shape = (config.N0,1))
    u1_0_test,u2_0_test = config.uinitial(x0_test)
    tb_test = jax.random.uniform(key = jax.random.PRNGKey(key[1]),minval = config.tl,maxval = config.tu,shape = (config.Nb,1))
    xc_test = jax.random.uniform(key = jax.random.PRNGKey(key[2]),minval = config.xl,maxval = config.xu,shape = (config.Nc ** 2,1))
    tc_test = jax.random.uniform(key = jax.random.PRNGKey(key[3]),minval = config.tl,maxval = config.tu,shape = (config.Nc ** 2,1))
    if config.type_csf == 'DN':
        evaluator = class_csf.DN_csf_Evaluator(config, model, x0_test, tb_test, xc_test, tc_test, u1_0_test, u2_0_test)
    elif config.type_csf == 'NN':
        evaluator = class_csf.NN_csf_Evaluator(config, model, x0_test, tb_test, xc_test, tc_test, u1_0_test, u2_0_test)
    elif config.type_csf == 'DD':
        evaluator = class_csf.DD_csf_Evaluator(config, model, x0_test, tb_test, xc_test, tc_test, u1_0_test, u2_0_test)
    elif config.type_csf == 'closed':
        evaluator = class_csf.closed_csf_Evaluator(config, model, x0_test, tb_test, xc_test, tc_test, u1_0_test, u2_0_test)

    # jit warm up
    print("Training CSF...")
    start_time = time.time()
    t0 = start_time
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
                if config.save_wandb:
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
                if config.type_csf == 'DN':
                    if log_dict['res1_test'] < config.res_tol and log_dict['res2_test'] < config.res_tol and log_dict['ic_rel_test'] < config.ic_tol and log_dict['ld_test'] < config.dn_tol and log_dict['rd_test'] < config.dn_tol and log_dict['ln_test'] < config.dn_tol:
                        break
                elif config.type_csf == 'NN':
                    if log_dict['res1_test'] < config.res_tol and log_dict['res2_test'] < config.res_tol and log_dict['ic_rel_test'] < config.ic_tol and log_dict['ld_test'] < config.dn_tol and log_dict['rd_test'] < config.dn_tol and log_dict['ln_test'] < config.dn_tol and log_dict['rn_test'] < config.dn_tol:
                        break
                elif config.type_csf == 'DD':
                    if log_dict['res1_test'] < config.res_tol and log_dict['res2_test'] < config.res_tol and log_dict['ic_rel_test'] < config.ic_tol and log_dict['ld_test'] < config.dn_tol and log_dict['rd_test'] < config.dn_tol:
                        break
                elif config.type_csf == 'closed':
                    if log_dict['res1_test'] < config.res_tol and log_dict['res2_test'] < config.res_tol and log_dict['ic_rel_test'] < config.ic_tol:
                        break

    #Run summary
    log_dict['total_time'] = time.time() - t0
    log_dict['epochs'] = step + 1

    return model, log_dict

def evaluate(config: ml_collections.ConfigDict):
    """
    Evaluate PINN for CSF trained in jaxpi
    ----------
    Parameters
    ----------
    config : ml_collections.ConfigDict

        Dictionary for training PINN in jaxpi

    uninitial : function

        Function that computes the initial condition


    Returns
    -------
    predicted values
    """
    # Initialize the model
    if config.type_csf == 'DN':
        model = class_csf.DN_csf(config)
    elif config.type_csf == 'NN':
        model = class_csf.NN_csf(config)
    elif config.type_csf == 'DD':
        model = class_csf.DD_csf(config)
    elif config.type_csf == 'closed':
        model = class_csf.closed_csf(config)

    # Restore the checkpoint
    ckpt_path = os.path.join(
        os.getcwd(), "ckpt", config.wandb.name,
    )
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params

    #Collocation data
    tx = jnp.array([[t,x] for t in jnp.linspace(config.tl, config.tu, config.Nt) for x in jnp.linspace(config.xl, config.xu, config.Nc)])

    #Predict
    u1_pred = model.u1_pred_fn(params, tx[:,0], tx[:,1])
    u2_pred = model.u2_pred_fn(params, tx[:,0], tx[:,1])

    #Save
    pred = jnp.append(tx,jnp.append(u1_pred.reshape((u1_pred.shape[0],1)),u2_pred.reshape((u2_pred.shape[0],1)),1),1)
    jnp.save(config.wandb.project + '_' + config.wandb.name + '.npy',pred)

    return pred

def csf(uinitial,xl,xu,tl,tu,type = 'DN',radius = None,file_name = 'test',Nt = 400,N0 = 10000,Nb = 10000,Nc = 500,config = None,save_wandb = False,wandb_project = 'CSF_project',seed = 534,demo = True,max_epochs = 150000,res_tol = 5e-5,dn_tol = 5e-4,ic_tol = 0.01,framerate = 10,ffmpeg = 'ffmpeg'):
    """
    Train PINN for CSF in jaxpi
    ----------
    Parameters
    ----------
    uninitial : function

        Function that computes the initial condition

    xl, xu, tl, tu : float

        Limits of the x and t domain

    type : str

        Type of CSF problem to train: 'DN' or 'NN' (in a circle), 'DD' or 'closed'

    radius : float

        Radius of the circle for 'DN' and 'NN'

    file_name : str

        File name to save results


    Nt : int

        Sample size for grid in t

    N0, Nb : int

        Initial and boundary condition test sample size

    Nc : int

        Number of points in each direction in sample size for PDE residuals


    config : ml_collections.ConfigDict

        Config dictionary to train PINNs in jaxpi. If not provided, use basic configurations

    save_wandb : logical

        Whether to save results in wandb

    wandb_project : str

        Name of wandb project

    seed : int

        Seed for initialising neural network

    demo : logical

        Whether to generate video with result

    max_epochs : int

        Maximum number of epochs to train

    res_tol, dn_tol, ic_tol : float

        Tolerance on test errors for early stop

    framerate : int

        Framerate for video. Default 10

    ffmpeg : str

        Path to ffmpeg

    Returns
    -------
    model, log_dict
    """
    #Set config file
    if config is None:
        config = get_base_config()

    config.wandb.project = wandb_project
    config.wandb.name = file_name
    config.seed = seed
    config.type_csf = type
    config.save_wandb = save_wandb
    config.uinitial = uinitial
    config.xl = xl
    config.xu = xu
    config.tl = tl
    config.tu = tu
    config.radius = radius
    config.rd = jnp.append(uinitial(xu)[0],uinitial(xu)[1])
    config.ld = jnp.append(uinitial(xl)[0],uinitial(xl)[1])
    config.Nt = Nt
    config.N0 = N0
    config.Nc = Nc
    config.Nb = Nb
    config.res_tol = res_tol
    config.dn_tol = dn_tol
    config.ic_tol = ic_tol
    config.training.max_steps = max_epochs
    if type == 'DN':
        config.weighting.init_weights = ml_collections.ConfigDict(
            {"ic": 1.0,
            "res1": 1.0,
            "res2": 1.0,
            'rd': 1.0,
            'ld': 1.0,
            'ln': 1.0
            }
        )
    elif type == 'NN':
        config.weighting.init_weights = ml_collections.ConfigDict(
            {"ic": 1.0,
            "res1": 1.0,
            "res2": 1.0,
            'rd': 1.0,
            'ld': 1.0,
            'ln': 1.0,
            'rn': 1.0
            }
        )
    elif type == 'DD':
        config.weighting.init_weights = ml_collections.ConfigDict(
            {"ic": 1.0,
            "res1": 1.0,
            "res2": 1.0,
            'rd': 1.0,
            'ld': 1.0
            }
        )
    if type == 'closed': #Add periodic condition
        config.weighting.init_weights = ml_collections.ConfigDict(
            {"ic": 1.0,
            "res1": 1.0,
            "res2": 1.0,
            'periodic1': 1.0,
            'periodic2': 1.0
            }
        )

    #Train model
    model, results = train_csf(config)

    #Evaluate
    pred = evaluate(config)

    #Generate demo
    if demo:
        demo_time_CSF(pred,radius,file_name_save = file_name,framerate = 10,ffmpeg = 'ffmpeg')

    #Print results
    pd_results = pd.DataFrame(list(results.items()))
    print(pd_results)
    pd_results.to_csv(file_name + '_results.csv')

    return results
