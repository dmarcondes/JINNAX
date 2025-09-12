from functools import partial

import jax
import jax.numpy as jnp
from jax import lax, jit, grad, vmap, jacrev, hessian
from jax.tree_util import tree_map

import optax

from jaxpi import archs
from jaxpi.models import ForwardIVP
from jaxpi.evaluator import BaseEvaluator
from jaxpi.utils import ntk_fn

class DN_csf(ForwardIVP):
    def __init__(self, config, uinitial, xl, xu, radius, rd, t_star):
        super().__init__(config)

        #Grid of time
        self.t_star = t_star

        #Initial condition function
        self.uinitial = uinitial

        #Boundary points
        self.xl = xl
        self.xu = xu

        #Radius left dirichlet condition
        self.radius = radius

        #Right dirichlet point
        self.rd = rd

        # Predictions over array of x fot t fixed
        self.u1_0_pred_fn = vmap(
            vmap(self.u1_net, (None, None, 0)), (None, None, 0)
        )
        self.u2_0_pred_fn = vmap(
            vmap(self.u2_net, (None, None, 0)), (None, None, 0)
        )

        #Prediction over array of t for x fixed
        self.u2_bound_pred_fn = vmap(
            vmap(self.u2_net, (None, 0, None)), (None, 0, None)
        )

        self.u1_bound_pred_fn = vmap(
            vmap(self.u1_net, (None, 0, None)), (None, 0, None)
        )

        #Vmap neural net
        self.u1_pred_fn = vmap(self.u1_net, (None, 0, 0))

        self.u2_pred_fn = vmap(self.u2_net, (None, 0, 0))

        #Vmap residual operator
        self.r_pred_fn = vmap(self.r_net, (None, 0, 0))

        #Derivatives on x for x fixed and t in a array
        self.u1_bound_x = vmap(vmap(grad(self.u1_net, argnums = 2), (None, 0, None)), (None, 0, None))
        self.u2_bound_x = vmap(vmap(grad(self.u2_net, argnums = 2), (None, 0, None)), (None, 0, None))

    #Neural net forward function
    def neural_net(self, params, t, x):
        t = t / self.t_star[-1]
        z = jnp.stack([t, x])
        _, outputs = self.state.apply_fn(params, z)
        u1 = outputs[0]
        u2 = outputs[1]
        return u1, u2

    #1st coordinate neural net forward function
    def u1_net(self, params, t, x):
        u1, _ = self.neural_net(params, t, x)
        return u1

    #2st coordinate neural net forward function
    def u2_net(self, params, t, x):
        _, u2 = self.neural_net(params, t, x)
        return u2

    #Residual operator
    def r_net(self, params, t, x):
        #Derivatives in x and t
        u1_x = grad(self.u1_net, argnums = 2)(params, t, x)
        u1_t = grad(self.u1_net, argnums = 1)(params, t, x)
        u2_x = grad(self.u2_net, argnums = 2)(params, t, x)
        u2_t = grad(self.u2_net, argnums = 1)(params, t, x)

        #Two derivatives in x
        u1_xx = hessian(self.u1_net, argnums = (2))(params, t, x)
        u2_xx = hessian(self.u2_net, argnums = (2))(params, t, x)

        #Each coordinate of residual operator
        return (u1_t - u1_xx/(u1_x ** 2 + u2_x ** 2 + 1e-5)) ** 2, (u2_t - u2_xx/(u1_x ** 2 + u2_x ** 2 + 1e-5)) ** 2

    #1st coordinate residual operator
    def r_net1(self, params, t, x):
        r1,_ = self.r_net(params, t, x)
        return r1

    #2nd coordinate residual operator
    def r_net2(self, params, t, x):
        _,r2 = self.r_net(params, t, x)
        return r2

    #Right Dirichlet Condition residual
    def res_right_dirichlet(self, params, t, x):
        #Apply neural net to point in the boundary
        u1 = self.u1_bound_pred_fn(params, t, x)
        u2 = self.u2_bound_pred_fn(params, t, x)
        #Return residual
        return (u1 - self.rd[0]) ** 2 + (u2 - self.rd[1]) ** 2

    #Right Dirichlet Condition residual non-vectorised
    def res_right_dirichlet_nv(self, params, t, x):
        #Apply neural net to point in the boundary
        u1 = self.u1_net(params, t, x)
        u2 = self.u2_net(params, t, x)
        #Return residual
        return (u1 - self.rd[0]) ** 2 + (u2 - self.rd[1]) ** 2

    #Left Dirichlet Condition residual
    def res_left_dirichlet(self, params, t, x):
        #Apply neural net to point in the boundary
        u1 = self.u1_bound_pred_fn(params, t, x)
        u2 = self.u2_bound_pred_fn(params, t, x)
        #Return residual
        return (jnp.sqrt(u1  ** 2 + u2 ** 2) - self.radius) ** 2

    #Left Dirichlet Condition residual non-vectorised
    def res_left_dirichlet_nv(self, params, t, x):
        #Apply neural net to point in the boundary
        u1 = self.u1_net(params, t, x)
        u2 = self.u2_net(params, t, x)
        #Return residual
        return (jnp.sqrt(u1  ** 2 + u2 ** 2) - self.radius) ** 2

    #Neumman condition residual non-vectorised
    def res_neumann_nv(self, params, t, x):
        #Apply neural net
        u1 = self.u1_net(params, t, x)
        u2 = self.u2_net(params, t, x)

        #Derivatives in x
        u1_x = grad(self.u1_net, argnums = 2)(params, t, x)
        u2_x = grad(self.u2_net, argnums = 2)(params, t, x)

        #Assuming that u(x,t) \in S, compute the vector normal to S at u(x,t)
        nS = jnp.append(u1,u2)/(jnp.sqrt(jnp.sum(jnp.append(u1,u2) ** 2)) + 1e-5)

        #Normal at u(x,y)
        nu = jnp.append(u2_x,(-1)*u1_x)/(jnp.sqrt(u1_x ** 2 + u2_x ** 2) + 1e-5)

        #Return inner product
        return jnp.sum(nS * nu) ** 2

    #Neumman condition residual
    def res_neumann(self, params, t, x):
        #Apply neural net to points in the boundary
        u1 = self.u1_bound_pred_fn(params, t, x)
        u2 = self.u2_bound_pred_fn(params, t, x)

        #Derivatives in x
        u1_x = self.u1_bound_x(params, t, x)
        u2_x = self.u2_bound_x(params, t, x)

        #Assuming that u(x,t) \in S, compute the vector normal to S at u(x,t)
        nS = jnp.append(u1,u2,1)/(jnp.sqrt(jnp.sum(jnp.append(u1,u2,1) ** 2,1)).reshape(u1.shape[0],1) + 1e-5)

        #Normal at u(x,y)
        nu = jnp.append(u2_x,(-1)*u1_x,1)/(jnp.sqrt(u1_x ** 2 + u2_x ** 2) + 1e-5)

        #Return inner product
        return jnp.sum(nS * nu,1).reshape(u1.shape[0],1) ** 2

    #Compute residuals with causal weights
    @partial(jit, static_argnums=(0,))
    def res_causal(self, params, batch):
        # Sort temporal coordinates
        t_sorted = batch[:, 0].sort()

        #Compute residuals
        res_pred1,res_pred2 = self.r_pred_fn(params, t_sorted, batch[:, 1])

        #Reshape
        res_pred1 = res_pred1.reshape(self.num_chunks, -1)
        res_pred2 = res_pred2.reshape(self.num_chunks, -1)

        #Compute mean residuals
        res_l1 = jnp.mean(res_pred1 ** 2, axis=1)
        res_l2 = jnp.mean(res_pred2 ** 2, axis=1)

        #Compute weights
        res_gamma1 = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ res_l1)))
        res_gamma2 = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ res_l2)))

        # Take minimum of the causal weights
        gamma = jnp.vstack([res_gamma1,res_gamma2])
        gamma = gamma.min(0)

        return res_l1, res_l2, gamma

    #Compute losses
    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        # Initial conditions loss
        u1_pred = self.u1_0_pred_fn(params, 0.0, batch[:, 1].reshape((batch.shape[0],1)))
        u2_pred = self.u2_0_pred_fn(params, 0.0, batch[:, 1].reshape((batch.shape[0],1)))
        u1_0,u2_0 = self.uinitial(batch[:, 1].reshape((batch.shape[0],1)))

        u1_ic_loss = jnp.mean((u1_pred - u1_0) ** 2)
        u2_ic_loss = jnp.mean((u2_pred - u2_0) ** 2)

        # Residual loss
        if self.config.weighting.use_causal == True:
            res_l1, res_l2, gamma = self.res_causal(params, batch)
            res_loss1 = jnp.mean(res_l1 * gamma)
            res_loss2 = jnp.mean(res_l2 * gamma)
        else:
            res_pred1,res_pred2 = self.r_pred_fn(
                params, batch[:, 0], batch[:, 1]
            )
            # Compute loss
            res_loss1 = jnp.mean(res_pred1 ** 2)
            res_loss2 = jnp.mean(res_pred2 ** 2)

        loss_dict = {
            "ic": (u1_ic_loss + u2_ic_loss)/(jnp.mean(u1_0 ** 2) + jnp.mean(u2_0 ** 2)),
            "res1": res_loss1,
            "res2": res_loss2,
            'rd': jnp.mean(self.res_right_dirichlet(params, batch[:, 0].reshape((batch.shape[0],1)), self.xu)),
            'ld': jnp.mean(self.res_left_dirichlet(params, batch[:, 0].reshape((batch.shape[0],1)), self.xl)),
            'ln': jnp.mean(self.res_neumann(params, batch[:, 0].reshape((batch.shape[0],1)), self.xl))
        }
        return loss_dict

    #Compute NTK
    @partial(jit, static_argnums = (0,))
    def compute_diag_ntk(self, params, batch):
        #Initial Condition
        u1_ic_ntk = vmap(
            vmap(ntk_fn, (None, None, None, 0)), (None, None, None, 0)
        )(self.u1_net, params, 0.0, batch[:, 1].reshape((batch.shape[0],1)))

        u2_ic_ntk = vmap(
            vmap(ntk_fn, (None, None, None, 0)), (None, None, None, 0)
        )(self.u2_net, params, 0.0, batch[:, 1].reshape((batch.shape[0],1)))

        #Right Dirichlet
        rd_ntk = vmap(
            vmap(ntk_fn, (None, None, 0, None)), (None, None, 0, None)
        )(self.res_right_dirichlet_nv, params, batch[:, 0].reshape((batch.shape[0],1)), self.xu)

        #Left Dirichlet
        ld_ntk = vmap(
            vmap(ntk_fn, (None, None, 0, None)), (None, None, 0, None)
        )(self.res_left_dirichlet_nv, params, batch[:, 0].reshape((batch.shape[0],1)), self.xl)

        #Left neumann
        ln_ntk = vmap(
            vmap(ntk_fn, (None, None, 0, None)), (None, None, 0, None)
        )(self.res_neumann_nv, params, batch[:, 0].reshape((batch.shape[0],1)), self.xl)

        # Consider the effect of causal weights
        if self.config.weighting.use_causal:
            batch = jnp.array([batch[:, 0].sort(), batch[:, 1]]).T
            res_ntk1 = vmap(ntk_fn, (None, None, 0, 0))(
                self.r_net1, params, batch[:, 0], batch[:, 1]
            )

            res_ntk2 = vmap(ntk_fn, (None, None, 0, 0))(
                self.r_net2, params, batch[:, 0], batch[:, 1]
            )

            res_ntk1 = res_ntk1.reshape(self.num_chunks, -1)
            res_ntk2 = res_ntk2.reshape(self.num_chunks, -1)

            res_ntk1 = jnp.mean(res_ntk1, axis=1)
            res_ntk2 = jnp.mean(res_ntk2, axis=1)

            _,_, casual_weights = self.res_causal(params, batch)
            res_ntk1 = res_ntk1 * casual_weights
            res_ntk2 = res_ntk2 * casual_weights
        else:
            res_ntk1 = vmap(ntk_fn, (None, None, 0, 0))(
                self.r_net1, params, batch[:, 0], batch[:, 1]
            )
            res_ntk2 = vmap(ntk_fn, (None, None, 0, 0))(
                self.r_net2, params, batch[:, 0], batch[:, 1]
            )

        ntk_dict = {
            "ic": u1_ic_ntk + u2_ic_ntk,
            "res1": res_ntk1,
            "res2": res_ntk2,
            'rd': rd_ntk,
            'ld': ld_ntk,
            'ln': ln_ntk
        }
        return ntk_dict
