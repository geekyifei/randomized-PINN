#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 17:55:28 2023

@author: yifeizong
"""

#Import dependencies
import jax 
import os
import jax.numpy as jnp
from jax import random, grad, vmap, jit
from jax.flatten_util import ravel_pytree
from jax.example_libraries import optimizers

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.linalg as spl
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

import itertools
import argparse
from functools import partial
from tqdm import trange
from time import perf_counter

#command line argument parser
# parser = argparse.ArgumentParser(description="1D nonLinear Poisson with randomized PINN")
# parser.add_argument(
#     "--rand_seed",
#     type=int,
#     default=8888,
#     help="random seed")
# parser.add_argument(
#     "--sigma",
#     type=float,
#     default=0.1,
#     help="Data noise level")
# parser.add_argument(
#     "--sigma_r",
#     type=float,
#     default=0.1,
#     help="Aleotoric uncertainty to the residual")
# parser.add_argument(
#     "--sigma_b",
#     type=float,
#     default=0.1,
#     help="Aleotoric uncertainty to the boundary data")
# parser.add_argument(
#     "--sigma_p",
#     type=float,
#     default=1,
#     help="Prior std")
# parser.add_argument(
#     "--Nres",
#     type=int,
#     default=512,
#     help="Number of reisudal points")
# parser.add_argument(
#     "--Nsamples",
#     type=int,
#     default=100,
#     help="Number of posterior samples")
# parser.add_argument(
#     "--nIter",
#     type=int,
#     default=8000,
#     help="Number of training epochs per realization")
# parser.add_argument(
#     "--data_load",
#     type=bool,
#     default=False,
#     help="If to load data")
# parser.add_argument(
#     "--method",
#     type=str,
#     default='rPINN',
#     help="Method for Bayesian training")
# parser.add_argument(
#     "--model_load",
#     type=bool,
#     default=False,
#     help="If to load existing samples")
# args = parser.parse_args()

#Define parameters
layers_u = [1, 50, 50, 1]
lbt = np.array([-0.7])
ubt = np.array([0.7]) 
lamb = 0.01
k = 0.7
rand_seed = 8888
Nres = 32
Nb = 2
sigma = 0.1
lambda_r = 20*2700/Nres
lambda_b = 1*2700/Nb
lambda_p = 1
gamma = sigma**2*lambda_r
Nsamples = 5000
nIter = 2000
method = 'DE'
model_load = False
num_print = 200
dataset = dict()

path_f   = f'1D_nonlinear_poisson_{method}_Nres_{Nres}_sigma_{sigma}_Nsamples_{Nsamples}_nIter_{nIter}'
path_fig = os.path.join(path_f,'figures')
if not os.path.exists(path_f):
    os.makedirs(path_f)
if not os.path.exists(path_fig):
    os.makedirs(path_fig)
f_rec = open(os.path.join(path_f,'record.out'), 'a+')

def u(x):
  return jnp.sin(6*x)**3

def f(x):
  return lamb*(-108*jnp.sin(6*x)**3 + 216*jnp.sin(6*x)*jnp.cos(6*x)**2) + k*jnp.tanh(u(x))

rl2e = lambda yest, yref : spl.norm(yest - yref, 2) / spl.norm(yref, 2) 
infe = lambda yest, yref : spl.norm(yest - yref, np.inf) 
lpp = lambda h, href, sigma: np.sum( -(h - href)**2/(2*sigma**2) - 1/2*np.log( 2*np.pi) - 2*np.log(sigma))

#create noisy boundary data
np.random.seed(rand_seed)
x_data = np.array([lbt[0], ubt[0]])[:,np.newaxis]
y_data = np.array([u(lbt[0]), u(ubt[0])])[:,np.newaxis].astype(np.float32) + np.random.normal(0,sigma,(2,1)).astype(np.float32)
data = jnp.concatenate([x_data,y_data], axis=1)
dataset.update({'data': data})

#create noisy forcing sampling
X_r = np.linspace(lbt[0], ubt[0], Nres)
X_r = jnp.sort(X_r, axis = 0)[:,np.newaxis]
y_r = f(X_r) + np.random.normal(0,sigma,(Nres,1)).astype(np.float32)
Dres = jnp.asarray(jnp.concatenate([X_r,y_r], axis=1))
dataset.update({'res': Dres})

# Define FNN  
def FNN(layers, activation=jnp.tanh):
   
    def init(prng_key): #return a list of (W,b) tuples
      def init_layer(key, d_in, d_out):
          key1, key2 = random.split(key)
          glorot_stddev = 1.0 / jnp.sqrt((d_in + d_out) / 2.)
          W = glorot_stddev * random.normal(key1, (d_in, d_out))
          b = jnp.zeros(d_out)
          return W, b
      key, *keys = random.split(prng_key, len(layers))
      params = list(map(init_layer, keys, layers[:-1], layers[1:]))
      return params

    def forward(params, inputs):
        Z = inputs
        for W, b in params[:-1]:
            outputs = jnp.dot(Z, W) + b
            Z = activation(outputs)
        W, b = params[-1]
        outputs = jnp.dot(Z, W) + b
        return outputs
    
    return init, forward

# Define the model
class PINN():
    def __init__(self, key, layers, dataset, lbt, ubt, lamb, k, sigma, lambda_r, lambda_b, lambda_p, gamma):  

      self.lbt = lbt #domain lower corner
      self.ubt = ubt #domain upper corner
      self.k = k
      self.lamb = lamb
      self.scale_coe = 0.5
      self.scale = 2 * self.scale_coe / (self.ubt - self.lbt)
      
      # Prepare normalized training data
      self.dataset = dataset
      self.X_res, self.y_res = self.normalize(dataset['res'][:,0:1]), dataset['res'][:,1:2]
      self.X_data, self.y_data = self.normalize(dataset['data'][:,0:1]), dataset['data'][:,1:2]
      
      # Initalize the network
      self.init, self.forward = FNN(layers, activation=jnp.tanh)
      self.params = self.init(key)
      raveled, self.unravel = ravel_pytree(self.params)
      self.num_params = raveled.shape[0]

      # Evaluate the state and the residual over the grid
      self.u_pred_map = vmap(self.predict_u, (None, 0)) 
      self.f_pred_map = vmap(self.predict_f, (None, 0))

      self.itercount = itertools.count()
      self.loss_log = []
      self.loss_likelihood_log = []
      self.loss_dbc_log = []
      self.loss_res_log = []
      
      # Optimizer
      lr = optimizers.exponential_decay(1e-3, decay_steps=5000, decay_rate=0.9)
      self.opt_init, \
      self.opt_update, \
      self.get_params = optimizers.adam(lr)
      self.opt_state = self.opt_init(self.params)
      
      self.lambda_r = lambda_r
      self.lambda_b = lambda_b
      self.lambda_p = lambda_p
      self.gamma = gamma
      self.sigma_p = jnp.sqrt(self.gamma)
      self.sigma_r = jnp.sqrt(self.gamma/self.lambda_r)
      self.sigma_b = jnp.sqrt(self.gamma/self.lambda_b)

      #Define random noise distributions
      # for residual term
      self.alpha_dist = tfd.Independent(tfd.Normal(loc= jnp.zeros_like(self.y_res.ravel()),
                                 scale= self.sigma_r*jnp.ones_like(self.y_res.ravel())), reinterpreted_batch_ndims = 1)
      # for boundary term
      self.beta_dist = tfd.Independent(tfd.Normal(loc= jnp.zeros_like(self.y_data.ravel()),
                                 scale= self.sigma_b*jnp.ones_like(self.y_data.ravel())), reinterpreted_batch_ndims = 1)
      # for regularization term
      self.omega_dist = tfd.Independent(tfd.Normal(loc= jnp.zeros((self.num_params,)),
                                 scale= self.sigma_p*jnp.ones((self.num_params,))), reinterpreted_batch_ndims = 1)
      

    # normalize inputs of DNN to [-0.5, 0.5]
    def normalize(self, X):
      if X.shape[1] == 1:
        return 2.0 * self.scale_coe * (X - self.lbt[0:1])/(self.ubt[0:1] - self.lbt[0:1]) - self.scale_coe
      if X.shape[1] == 2:
        return 2.0 * self.scale_coe * (X - self.lbt[0:2])/(self.ubt[0:2] - self.lbt[0:2]) - self.scale_coe
      if X.shape[1] == 3:
        return 2.0 * self.scale_coe * (X - self.lbt)/(self.ubt - self.lbt) - self.scale_coe

    @partial(jit, static_argnums=(0,))
    def u_net(self, params, x):
      inputs = jnp.hstack([x])
      outputs = self.forward(params, inputs)
      return outputs[0] 

    @partial(jit, static_argnums=(0,))
    def res_net(self, params, x): 
      u = self.u_net(params, x)
      u_xx = grad(grad(self.u_net, argnums=1), argnums=1)(params, x)*self.scale[0]**2
      return self.lamb*u_xx + self.k*jnp.tanh(u)  

    @partial(jit, static_argnums=(0,))
    def predict_u(self, params, x):
    # Normalize input first, and then predict
      x = 2.0 * self.scale_coe * (x - self.lbt[0])/(self.ubt[0] - self.lbt[0]) - self.scale_coe
      return self.u_net(params, x) 

    @partial(jit, static_argnums=(0,))
    def predict_f(self, params, x):
    # Normalize input first, and then predict
      x = 2.0 * self.scale_coe * (x - self.lbt[0])/(self.ubt[0] - self.lbt[0]) - self.scale_coe
      return self.res_net(params, x)  

    @partial(jit, static_argnums=(0,))
    def loss_dbc(self, params, beta):
      u_pred = vmap(self.u_net, (None, 0))(params, self.X_data[:,0])
      loss_bc = jnp.sum((u_pred.flatten() - self.y_data.flatten() -  beta)**2)
      return loss_bc

    @partial(jit, static_argnums=(0,))
    def loss_res(self, params, alpha): 
      f_pred = vmap(self.res_net, (None, 0))(params, self.X_res[:,0])
      loss_res = jnp.sum((f_pred.flatten() - self.y_res.flatten() - alpha)**2)
      return loss_res
  
    @partial(jit, static_argnums=(0,))
    def l2_regularizer(self, params, omega):
      return jnp.sum((ravel_pytree(params)[0] - omega)**2)

    @partial(jit, static_argnums=(0,))  
    def loss(self, params, alpha, beta, omega):
      return 1/self.sigma_r**2*self.loss_res(params, alpha) + 1/self.sigma_b**2*self.loss_dbc(params, beta) + \
                  1/self.sigma_p**2*self.l2_regularizer(params, omega) 

    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, alpha, beta, omega):
        params = self.get_params(opt_state)
        g = grad(self.loss, argnums=0)(params, alpha, beta, omega)

        return self.opt_update(i, g, opt_state)

    def train(self, nIter, num_print, alpha, beta, omega):
        pbar = trange(nIter)
        # Main training loop
        for it in pbar:
            self.current_count = next(self.itercount)
            self.opt_state = self.step(self.current_count, self.opt_state, alpha, beta, omega)
            
            if it % num_print == 0:
                params = self.get_params(self.opt_state)

                loss_value = self.loss(params, alpha, beta, omega)
                loss_res_value = self.loss_res(params, alpha)
                loss_dbc_value = self.loss_dbc(params, beta)
                loss_reg_value = self.l2_regularizer(params, omega)


                pbar.set_postfix({'Loss': loss_value, 
                          'Loss_res': loss_res_value,
                          'Loss_dbc': loss_dbc_value,
                          'Loss_reg': loss_reg_value})
        self.loss_log.append(loss_value)
        self.loss_likelihood_log.append(loss_res_value + loss_dbc_value)
        self.loss_res_log.append(loss_res_value)
        self.loss_dbc_log.append(loss_dbc_value)
    
    def de_sample(self, Nsample, nIter, num_print, key):
        #Using deep ensemble methods
        params_sample = []
        alpha = beta = omega = 0
    
        for it in range(Nsample):
            key, *keys = random.split(key, 2)
      
            lr = optimizers.exponential_decay(1e-3, decay_steps=5000, decay_rate=0.9)
            self.opt_init, \
            self.opt_update, \
            self.get_params = optimizers.adam(lr)
            self.params = self.init(keys[0])
            self.opt_state = self.opt_init(self.params)
            self.itercount = itertools.count()
            self.train(nIter, num_print, alpha, beta, omega)
      
            params = self.get_params(self.opt_state)
            params_sample.append(ravel_pytree(params)[0])
            print(f'{it}-th sample finished')
        return jnp.array(params_sample)
    
    def rpinn_sample(self, Nsample, nIter, num_print, key):
        #sample with randomized PINN
        params_sample = []
        alpha_sample = []
        beta_sample = []
        omega_sample = []
    
        for it in range(Nsample):
            key, *keys = random.split(key, 5)
            #key, *keys = random.split(key, 4)
            alpha = self.alpha_dist.sample(1, keys[0])[0]
            beta = self.beta_dist.sample(1, keys[1])[0]
            omega = self.omega_dist.sample(1, keys[2])[0]
      
            lr = optimizers.exponential_decay(1e-3, decay_steps=5000, decay_rate=0.9)
            self.opt_init, \
            self.opt_update, \
            self.get_params = optimizers.adam(lr)
            self.params = self.init(keys[3])
            self.opt_state = self.opt_init(self.params)
            self.itercount = itertools.count()
            self.train(nIter, num_print, alpha, beta, omega)
      
            params = self.get_params(self.opt_state)
            params_sample.append(ravel_pytree(params)[0])
            alpha_sample.append(alpha)
            beta_sample.append(beta)
            omega_sample.append(omega)
            print(f'{it}-th sample finished')
        return jnp.array(params_sample), jnp.array(alpha_sample),\
            jnp.array(beta_sample), jnp.array(omega_sample)
            
    # def metropolis_ratio(self, theta_old, theta_new, alpha_old, alpha_new):
    #     def fn(theta, alpha):
    #         res = self.r_pred_vector(theta)
    #         delta = res - alpha
    #         prod = jnp.einsum('i,i->', res, delta)
    #         return prod
        
    #     jac_old  = jax.jacfwd(self.r_pred_vector, argnums = 0)(theta_old) #jacobian dr/dtheta (Nres, num_params)
    #     jac_new  = jax.jacfwd(self.r_pred_vector, argnums = 0)(theta_new) #jacobian dr/dtheta
    #     hess_old = jax.hessian(fn, argnums = 0)(theta_old, alpha_old) #hessian
    #     hess_new = jax.hessian(fn, argnums = 0)(theta_new, alpha_new)
        
    #     #logdet_old = np.linalg.slogdet(sigma_r**2*Sigma + hess.T +  jnp.einsum('ik,kj->ij', jac.T, jac))[1]
    #     #logdet_new = np.linalg.slogdet(sigma_r**2*Sigma + hess.T +  jnp.einsum('ik,kj->ij', jac.T, jac))[1]
    #     #ratio = np.sqrt(np.exp(logdet_new - logdet_old))
        
    #     det_old = np.linalg.det(sigma_r**2*self.Sigma + hess_old.T +  jnp.einsum('ik,kj->ij', jac_old.T, jac_old))
    #     det_new = np.linalg.det(sigma_r**2*self.Sigma + hess_new.T +  jnp.einsum('ik,kj->ij', jac_new.T, jac_new))
    #     ratio = np.sqrt(np.math.abs(det_new))/np.math.sqrt(np.math.abs(det_old))
        
    #    return ratio
  
    # def rpinn_sample_metro(self, Nsample, nIter, num_print, key):
    #     params_sample = []
    #     alpha_sample = []
    #     beta_sample = []
    #     omega_sample = []

    #     for it in range(Nsample):
    #         key, *keys = random.split(key, 4)
    #         alpha = self.alpha_dist.sample(1, keys[0])[0]
    #         beta = self.beta_dist.sample(1, keys[1])[0]
    #         omega = self.omega_dist.sample(1, keys[2])[0]
      
    #         lr = optimizers.exponential_decay(1e-3, decay_steps=5000, decay_rate=0.9)
    #         self.opt_init, \
    #         self.opt_update, \
    #         self.get_params = optimizers.adam(lr)
    #         #self.params = self.init(keys[3])
    #         self.opt_state = self.opt_init(self.params)
    #         self.itercount = itertools.count()
    #         self.train(nIter, num_print, alpha, beta, omega)
      
    #         theta_new = ravel_pytree(self.get_params(self.opt_state))[0]
    #         ratio = np.abs()
    #         u = np.random.uniform(low=0.0, high=1.0, size=())
    #         params_sample.append()
    #         alpha_sample.append(alpha)
    #         beta_sample.append(beta)
    #         omega_sample.append(omega)
    #         print(f'{it}-th sample finished')
    #     return jnp.array(params_sample), jnp.array(alpha_sample),\
    #         jnp.array(beta_sample), jnp.array(omega_sample)
    
    
key = random.PRNGKey(rand_seed)
key, subkey = random.split(key, 2)
model = PINN(key, layers_u, dataset, lbt, ubt, lamb, k, sigma, lambda_r, lambda_b, lambda_p, gamma)
if model_load == False:
    if method == 'rPINN':
        
        ts = perf_counter()
        samples, alpha_ens, beta_ens, omega_ens = model.rpinn_sample(Nsamples, nIter = nIter, 
                                                              num_print = num_print, key = subkey)
        timings = perf_counter() - ts
        print(f"rPINN: {timings} s")
        print(f"rPINN: {timings} s", file = f_rec)
        
    elif method == 'rPINN_metro':
        
        ts = perf_counter()
        samples, alpha_ens, beta_ens, omega_ens = model.rpinn_sample_metro(Nsamples, nIter = nIter, 
                                                              num_print = num_print, key = subkey)
        timings = perf_counter() - ts
        print(f"rPINN-metro: {timings} s")
        print(f"rPINN-metro: {timings} s", file = f_rec)
        
    elif method == 'DE':
        
        ts = perf_counter()
        samples = model.de_sample(Nsamples, nIter = nIter, num_print = num_print, key = subkey)
        timings = perf_counter() - ts
        print(f"Deep ensemble: {timings} s")
        print(f"Deep ensemble: {timings} s", file = f_rec)
        
    else:
        samples, omega_ens = model.rms_sample(Nsamples, nIter = nIter, num_print = num_print, key = subkey)
    
    np.savetxt(os.path.join(path_f,f'{method}_posterior_samples_Nres_{Nres}_sigma_{sigma}_Nsamples_{Nsamples}_nIter_{nIter}.out'), samples)
else:
    samples = np.loadtxt(os.path.join(path_f,f'{method}_posterior_samples_Nres_{Nres}_sigma_{sigma}__Nsamples_{Nsamples}_nIter_{nIter}.out'))

Npred = 201
x_pred_index = jnp.linspace(-0.7,0.7,Npred)
u_ref = u(x_pred_index)
f_ref = f(x_pred_index)

@jit
def get_u_pred(sample):
  return model.u_pred_map(model.unravel(sample),x_pred_index)

@jit
def get_f_pred(sample):
  return model.f_pred_map(model.unravel(sample),x_pred_index)

u_pred_ens = vmap(get_u_pred)(samples)
f_pred_ens = vmap(get_f_pred)(samples)
np.savetxt(os.path.join(path_f,'u_pred_ens.out'), u_pred_ens)
np.savetxt(os.path.join(path_f,'f_pred_ens.out'), f_pred_ens)

u_pred_ens_mean = np.mean(u_pred_ens, axis = 0)
u_pred_ens_std = np.std(u_pred_ens, axis = 0)
f_pred_ens_mean = np.mean(f_pred_ens, axis = 0)
f_pred_ens_std = np.std(f_pred_ens, axis = 0)

u_env = np.logical_and( (u_pred_ens_mean < u_ref + 2*u_pred_ens_std), (u_pred_ens_mean > u_ref - 2*u_pred_ens_std) )
f_env = np.logical_and( (f_pred_ens_mean < f_ref + 2*f_pred_ens_std), (f_pred_ens_mean > f_ref - 2*f_pred_ens_std) )

# =============================================================================
# Posterior Statistics
# =============================================================================

rl2e_u = rl2e(u_pred_ens_mean, u_ref)
infe_u = infe(u_pred_ens_mean, u_ref)
lpp_u = lpp(u_pred_ens_mean, u_ref, u_pred_ens_std)
rl2e_f = rl2e(f_pred_ens_mean, f_ref)
infe_f = infe(f_pred_ens_mean, f_ref)
lpp_f = lpp(f_pred_ens_mean, f_ref, f_pred_ens_std)

print('u prediction:\n')
print('Relative RL2 error: {}'.format(rl2e_u))
print('Absolute inf error: {}'.format(infe_u))
print('Average standard deviation: {}'.format(np.mean(u_pred_ens_std)))
print('log predictive probability: {}'.format(lpp_u))
print('Percentage of coverage:{}\n'.format(np.sum(u_env)/Npred))

print('f prediction:\n')
print('Relative RL2 error: {}'.format(rl2e_f))
print('Absolute inf error: {}'.format(infe_f))
print('Average standard deviation: {}'.format(np.mean(f_pred_ens_std)))
print('log predictive probability: {}'.format(lpp_f))
print('Percentage of coverage:{}\n'.format(np.sum(f_env)/Npred))

print('u prediction:\n', file = f_rec)
print('Relative RL2 error: {}'.format(rl2e_u), file = f_rec)
print('Absolute inf error: {}'.format(infe_u), file = f_rec)
print('Average standard deviation: {}'.format(np.mean(u_pred_ens_std)), file = f_rec)
print('log predictive probability: {}'.format(lpp_u), file = f_rec)
print('Percentage of coverage:{}\n'.format(np.sum(u_env)/Npred), file = f_rec)

print('f prediction:\n', file = f_rec)
print('Relative RL2 error: {}'.format(rl2e_f), file = f_rec)
print('Absolute inf error: {}'.format(infe_f), file = f_rec)
print('Average standard deviation: {}'.format(np.mean(f_pred_ens_std)), file = f_rec)
print('log predictive probability: {}'.format(lpp_f), file = f_rec)
print('Percentage of coverage:{}\n'.format(np.sum(f_env)/Npred), file = f_rec)


# =============================================================================
# Plot posterior predictions
# =============================================================================

fig, ax = plt.subplots(dpi = 300, figsize = (4,4))
ax.plot(x_pred_index, u_ref, 'k-', label='Exact')
# for i in range(1, 5000, 500):
#     ax.plot(x_pred_index, get_u_pred(samples[i]))
ax.plot(x_pred_index, u_pred_ens_mean, markersize = 1, markevery=2, markerfacecolor='None', label= f'{method} mean', alpha = 0.8)
ax.fill_between(x_pred_index, u_pred_ens_mean + 2 * u_pred_ens_std, u_pred_ens_mean - 2 * u_pred_ens_std,
                alpha = 0.3, label = r'$95 \% $ CI')
ax.scatter(data[:,0], data[:,1], label='Obs' , s = 20, facecolors='none', edgecolors='b')
ax.set_xlabel('$x$', fontsize=16)
ax.set_ylabel('$u(x)$', fontsize=16)
ax.set_xlim(-0.72,0.72)
ax.set_ylim(-2.5,2.5)
ax.tick_params(axis='both', which = 'major', labelsize=13)
ax.legend(fontsize=10)
fig.tight_layout()
plt.savefig(os.path.join(path_fig,f'1D_nonlinear_poisson_{method}_Nres_{Nres}_sigma_{sigma}_Nsamples_{Nsamples}_nIter_{nIter}_upred.png'))
plt.show()

fig, ax = plt.subplots(dpi = 300, figsize = (4,4))
r_ref = f(x_pred_index)
ax.plot(x_pred_index, r_ref, 'k-', label='Exact')
# for i in range(1, 5000, 500):
#     ax.plot(x_pred_index, get_r_pred(samples[i]))
ax.plot(x_pred_index, f_pred_ens_mean, markersize = 1, markevery=2, markerfacecolor='None', label=f'{method} mean', alpha = 0.8)
ax.fill_between(x_pred_index, f_pred_ens_mean + 2 * f_pred_ens_std, f_pred_ens_mean - 2 * f_pred_ens_std,
                alpha = 0.3, label = r'$95 \% $ CI')
ax.scatter(Dres[:,0], Dres[:,1], label='Obs' , s = 20, facecolors='none', edgecolors='b')
ax.set_xlabel('$x$', fontsize=16)
ax.set_xlim(-0.72,0.72)
ax.set_ylim(-2.5,2.5)
ax.set_ylabel('$f(x)$', fontsize=16)
ax.tick_params(axis='both', which = 'major', labelsize=13)
ax.legend(fontsize=10, loc= 'upper left')
fig.tight_layout()
plt.savefig(os.path.join(path_fig,f'1D_nonlinear_poisson_{method}_Nres_{Nres}_sigma_{sigma}_Nsamples_{Nsamples}_nIter_{nIter}_fpred.png'))
plt.show()


f_rec.close()

# idx0, idx1, idx2 = np.argsort(theta_mean_hmc)[-3:]
# theta0_hmc, theta1_hmc, theta2_hmc = hmc_samples[:,idx0], hmc_samples[:,idx1], hmc_samples[:,idx2]
# theta0_rpinn, theta1_rpinn, theta2_rpinn = samples[:,idx0], samples[:,idx1], samples[:,idx2]
# df1 = pd.dataFrame({'Method': 'HMC', r'$\theta_0$': theta0_hmc, r'$\theta_1$': theta1_hmc, r'$\xi_2$': theta2_hmc})
# df2 = pd.dataFrame({'Method': 'rPICKLE', r'$\theta_0$': theta0_rpinn, r'$\theta_1$':theta1_rpinn, r'$\xi_2$': theta2_rpinn})
# df = pd.concat([df1, df2], ignore_index=True)
# # g = sns.jointplot(data=df, x='xi_0', y='xi_1',  hue='method', joint_kws={'alpha': 0.5}, kind = 'kde')
# # g.ax_joint.tick_params(labelsize=16)
# # g.ax_joint.set_xlabel(r"$\xi_0$", fontsize=24)
# # g.ax_joint.set_ylabel(r"$\xi_1$", fontsize=24)

# plt.figure(figsize=(4,4))
# g = sns.PairGrid(df, hue='Method')
# g.map_diag(sns.histplot)
# g.map_offdiag(sns.kdeplot)
# g.add_legend(fontsize=18)
# plt.gcf().set_dpi(600)
# g.tight_layout()
# plt.show()


# fig, ax = plt.subplots(dpi = 300, figsize = (4,4))
# r_ref = f(x_pred_index)
# ax.plot(x_pred_index, r_ref, 'k-', label='Exact')
# # for i in range(1, 5000, 500):
# #     ax.plot(x_pred_index, get_r_pred(samples[i]))
# ax.plot(x_pred_index, r_pred_ens_mean_hmc, markersize = 1, markevery=2, markerfacecolor='None', label=r'HMC mean', alpha = 0.8)
# ax.fill_between(x_pred_index, r_pred_ens_mean_hmc + 2 * r_pred_ens_std_hmc, r_pred_ens_mean_hmc - 2 * r_pred_ens_std_hmc,
#                 alpha = 0.3, label = r'$95 \% $ CI')
# #ax.scatter(Dres[:,0], Dres[:,1], label='Obs' , s = 20, facecolors='none', edgecolors='b')
# ax.set_xlabel('$X$', fontsize=16)
# ax.set_xlim(-1.02,1.02)
# ax.set_ylim(-1.2,1.2)
# ax.set_ylabel('$r(x)$', fontsize=16)
# ax.tick_params(axis='both', which = 'major', labelsize=13)
# ax.legend(fontsize=10, loc= 'upper left')
# #plt.savefig(os.path.join(path_fig,'r_pred.png'))
# plt.show()


#Hessian specification

# hess_fn = jax.hessian(target_log_prob_fn)
# hessian0 = hess(np.mean(sample0, axis = 0))
# _, s0, _ = jax.scipy.linalg.svd(jax.scipy.linalg.inv(hessian0))
# fig, ax = plt.subplots(dpi = 300, figsize = (4,4))
# ax.plot(s[i], linestyle = linestyle[i], marker = mark, markersize = 2, markevery= 100, markerfacecolor='None', label=f'chain{i+1}', alpha = 0.8)
# ax.set_xlabel('Index', fontsize=16)
# ax.set_ylabel('Eigenvalues', fontsize=16)
# plt.yscale('log')
# ax.tick_params(axis='both', which = 'major', labelsize=13)
# ax.legend(fontsize=8)
# plt.show()

# =============================================================================
#  Loss landscape
# =============================================================================


# def normalize_weights(weights, origin):
#     return weights* jnp.abs(origin)/ jnp.abs(weights)

# class RandomCoordinates(object):
#     def __init__(self, origin):
#         self.origin = origin # (num_params,)
#         self.v0 = normalize_weights(
#             jax.random.normal(key = random.PRNGKey(88), shape = self.origin.shape), 
#             origin)
#         self.v1 = normalize_weights(
#             jax.random.normal(key = random.PRNGKey(66), shape = self.origin.shape), 
#             origin)

#     def __call__(self, a, b):
#         return a*self.v0 + b*self.v1 + self.origin


# class LossSurface(object):
#     def __init__(self, loss_fn, coords):
#         self.loss_fn = loss_fn
#         self.coords = coords

#     def compile(self, range, num_points):
#         loss_fn_0d = lambda x, y: self.loss_fn(self.coords(x,y))
#         loss_fn_1d = jax.vmap(loss_fn_0d, in_axes = (0,0), out_axes = 0)
#         loss_fn_2d = jax.vmap(loss_fn_1d, in_axes = (0,0), out_axes = 0)
        
#         self.a_grid = jnp.linspace(-1.0, 1.0, num=num_points) ** 3 * range
#         self.b_grid = jnp.linspace(-1.0, 1.0, num=num_points) ** 3 * range
#         self.aa, self.bb = jnp.meshgrid(self.a_grid, self.b_grid)
#         self.loss_grid = loss_fn_2d(self.aa, self.bb)
        
#     def project_points(self, points):
#         x = jax.vmap(lambda x: jnp.dot(x, self.coords.v0)/jnp.linalg.norm(self.coords.v0), 0, 0)(points)
#         y = jax.vmap(lambda y: jnp.dot(y, self.coords.v1)/jnp.linalg.norm(self.coords.v1), 0, 0)(points)
#         return x, y
        
#     def plot(self, levels=30, points  = None, ax=None, **kwargs):
#         xs = self.a_grid
#         ys = self.b_grid
#         zs = self.loss_grid
#         if ax is None:
#             fig, ax = plt.subplots(dpi = 600, **kwargs)
#             ax.set_title("Loss Surface")
#             ax.set_aspect("equal")
            
#         # Set Levels
#         min_loss = zs.min()
#         max_loss = zs.max()
#         # levels = jnp.exp(
#         #     jnp.linspace(
#         #         jnp.log(min_loss), jnp.log(max_loss), num=levels
#         #     )
#         # )
#         levels = jnp.exp(
#                 jnp.log(min_loss) + 
#                 jnp.linspace(0., 1.0, num=levels) ** 3 * (jnp.log(max_loss))- jnp.log(min_loss))
#         # Create Contour Plot
#         CS = ax.contour(
#             xs,
#             ys,
#             zs,
#             levels=levels,
#             cmap= 'magma',
#             linewidths=0.75,
#             norm = mpl.colors.LogNorm(vmin=min_loss, vmax=max_loss * 2.0),
#         )
#         point_x, point_y = self.project_points(points)
#         origin_x, origin_y = self.project_points(self.coords.origin)
#         ax.scatter(point_x, point_y, s = 0.25, c = 'g', marker = 'o')
#         ax.scatter(origin_x, origin_y, s = 1, c = 'r', marker = 'x')
#         ax.clabel(CS, fontsize=8, fmt="%1.2f")
#         #plt.colorbar(CS)
#         plt.show()
#         return ax

# theta_prior_m = jnp.zeros((pinn.num_params,))
# theta_prior_std = jnp.ones((pinn.num_params,))
# prior_dist = tfd.Independent(tfd.Normal(loc= theta_prior_m, scale= theta_prior_std),
#                   reinterpreted_batch_ndims= 1)

# def target_log_prob_fn(theta):
#     prior = prior_dist.log_prob(theta)
#     r_likelihood = jnp.sum( -jnp.log(sigma_r) - jnp.log(2*jnp.pi)/2 -(y_r.ravel() - pinn.r_pred_vector(theta))**2/(2*sigma_r**2))
#     u_likelihood = jnp.sum( -jnp.log(sigma_b) - jnp.log(2*jnp.pi)/2 -(y_data.ravel() - pinn.u_pred_vector(theta))**2/(2*sigma_b**2))
#     return -(prior + r_likelihood + u_likelihood)

# # def pinn_loss_fn(theta):
# #     prior = 1/2*jnp.linalg.norm(theta)**2
# #     r_likelihood = jnp.sum( (y_r.ravel() - pinn.r_pred_vector(theta))**2/(2*sigma_r**2))
# #     u_likelihood = jnp.sum( (y_data.ravel() - pinn.u_pred_vector(theta))**2/(2*sigma_b**2))
# #     return prior + r_likelihood + u_likelihood

# optim_params = pinn.deterministic_run(random.PRNGKey(999), 100000, 200)
# coords = RandomCoordinates(optim_params)
# loss_surface = LossSurface(target_log_prob_fn, coords)
# loss_surface.compile(range = 5, num_points= 500)
# ax = loss_surface.plot(levels = 30, points = hmc_samples)


