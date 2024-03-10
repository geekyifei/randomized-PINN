#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 11:31:38 2023

@author: yifeizong
"""

import jax 
import os
import jax.numpy as jnp
from jax import random, grad, vmap, jit
from jax.flatten_util import ravel_pytree
from jax.example_libraries import optimizers

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.linalg as spl
import seaborn as sns
import pandas as pd
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

import itertools
import argparse
from functools import partial
from tqdm import trange
from time import perf_counter

#command line argument parser
parser = argparse.ArgumentParser(description="1D nonLinear Poisson with HMC")
parser.add_argument(
    "--rand_seed",
    type=int,
    default=42,
    help="random seed")
parser.add_argument(
    "--sigma",
    type=float,
    default=0.1,
    help="Data uncertainty")
parser.add_argument(
    "--sigma_r",
    type=float,
    default=0.1,
    help="Aleotoric uncertainty to the residual")
parser.add_argument(
    "--sigma_d",
    type=float,
    default=0.04,
    help="Aleotoric uncertainty to the data")
parser.add_argument(
    "--sigma_p",
    type=float,
    default=1.452,
    help="Prior std")
parser.add_argument(
    "--Nres",
    type=int,
    default=128,
    help="Number of reisudal points")
parser.add_argument(
    "--Nsamples",
    type=int,
    default=5000,
    help="Number of Posterior samples")
parser.add_argument(
    "--Nburn",
    type=int,
    default=100000,
    help="Number of Posterior samples")
parser.add_argument(
    "--data_load",
    type=bool,
    default=False,
    help="If to load data")
args = parser.parse_args()

#Define parameters
layers_u = [1, 50, 50, 1]
lbt = np.array([-0.7])
ubt = np.array([0.7]) 
lamb = 0.01
k = 0.7
dataset = dict()
rand_seed = args.rand_seed
Nres = args.Nres
sigma = args.sigma
sigma_r = args.sigma_r
sigma_b = args.sigma_d
sigma_p = args.sigma_p
Nsamples = args.Nsamples
Nburn = args.Nburn
Nchains = 9
dataset = dict()
path_f   = f'1D_nonlinear_poisson_HMC_Nres_{Nres}_sigma_{sigma}_Nsamples_{Nsamples}_Nburn_{Nburn}'
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

# Create synthetic noisy data
#create noisy boundary data
np.random.seed(rand_seed)
x_data = np.array([lbt[0], ubt[0]])[:,np.newaxis]
y_data = np.array([u(lbt[0]), u(ubt[0])])[:,np.newaxis].astype(np.float32) + np.random.normal(0,sigma,(2,1)).astype(np.float32)
data = jnp.concatenate([x_data,y_data], axis=1)
dataset.update({'data': data})

#create noisy forcing sampling
X_r = np.linspace(lbt[0], ubt[0], Nres)
X_r = jnp.sort(X_r, axis = 0)[:,np.newaxis]
y_r = f(X_r) + np.random.normal(0,sigma,(Nres,1))
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
    def __init__(self, key, layers, dataset, lbt, ubt, lamb, k, sigma_r, sigma_b, sigma_p):  

      self.lbt = lbt #domain lower corner
      self.ubt = ubt #domain upper corner
      self.k = k
      self.lamb = lamb
      self.sigma_r = sigma_r
      self.sigma_b = sigma_b
      self.sigma_p = sigma_p

      # Prepare normalized training data
      self.dataset = dataset
      self.X_res, self.y_res = dataset['res'][:,0:1], dataset['res'][:,1:2]
      self.X_data, self.y_data = dataset['data'][:,0:1], dataset['data'][:,1:2]
      
      # Initalize the network
      self.init, self.forward = FNN(layers, activation=jnp.tanh)
      self.params = self.init(key)
      _, self.unravel = ravel_pytree(self.params)
      self.num_params = ravel_pytree(self.params)[0].shape[0]

      # Evaluate the network and the residual over the grid
      self.u_pred_map = vmap(self.predict_u, (None, 0)) 
      self.f_pred_map = vmap(self.predict_f, (None, 0))

      self.itercount = itertools.count()
      self.loss_log = []
      self.loss_likelihood_log = []
      self.loss_dbc_log = []
      self.loss_res_log = []

      lr = optimizers.exponential_decay(1e-3, decay_steps=5000, decay_rate=0.9)
      self.opt_init, \
      self.opt_update, \
      self.get_params = optimizers.adam(lr)
      self.opt_state = self.opt_init(self.params)

    @partial(jit, static_argnums=(0,))
    def u_net(self, params, x):
      inputs = jnp.hstack([x])
      outputs = self.forward(params, inputs)
      return outputs[0] 

    @partial(jit, static_argnums=(0,))
    def res_net(self, params, x): 
      u = self.u_net(params, x)
      u_xx = grad(grad(self.u_net, argnums=1), argnums=1)(params, x)
      return self.lamb*u_xx + self.k*jnp.tanh(u)  

    @partial(jit, static_argnums=(0,))
    def predict_u(self, params, x):
      return self.u_net(params, x) 

    @partial(jit, static_argnums=(0,))
    def predict_f(self, params, x):
      return self.res_net(params, x) 

    @partial(jit, static_argnums=(0,))
    def u_pred_vector(self, params):
      u_pred_vec = vmap(self.u_net, (None, 0))(self.unravel(params), self.X_data[:,0])
      return u_pred_vec

    @partial(jit, static_argnums=(0,))
    def f_pred_vector(self, params): 
      f_pred_vec = vmap(self.res_net, (None, 0))(self.unravel(params), self.X_res[:,0])
      return f_pred_vec  


key1, key2 = random.split(random.PRNGKey(0), 2)
pinn = PINN(key2, layers_u, dataset, lbt, ubt, lamb, k, sigma_r, sigma_b, sigma_p)
num_params = ravel_pytree(pinn.params)[0].shape[0]

def target_log_prob_fn(theta):
  prior = -1/(2*sigma_p**2)*jnp.sum( theta**2 )
  r_likelihood = -1/(2*sigma_r**2)*jnp.sum( (y_r.ravel() - pinn.f_pred_vector(theta))**2 )
  u_likelihood = -1/(2*sigma_b**2)*jnp.sum( (y_data.ravel() - pinn.u_pred_vector(theta))**2 )
  return prior + r_likelihood + u_likelihood

new_key, *subkeys = random.split(key1, Nchains + 1)
init_state = jnp.zeros((1, num_params))
for key in subkeys[:-1]:
    init_state = jnp.concatenate([init_state,random.normal(key ,(1, num_params))], axis=0)

# key3, key4, key5 = random.split(key1, 3)
# init_state = jnp.zeros((1, num_params))
# init_state = jnp.concatenate([init_state,random.normal(key4 ,(1, num_params))], axis=0)
# init_state = jnp.concatenate([init_state,3 + random.normal(key5 ,(1, num_params))], axis=0)

nuts_kernel = tfp.mcmc.NoUTurnSampler(
  target_log_prob_fn = target_log_prob_fn, step_size = 0.0005, max_tree_depth=10, max_energy_diff=1000.0,
  unrolled_leapfrog_steps=1, parallel_iterations=30)

kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
    inner_kernel=nuts_kernel, num_adaptation_steps=int(Nburn * 0.75))

def run_chain(init_state, key):
  samples, trace = tfp.mcmc.sample_chain(
      num_results= Nsamples,
      num_burnin_steps= Nburn,
      current_state= init_state,
      kernel= kernel,
      seed=key,
      trace_fn= lambda _,pkr: [pkr.inner_results.log_accept_ratio,
                    pkr.inner_results.target_log_prob,
                    pkr.inner_results.step_size]
    )
  return samples, trace

# nuts_kernel = tfp.mcmc.NoUTurnSampler(
#   target_log_prob_fn = target_log_prob_fn, step_size = 0.0001, max_tree_depth=10, max_energy_diff=1000.0,
#   unrolled_leapfrog_steps=1, parallel_iterations=30)

# def run_chain(init_state, key):
#   samples, trace = tfp.mcmc.sample_chain(
#       num_results= Nsamples,
#       num_burnin_steps= Nburn,
#       current_state= init_state,
#       kernel= nuts_kernel,
#       seed=key,
#       trace_fn= lambda _,pkr: [pkr.log_accept_ratio,
#                     pkr.target_log_prob,
#                     pkr.step_size]
#     )
#   return samples, trace

ts = perf_counter()
print('\nStart HMC Sampling')
#states, trace = jit(run_chain)(init_state, key3)
states, trace = jit(vmap(run_chain, in_axes=(0, None)))(init_state, new_key)
print('\nFinish HMC Sampling')
timings = perf_counter() - ts
print(f"HMC: {timings} s")
np.save(os.path.join(path_f,'chains'), states)

# =============================================================================
# Post-processing HMC results
# =============================================================================

accept_ratio = np.exp(trace[0])
target_log_prob = trace[1]
step_size = trace[2]
print(f'Average accept ratio for each chain: {np.mean(accept_ratio, axis = 1)}')
print(f'Average step size for each chain: {np.mean(step_size, axis = 1)}')
print(f'Average accept ratio for each chain: {np.mean(accept_ratio, axis = 1)}', file = f_rec)
print(f'Average step size for each chain: {np.mean(step_size, axis = 1)}', file = f_rec)

mark = [None, 'o', None]
linestyle = ['solid', 'dotted', 'dashed']
fig, ax = plt.subplots(dpi = 300, figsize = (4,4))
for i, mark in enumerate(mark):
  ax.plot(np.arange(Nsamples)[::10], target_log_prob[i,::10], marker = mark, markersize = 2, markevery= 100, markerfacecolor='None', linestyle = 'dashed', label = f'chain {i + 1}', alpha = 0.8)
ax.set_xlabel('Sample index', fontsize = 15)
ax.set_ylabel('Negative log prob', fontsize = 15)
ax.tick_params(axis='both', which = 'major', labelsize=12)
ax.set_xlim(0,Nsamples)
ax.legend(fontsize=6)
plt.savefig(os.path.join(path_fig,'target_log_prob.png'))
plt.show()

Npred = 201
x_pred_index = jnp.linspace(-0.7,0.7,Npred)
f_ref = f(x_pred_index)
u_ref = u(x_pred_index)
samples = states #(3, 10000, 2701)
samples = states

@jit
def get_u_pred(sample):
  return pinn.u_pred_map(pinn.unravel(sample),x_pred_index)

@jit
def get_f_pred(sample):
  return pinn.f_pred_map(pinn.unravel(sample),x_pred_index)

u_pred_ens = np.array([vmap(get_u_pred)(samples[i,:,:]) for i in range(samples.shape[0])]) # (3, 10000, 201)
f_pred_ens = np.array([vmap(get_f_pred)(samples[i,:,:]) for i in range(samples.shape[0])]) 

u_pred_ens_mean = np.mean(u_pred_ens, axis = 1) #(3, 201)
u_pred_ens_std = np.std(u_pred_ens, axis = 1)
f_pred_ens_mean = np.mean(f_pred_ens, axis = 1)
f_pred_ens_std = np.std(f_pred_ens, axis = 1)

u_env = np.logical_and( (u_pred_ens_mean < u_ref + 2*u_pred_ens_std), (u_pred_ens_mean > u_ref - 2*u_pred_ens_std) )
f_env = np.logical_and( (f_pred_ens_mean < f_ref + 2*f_pred_ens_std), (f_pred_ens_mean > f_ref - 2*f_pred_ens_std) )

# =============================================================================
# Posterior Statistics
# =============================================================================

for i in range(Nchains):
    rl2e_u = rl2e(u_pred_ens_mean[i, :], u_ref)
    infe_u = infe(u_pred_ens_mean[i, :], u_ref)
    lpp_u = lpp(u_pred_ens_mean[i, :], u_ref, u_pred_ens_std[i, :])
    rl2e_f = rl2e(f_pred_ens_mean[i, :], f_ref)
    infe_f = infe(f_pred_ens_mean[i, :], f_ref)
    lpp_f = lpp(f_pred_ens_mean[i, :], f_ref, f_pred_ens_std[i, :])
    
    print(f'chain {i}:\n')
    print('u prediction:\n')
    print('Relative RL2 error: {}'.format(rl2e_u))
    print('Absolute inf error: {}'.format(infe_u))
    print('Average standard deviation: {}'.format(np.mean(u_pred_ens_std[i, :])))
    print('log predictive probability: {}'.format(lpp_u))
    print('Percentage of coverage:{}\n'.format(np.sum(u_env[i, :])/Npred))
    
    print('f prediction:\n')
    print('Relative RL2 error: {}'.format(rl2e_f))
    print('Absolute inf error: {}'.format(infe_f))
    print('Average standard deviation: {}'.format(np.mean(f_pred_ens_std[i, :])))
    print('log predictive probability: {}'.format(lpp_f))
    print('Percentage of coverage:{}\n'.format(np.sum(f_env[i, :])/Npred))
    
    print(f'chain {i}:\n', file = f_rec)
    print('u prediction:\n', file = f_rec)
    print('Relative RL2 error: {}'.format(rl2e_u), file = f_rec)
    print('Absolute inf error: {}'.format(infe_u), file = f_rec)
    print('Average standard deviation: {}'.format(np.mean(u_pred_ens_std[i, :])), file = f_rec)
    print('log predictive probability: {}'.format(lpp_u), file = f_rec)
    print('Percentage of coverage:{}\n'.format(np.sum(u_env[i, :])/Npred), file = f_rec)
    
    print('f prediction:\n', file = f_rec)
    print('Relative RL2 error: {}'.format(rl2e_f), file = f_rec)
    print('Absolute inf error: {}'.format(infe_f), file = f_rec)
    print('Average standard deviation: {}'.format(np.mean(f_pred_ens_std[i, :])), file = f_rec)
    print('log predictive probability: {}'.format(lpp_f), file = f_rec)
    print('Percentage of coverage:{}\n'.format(np.sum(f_env[i, :])/Npred), file = f_rec)

# fig, ax = plt.subplots(dpi = 300, figsize = (4,4))
# ax.plot(x_pred_index, u_ref, 'k-', label='Exact')
# mark = [None, None, None]
# linestyle = ['solid', 'dotted', 'dashed']
# color = ['#2ca02c', '#ff7f0e', '#1f77b4'] # green , orange, blue
# for i, c in enumerate(color):
#     ax.plot(x_pred_index, u_pred_ens_mean[i, :], color = c, markersize = 1, markevery=2, markerfacecolor='None', label=f'Chain {i}', alpha = 0.8)
#     #ax.scatter(x_pred_index, u_pred_ens_mean[i], marker = mark, label=f'Pred mean - chain{i+1}' , s = 15)
#     ax.fill_between(x_pred_index, u_pred_ens_mean[i,:] + 2 * u_pred_ens_std[i,:], u_pred_ens_mean[i,:] - 2 * u_pred_ens_std[i,:], alpha = 0.3)
#                     #alpha = 0.3, label = f'$95\%$ CI')
# ax.scatter(x_data, y_data, label='Obs' , s = 20, facecolors='none', edgecolors='b')
# ax.set_xlabel('$x$', fontsize=16)
# ax.set_ylabel('$u(x)$', fontsize=16)
# ax.set_xlim(-0.72,0.72)
# ax.set_ylim(-2.5,2.5)
# ax.tick_params(axis='both', which = 'major', labelsize=13)
# ax.legend(fontsize=10, loc = 'upper left')
# fig.tight_layout()
# plt.show()
# plt.savefig(os.path.join(path_fig,'u_pred.png'))


# fig, ax = plt.subplots(dpi = 300, figsize = (4,4))
# ax.plot(x_pred_index, f_ref, 'k-', label='Exact')
# color = ['#2ca02c', '#ff7f0e', '#1f77b4'] # green , orange, blue
# for i, c in enumerate(color):
#     ax.plot(x_pred_index, f_pred_ens_mean[i, :], color = c, markersize = 1, markevery=2, markerfacecolor='None', label=f'Chain {i}', alpha = 0.8)
#     #ax.scatter(x_pred_index, u_pred_ens_mean[i], marker = mark, label=f'Pred mean - chain{i+1}' , s = 15)
#     ax.fill_between(x_pred_index, f_pred_ens_mean[i,:] + 2 * f_pred_ens_std[i,:], f_pred_ens_mean[i,:] - 2 * f_pred_ens_std[i,:], alpha = 0.3)
#                     #alpha = 0.3, label = f'$95\%$ CI')
# ax.scatter(X_r, y_r, label='Obs' , s = 20, facecolors='none', edgecolors='b')
# ax.set_xlabel('$x$', fontsize=16)
# ax.set_ylabel('$f(x)$', fontsize=16)
# ax.set_xlim(-0.72,0.72)
# ax.set_ylim(-2.5,2.5)
# ax.tick_params(axis='both', which = 'major', labelsize=13)
# ax.legend(fontsize=10, loc = 'upper left')
# fig.tight_layout()
# plt.show()
# plt.savefig(os.path.join(path_fig,'f_pred.png'))

#rhat and ess
rhat = tfp.mcmc.diagnostic.potential_scale_reduction(states.transpose((1,0,2)), independent_chain_ndims=1)
ess =  tfp.mcmc.effective_sample_size(states[0], filter_beyond_positive_pairs=True)
rhat_u = tfp.mcmc.diagnostic.potential_scale_reduction(u_pred_ens.transpose((1,0,2)), independent_chain_ndims=1)
rhat_f = tfp.mcmc.diagnostic.potential_scale_reduction(f_pred_ens.transpose((1,0,2)), independent_chain_ndims=1)

fig, ax = plt.subplots(1, 1, figsize=(4, 4), sharex='col', sharey='col', dpi = 300)
g = sns.histplot(rhat, bins = 50, kde=True, kde_kws = {'gridsize':5000})
g.tick_params(labelsize=16)
g.set_xlabel("$\hat{r}$", fontsize=18)
g.set_ylabel("Count", fontsize=18)
fig.tight_layout()
plt.savefig(os.path.join(path_fig,'rhat.png'))
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(4, 4), sharex='col', sharey='col', dpi = 300)
g = sns.histplot(rhat_u, bins = 25, kde=True, kde_kws = {'gridsize':5000})
g.tick_params(labelsize=16)
g.set_xlabel("$\hat{r}_{u}$", fontsize=18)
g.set_ylabel("Count", fontsize=18)
fig.tight_layout()
plt.savefig(os.path.join(path_fig,'rhat_u.png'))
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(4, 4), sharex='col', sharey='col', dpi = 300)
g = sns.histplot(rhat_f, bins = 25, kde=True, kde_kws = {'gridsize':5000})
g.tick_params(labelsize=16)
g.set_xlabel("$\hat{r}_{f}$", fontsize=18)
g.set_ylabel("Count", fontsize=18)
fig.tight_layout()
plt.savefig(os.path.join(path_fig,'rhat_f.png'))
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(4, 4), sharex='col', sharey='col', dpi = 300)
g = sns.histplot(ess, bins = 50, kde=True, kde_kws = {'gridsize':5000})
g.tick_params(labelsize=16)
g.set_xlabel("ESS", fontsize=18)
g.set_ylabel("Count", fontsize=18)
fig.tight_layout()
plt.savefig(os.path.join(path_fig,'ess.png'))
plt.show()

# trace plots
idx_low = np.argmin(rhat)
idx_high = np.argmax(rhat)
samples1 = states[:,:,idx_low]
df1 = pd.DataFrame({'chains': 'chain1', 'indice':np.arange(0, samples1[0,].shape[0], 5), 'trace':samples1[0, ::5]})
df2 = pd.DataFrame({'chains': 'chain2', 'indice':np.arange(0, samples1[1,].shape[0], 5), 'trace':samples1[1, ::5]})
df3 = pd.DataFrame({'chains': 'chain3', 'indice':np.arange(0, samples1[2,].shape[0], 5), 'trace':samples1[2, ::5]})
df = pd.concat([df1, df2, df3], ignore_index=True)
plt.figure(figsize=(4,4))
g = sns.jointplot(data=df, x='indice', y='trace', xlim=(0, 5000), ylim=(-4, 4), hue='chains', joint_kws={'alpha': 0.6})
g.ax_joint.tick_params(labelsize=18)
g.ax_joint.set_xlabel("Index", fontsize=24)
g.ax_joint.set_ylabel("Trace", fontsize=24)
g.ax_joint.legend(fontsize=16)
g.ax_marg_x.remove()
#plt.title('Trace plot for parameter with lowest $\hat{r}$')
plt.gcf().set_dpi(300)
plt.savefig(os.path.join(path_fig,'trace_plot_rhat_lowest.png'))
fig.tight_layout()
plt.show()

samples2 = states[:,:,idx_high]
df1 = pd.DataFrame({'chains': 'chain1', 'indice':np.arange(0, samples2[0, ::].shape[0], 5), 'trace':samples2[0, ::5]})
df2 = pd.DataFrame({'chains': 'chain2', 'indice':np.arange(0, samples2[1, ::].shape[0], 5), 'trace':samples2[1, ::5]})
df3 = pd.DataFrame({'chains': 'chain3', 'indice':np.arange(0, samples2[2, ::].shape[0], 5), 'trace':samples2[2, ::5]})
df = pd.concat([df1,df2, df3], ignore_index=True)
plt.figure(figsize=(4,4))
#g = sns.jointplot(data=df, x='indice', y='trace', xlim=(0, 5000), ylim=(-4, 4), hue='chains', joint_kws={'alpha': 1})
g = sns.jointplot(data=df, x='indice', y='trace', xlim=(0, 5000), ylim=(-4, 4), hue='chains', joint_kws={'alpha': 0.6})
g.ax_joint.tick_params(labelsize=18)
g.ax_joint.set_xlabel("Index", fontsize=24)
g.ax_joint.set_ylabel("Trace", fontsize=24)
g.ax_joint.legend(fontsize=16)
g.ax_marg_x.remove()
#plt.title('Trace plot for parameter with highest $\hat{r}$')
plt.gcf().set_dpi(300)
plt.savefig(os.path.join(path_fig,'trace_plot_rhat_highest.png'))
fig.tight_layout()
plt.show()

# =============================================================================
# Compute Hessian at the mean
# =============================================================================
cov0 = np.cov(samples[0].T)
cov0_diag = np.diag(cov0)
cov1 = np.cov(samples[1].T)
cov1_diag = np.diag(cov1)
cov2 = np.cov(samples[2].T)
cov2_diag = np.diag(cov2)

fig, ax  = plt.subplots(dpi = 300, figsize = (4,4))
g = ax.imshow(cov0[:100, :100], interpolation = 'nearest', cmap = 'hot')
plt.colorbar(g)
plt.show()

fig, ax  = plt.subplots(dpi = 300, figsize = (4,4))
g = ax.imshow(cov1[:100, :100], interpolation = 'nearest', cmap = 'hot')
plt.colorbar(g)
plt.show()

chain0 = states[0]
chain1 = states[1]
chain2 = states[2]
chain0_m = np.mean(chain0, axis = 0)
chain1_m = np.mean(chain1, axis = 0)
chain2_m = np.mean(chain2, axis = 0)
hess = jax.hessian(target_log_prob_fn)
hess_chain0 = hess(chain0_m)
_, s0, _ = jax.scipy.linalg.svd(jax.scipy.linalg.inv(hess_chain0))
hess_chain1 = hess(chain1_m)
_, s1, _ = jax.scipy.linalg.svd(jax.scipy.linalg.inv(hess_chain1))
hess_chain2 = hess(chain2_m)
_, s2, _ = jax.scipy.linalg.svd(jax.scipy.linalg.inv(hess_chain2))

s = np.concatenate((s0[np.newaxis, :], s1[np.newaxis, :], s2[np.newaxis, :]), axis = 0)
np.savetxt(os.path.join(path_f,'singular_values_posterior_hessian.out'), s)

fig, ax = plt.subplots(dpi = 300, figsize = (4,4))
#mark = [None, 'o', None]
linestyle = ['solid', 'dotted', 'dashed']
for i, ls in enumerate(linestyle):
  ax.plot(s[i], linestyle = ls, marker = None, markersize = 2, markevery= 100, markerfacecolor='None', label=f'chain{i+1}', alpha = 0.8)
ax.set_xlabel('Index', fontsize=16)
ax.set_ylabel('Eigenvalues', fontsize=16)
plt.yscale('log')
ax.tick_params(axis='both', which = 'major', labelsize=13)
ax.legend(fontsize=8)
plt.savefig(os.path.join(path_fig,'singular_values_posterior_hessian.png'))
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(4, 4), sharex='col', sharey='col', dpi = 300)
g = sns.histplot(chain0_m, bins = 50, kde=True, kde_kws = {'gridsize':5000})
g = sns.histplot(chain1_m, bins = 50, kde=True, kde_kws = {'gridsize':5000})
g = sns.histplot(chain2_m, bins = 50, kde=True, kde_kws = {'gridsize':5000})
g.tick_params(labelsize=16)
g.set_xlabel("Weight", fontsize=18)
g.set_ylabel("Count", fontsize=18)
fig.tight_layout()
plt.savefig(os.path.join(path_fig,'weight.png'))
plt.show()

# =============================================================================
# Plot posterior space
# =============================================================================


# class RandomCoordinates(object):
#     # randomly choose some directions
#     def __init__(self, origin):
#         self.origin = origin # (num_params,)
#         self.v0 = self.normalize(
#             random.normal(key = random.PRNGKey(88), shape = self.origin.shape), 
#             self.origin)  
#         self.v1 = self.normalize(
#             random.normal(key = random.PRNGKey(66), shape = self.origin.shape), 
#             self.origin)

#     def __call__(self, a, b):
#         return a*self.v0 + b * self.v1 + self.origin
    
#     def normalize(self, weights, origin):
#         return weights * jnp.abs(origin)/ jnp.abs(weights) #


# class LossSurface(object):
#     def __init__(self, loss_fn, coords):
#         self.loss_fn = loss_fn
#         self.coords = coords

#     def compile(self, range, num_points):
#         loss_fn_0d = lambda x, y: self.loss_fn(self.coords(x,y)) 
#         loss_fn_1d = jax.vmap(loss_fn_0d, in_axes = (0,0), out_axes = 0)
#         loss_fn_2d = jax.vmap(loss_fn_1d, in_axes = (0,0), out_axes = 0)
        
#         self.a_grid = jnp.linspace(-1.0, 1.0, num=num_points) ** 3 * range #(-5, 5) power rate
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
#             fig, ax = plt.subplots(dpi = 300, **kwargs)
#             ax.set_title("Loss Surface")
#             ax.set_aspect("equal")
            
#         # Set Levels
#         min_loss = zs.min()
#         max_loss = zs.max()
#         levels = jnp.linspace(
#                 max_loss, min_loss, num=levels
#             )[::-1]
            
#         # levels = jnp.exp(
#         #         jnp.log(min_loss) + 
#         #         jnp.linspace(0., 1.0, num=levels) ** 3 * (jnp.log(max_loss))- jnp.log(min_loss))
        
#         # Create Contour Plot
#         CS = ax.contourf(
#             xs,
#             ys,
#             zs,
#             levels=levels,
#             cmap= 'magma',
#             linewidths=0.75,
#             norm = mpl.colors.Normalize(vmin = min_loss, vmax = max_loss),
#         )
#         for i in points:
#         #origin_x, origin_y = self.project_points(self.coords.origin)
#             point_x, point_y = self.project_points(i)
#             ax.scatter(point_x, point_y, s = 20)
#         #ax.scatter(origin_x, origin_y, s = 1, c = 'r', marker = 'x')
#         ax.clabel(CS, fontsize=8, fmt="%1.2f")
#         #plt.colorbar(CS)
#         plt.show()
#         return ax

# coords = RandomCoordinates(chain0_m)
# loss_surface = LossSurface(target_log_prob_fn, coords)
# loss_surface.compile(range = 5, num_points= 500)
# ax = loss_surface.plot(levels = 15, points = [chain1_m, chain2_m])
# =============================================================================
# Plot different chains 
# =============================================================================

u_pred_ens = np.array([vmap(get_u_pred)(samples[i,:,:]) for i in range(samples.shape[0])]) 
f_pred_ens = np.array([vmap(get_f_pred)(samples[i,:,:]) for i in range(samples.shape[0])]) 

u_pred_ens_mean = np.mean(u_pred_ens, axis = 1)
u_pred_ens_std = np.std(u_pred_ens, axis = 1)
f_pred_ens_mean = np.mean(f_pred_ens, axis = 1)
f_pred_ens_std = np.std(f_pred_ens, axis = 1)

fig, ax = plt.subplots(dpi=300, figsize=(4,4))
ax.plot(x_pred_index, u_ref, 'k-', label='Exact', zorder=5) # Higher zorder to ensure the line is on top
color = ['#2ca02c', '#ff7f0e', '#1f77b4'] # green, orange, blue
linestyle = ['solid', 'dashdot', 'dashed']
# Adjust the zorder for fill_between
zorders_fill = [1, 2, 3] # blue highest, then orange, then green
# Plot lines and fill regions
for i, (c, ls, z) in enumerate(zip(color, linestyle, zorders_fill)):
    ax.plot(x_pred_index, u_pred_ens_mean[i, :], color=c, linestyle=ls, markersize=1, markevery=2, markerfacecolor='None', label=f'Chain {i}', alpha=0.8, zorder=z+1)
    ax.fill_between(x_pred_index, u_pred_ens_mean[i,:] + 2 * u_pred_ens_std[i,:], u_pred_ens_mean[i,:] - 2 * u_pred_ens_std[i,:], color=color[i], alpha=0.4, zorder=z)
ax.scatter(x_data, y_data, label='Obs' , s = 20, facecolors='none', edgecolors='b', zorder=6) # Higher zorder to ensure the scatter is on top
ax.set_xlabel('$x$', fontsize=16)
ax.set_ylabel('$u(x)$', fontsize=16)
ax.set_xlim(-0.72,0.72)
ax.set_ylim(-2.5,2.5)
ax.tick_params(axis='both', which='major', labelsize=13)
ax.legend(fontsize=10, loc='upper left')
fig.tight_layout()
plt.savefig(os.path.join(path_fig,f'1D_nonlinear_poisson_HMC_Nres_{Nres}_sigma_{sigma}_Nsamples_{Nsamples}_upred.png'))
plt.show()

fig, ax = plt.subplots(dpi=300, figsize=(4,4))
ax.plot(x_pred_index, f_ref, 'k-', label='Exact', zorder=5) # Higher zorder to ensure the line is on top
color = ['#2ca02c', '#ff7f0e', '#1f77b4'] # green, orange, blue
linestyle = ['solid', 'dashdot', 'dashed']
# Adjust the zorder for fill_between
zorders_fill = [1, 2, 3] # blue highest, then orange, then green
# Plot lines and fill regions
for i, (c, ls, z) in enumerate(zip(color, linestyle, zorders_fill)):
    ax.plot(x_pred_index, f_pred_ens_mean[i, :], color=c, linestyle=ls, markersize=1, markevery=2, markerfacecolor='None', label=f'Chain {i}', alpha=0.8, zorder=z+1)
    ax.fill_between(x_pred_index, f_pred_ens_mean[i,:] + 2 * f_pred_ens_std[i,:], f_pred_ens_mean[i,:] - 2 * f_pred_ens_std[i,:], color=color[i], alpha=0.4, zorder=z)
ax.scatter(X_r, y_r, label='Obs', s=20, facecolors='none', edgecolors='b', zorder=6) # Higher zorder to ensure the scatter is on top
ax.set_xlabel('$x$', fontsize=16)
ax.set_ylabel('$f(x)$', fontsize=16)
ax.set_xlim(-0.72,0.72)
ax.set_ylim(-2.5,2.5)
ax.tick_params(axis='both', which='major', labelsize=13)
ax.legend(fontsize=10, loc='upper left')
fig.tight_layout()
plt.savefig(os.path.join(path_fig,f'1D_nonlinear_poisson_HMC_Nres_{Nres}_sigma_{sigma}_Nsamples_{Nsamples}_fpred.png'))
plt.show()


# fig, ax = plt.subplots(dpi = 300, figsize = (4,4))
# ax.plot(x_pred_index, u_ref, 'k-', label='Exact')
# ax.plot(x_pred_index, u_pred_ens_mean, markersize = 1, markevery=2, markerfacecolor='None', label= f'HMC mean', alpha = 0.8)
# ax.fill_between(x_pred_index, u_pred_ens_mean + 2 * u_pred_ens_std, u_pred_ens_mean - 2 * u_pred_ens_std,
#                 alpha = 0.3, label = r'$95 \% $ CI')
# ax.scatter(data[:,0], data[:,1], label='Obs' , s = 20, facecolors='none', edgecolors='b')
# ax.set_xlabel('$x$', fontsize=16)
# ax.set_ylabel('$u(x)$', fontsize=16)
# ax.set_xlim(-0.72,0.72)
# ax.set_ylim(-2.5,2.5)
# ax.tick_params(axis='both', which = 'major', labelsize=13)
# ax.legend(fontsize=10)
# fig.tight_layout()
# plt.savefig(os.path.join(path_fig,f'1D_nonlinear_poisson_HMC_mean_Nres_{Nres}_sigma_{sigma}_Nsamples_{Nsamples}_upred.png'))
# plt.show()

# fig, ax = plt.subplots(dpi = 300, figsize = (4,4))
# r_ref = f(x_pred_index)
# ax.plot(x_pred_index, r_ref, 'k-', label='Exact')
# ax.plot(x_pred_index, f_pred_ens_mean, markersize = 1, markevery=2, markerfacecolor='None', label=f'HMC mean', alpha = 0.8)
# ax.fill_between(x_pred_index, f_pred_ens_mean + 2 * f_pred_ens_std, f_pred_ens_mean - 2 * f_pred_ens_std,
#                 alpha = 0.3, label = r'$95 \% $ CI')
# ax.scatter(Dres[:,0], Dres[:,1], label='Obs' , s = 20, facecolors='none', edgecolors='b')
# ax.set_xlabel('$x$', fontsize=16)
# ax.set_xlim(-0.72,0.72)
# ax.set_ylim(-2.5,2.5)
# ax.set_ylabel('$f(x)$', fontsize=16)
# ax.tick_params(axis='both', which = 'major', labelsize=13)
# ax.legend(fontsize=10, loc= 'upper left')
# fig.tight_layout()
# plt.savefig(os.path.join(path_fig,f'1D_nonlinear_poisson_HMC_mean_Nres_{Nres}_sigma_{sigma}_Nsamples_{Nsamples}_fpred.png'))
# plt.show()


f_rec.close()


sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

# Generating data for a normal distribution
x = np.linspace(-4, 4, 1000)
y = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

# Creating the plot
fig, ax = plt.subplots(dpi = 100)
ax.plot(x, y, color='black')  # black solid line for the normal distribution
ax.fill_between(x, y, color='#C5E0B4')  # filling below the line with yellow color

# Removing axis
ax.set_axis_off()

plt.show()

# x = np.linspace(-5, 5, 100)
# y = np.linspace(-5, 5, 100)
# X, Y = np.meshgrid(x, y)
# Z = np.exp(-0.5 * ((X - 1)**2 + (Y - 1)**2) / 1) + np.exp(-0.5 * ((X + 2)**2 + (Y + 2)**2) / 0.5) + np.exp(-0.2 * ((X - 1.5)**2 + (Y - 2)**2) / 0.2) + np.exp(-0.7 * ((X + 3)**2 + (Y + 3)**2) / 1.2)

# # Plotting
# fig = plt.figure(dpi = 200)
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color='#FFE699', edgecolor='none')

# ax.set_axis_off()  # Remove axes for clean visualization

# plt.show()