#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 22:41:24 2024

@author: yifei_linux
"""

#Import dependencies
import jax 
import os
import jax.numpy as jnp
from jax import random, grad, vmap, jit
from jax.flatten_util import ravel_pytree
from jax.example_libraries import optimizers
# from jax.lib import xla_bridge

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
from pyDOE import lhs
from time import perf_counter

#command line argument parser
parser = argparse.ArgumentParser(description="2D Darcy with HMC")
parser.add_argument(
    "--rand_seed",
    type=int,
    default=111,
    help="random seed")
parser.add_argument(
    "--Nres",
    type=int,
    default=500,
    help="Number of reisudal points")
parser.add_argument(
    "--Nchains",
    type=int,
    default=3,
    help="Number of Posterior chains")
parser.add_argument(
    "--Nsamples",
    type=int,
    default=500,
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

# print(xla_bridge.get_backend().platform)
#jax.config.update('jax_platform_name', 'cpu')
print(f'jax is using: {jax.devices()} \n')

#Define parameters
layers_k = [2, 60, 60, 60, 60, 1]
layers_h = [2, 60, 60, 60, 60, 1]
lbt = np.array([0., 0.])
ubt = np.array([1., 0.5])
num_print = 200
Nk = 40
Nh = 40
sigma = 0.1
sigma_r = 0.3536
sigma_d = 0.1
sigma_nbl = 0.0632
sigma_nbb = 0.089
sigma_nbt = 0.089
sigma_dbr = 0.0632
sigma_p = 2.369
rand_seed = args.rand_seed
Nres = args.Nres
Nchains = args.Nchains
Nsamples = args.Nsamples
Nburn = args.Nburn
model_load = args.data_load
dataset = dict()
x = np.linspace(lbt[0], ubt[0], 256)
y = np.linspace(lbt[1], ubt[1], 128)
XX, YY = np.meshgrid(x,y)

#Load data
k_ref = np.loadtxt('k_ref_05.out', dtype=float) #(32768,)
h_ref = np.loadtxt('h_ref_05.out', dtype=float) #(32768,)
y_ref = np.log(k_ref)
k_ref = y_ref
h_ref = h_ref - h_ref.min(0) 
x_ref = np.loadtxt('coord_ref.out', dtype=float)  #(32768, 2)
N = k_ref.shape[0]

#Create dataset
#k, h measurements and residual points
np.random.seed(rand_seed)
#idx_k = np.random.choice(N, Nk, replace= False) 
idx_k = np.loadtxt('Nk_40_Nh_40_randseed_111_idxk.out').astype(np.int64)
y_k, x_k = k_ref[idx_k][:,np.newaxis] + np.random.normal(0,sigma,(Nk,1)).astype(np.float32), x_ref[idx_k, :]
k_data = jnp.concatenate([x_k,y_k], axis=1)
#idx_h = np.random.choice(N, Nh, replace= False)
idx_h = np.loadtxt('Nk_40_Nh_40_randseed_111_idxk.out').astype(np.int64)
y_h, x_h = h_ref[idx_h][:,np.newaxis] + np.random.normal(0,sigma,(Nh,1)).astype(np.float32), x_ref[idx_h, :]
h_data = jnp.concatenate([x_h,y_h], axis=1)

x_nor = lhs(2,200000)[:Nres,:]
#x_nor = np.loadtxt(f'Nk_40_Nh_40_Nres_{Nres}_randseed_111_xnor.out')
x_res = lbt + (ubt -lbt) * x_nor
y_res= np.zeros((Nres,1)) + np.random.normal(0,sigma,(Nres,1)).astype(np.float32)
res = jnp.concatenate([x_res, y_res],axis=1)

# Dirichlet BC at right
x2_dbr = np.linspace(lbt[1],ubt[1],16)[:,np.newaxis]
x1_dbr = ubt[0]*jnp.ones_like(x2_dbr)
y_dbr = jnp.zeros_like(x2_dbr) + np.random.normal(0,sigma,(16,1)).astype(np.float32)
dbr = jnp.concatenate([x1_dbr,x2_dbr,y_dbr],axis=1)

# Neumann BC at lefth
x2_nbl = np.linspace(lbt[1],ubt[1],16)[:,np.newaxis]
x1_nbl = lbt[0]*jnp.ones_like(x2_nbl)
y_nbl = jnp.ones_like(x2_nbl) + np.random.normal(0,sigma,(16,1)).astype(np.float32)
nbl = jnp.concatenate([x1_nbl,x2_nbl,y_nbl],axis=1)

# Neumann BC at top
x1_nbt = np.linspace(lbt[0],ubt[0],32)[:,np.newaxis]
x2_nbt = ubt[1]*jnp.ones_like(x1_nbt)
y_nbt = jnp.zeros_like(x1_nbt) + np.random.normal(0,sigma,(32,1)).astype(np.float32)
nbt = jnp.concatenate([x1_nbt,x2_nbt,y_nbt],axis=1)

# Neumann BC at below
x1_nbb = np.linspace(lbt[0],ubt[0],32)[:,np.newaxis]
x2_nbb = lbt[1]*jnp.ones_like(x1_nbb)
y_nbb = jnp.zeros_like(x1_nbb) + np.random.normal(0,sigma,(32,1)).astype(np.float32)
nbb = jnp.concatenate([x1_nbb,x2_nbb,y_nbb],axis=1)

dataset.update({'k_data': k_data})
dataset.update({'h_data': h_data})
dataset.update({'res': res})
dataset.update({'dbr': dbr})
dataset.update({'nbl': nbl})
dataset.update({'nbt': nbt})
dataset.update({'nbb': nbb})

path_f   = f'2D_Nk_{Nk}_Nh_{Nh}_Nres_{Nres}_sigma_{sigma}_Nburn_{Nburn}_Nsamples_{Nsamples}_HMC'
path_fig = os.path.join(path_f,'figures')
if not os.path.exists(path_f):
    os.makedirs(path_f)
if not os.path.exists(path_fig):
    os.makedirs(path_fig)
f_rec = open(os.path.join(path_f,'record.out'), 'a+')

print(f'method:HMC rand_seed:{rand_seed}', file = f_rec)
print(f'layers_k:{layers_k} layers_h:{layers_h}', file = f_rec)
print(f'Nk:{Nk} Nh:{Nh} Nres:{Nres}\n', file = f_rec)
print(f'sigma:{sigma} sigma_r:{sigma_r} sigma_nbl:{sigma_nbl} sigma_nbb:{sigma_nbb} sigma_nbt:{sigma_nbt} sigma_dbr:{sigma_dbr} sigma_p:{sigma_p}\n')
print(f'sigma:{sigma} sigma_r:{sigma_r} sigma_nbl:{sigma_nbl} sigma_nbb:{sigma_nbb} sigma_nbt:{sigma_nbt} sigma_dbr:{sigma_dbr} sigma_p:{sigma_p}\n', file = f_rec)

rl2e = lambda yest, yref : spl.norm(yest - yref, 2) / spl.norm(yref, 2) 
infe = lambda yest, yref : spl.norm(yest - yref, np.inf) 
lpp = lambda h, href, sigma: np.sum( -(h - href)**2/(2*sigma**2) - 1/2*np.log( 2*np.pi) - 2*np.log(sigma))

def pcolormesh(XX, YY, Z, points = None, title = None, savefig = None, cmap='jet', vmax = None, vmin = None):
    fig, ax = plt.subplots(dpi = 300, figsize = (6,4))
    if vmax is not None:
        c = ax.pcolormesh(XX, YY, Z, vmin = vmin, vmax = vmax, cmap=cmap)
    else:
        c = ax.pcolormesh(XX, YY, Z, vmin = np.min(Z), vmax = np.max(Z), cmap=cmap)
    if points is not None:
        plt.plot(points[:,0], points[:,1], 'ko', markersize = 1.0)
    fig.colorbar(c, ax=ax, fraction= 0.05, pad= 0.05)
    ax.tick_params(axis='both', which = 'major', labelsize=16)
    ax.set_xlabel('$x_1$', fontsize=20)
    ax.set_ylabel('$x_2$', fontsize=20)
    if title is not None:
        ax.set_title(title, fontsize=14)
    fig.tight_layout()
    #ax.set_aspect('equal')
    plt.show()
    if savefig is not None:
        plt.savefig(os.path.join(path_fig,f'{savefig}.png'))

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
    def __init__(self, key, layers_k, layers_h, dataset, lbt, ubt, sigma_r, sigma_d, sigma_nbl, sigma_nbb, sigma_nbt, sigma_dbr, sigma_p):  

        self.lbt = lbt #domain lower corner
        self.ubt = ubt #domain upper corner
        self.sigma_r = sigma_r #residual term
        self.sigma_d = sigma_d #data term
        self.sigma_nbl = sigma_nbl
        self.sigma_nbb = sigma_nbb
        self.sigma_nbt = sigma_nbt
        self.sigma_dbr = sigma_dbr
        self.sigma_p = sigma_p #prior term
        self.itercount = itertools.count()
        
        # Prepare normalized training data
        self.dataset = dataset
        self.x_res, self.y_res = dataset['res'][:,0:2], dataset['res'][:,2:3]
        self.x_dbr, self.y_dbr = dataset['dbr'][:,0:2], dataset['dbr'][:,2:3]
        self.x_nbl, self.y_nbl = dataset['nbl'][:,0:2], dataset['nbl'][:,2:3]
        self.x_nbt, self.y_nbt = dataset['nbt'][:,0:2], dataset['nbt'][:,2:3]
        self.x_nbb, self.y_nbb = dataset['nbb'][:,0:2], dataset['nbb'][:,2:3]
        self.x_h, self.y_h = dataset['h_data'][:,0:2], dataset['h_data'][:,2:3]
        self.x_k, self.y_k = dataset['k_data'][:,0:2], dataset['k_data'][:,2:3]
        
        # Initalize the network
        key, *keys = random.split(key, num = 3)
        self.init_k, self.forward_k = FNN(layers_k, activation=jnp.tanh)
        self.params_k = self.init_k(keys[0])
        raveled_k, self.unravel_k = ravel_pytree(self.params_k)
        self.num_params_k = raveled_k.shape[0]
        
        self.init_h, self.forward_h = FNN(layers_h, activation=jnp.tanh)
        self.params_h = self.init_h(keys[1])
        raveled_h, self.unravel_h = ravel_pytree(self.params_h)
        self.num_params_h = raveled_h.shape[0]
        self.num_params = self.num_params_k + self.num_params_h 
        
        # Evaluate the state, parameter and the residual over the grid
        self.h_pred_map = vmap(self.h_net, (None, 0, 0)) 
        self.k_pred_map = vmap(self.k_net, (None, 0, 0))  
        self.r_pred_map = vmap(self.res_net, (None, 0, 0))
          
        # Optimizer
        lr = optimizers.exponential_decay(1e-4, decay_steps=5000, decay_rate=0.9)
        self.opt_init, \
        self.opt_update, \
        self.get_params = optimizers.adam(lr)
          
        self.opt_state_k = self.opt_init(self.params_k)
        self.opt_state_h = self.opt_init(self.params_h)

    @partial(jit, static_argnums=(0,))
    def h_net(self, params, x1, x2): #no problem
        inputs = jnp.hstack([x1, x2])
        outputs = self.forward_h(params[1], inputs)
        return outputs[0] 
          
    @partial(jit, static_argnums=(0,))
    def k_net(self, params, x1, x2): #no problem
        inputs = jnp.hstack([x1, x2])
        outputs = self.forward_k(params[0], inputs)
        return outputs[0]
    
    @partial(jit, static_argnums=(0,))
    def qx(self, params, x1, x2):
        k = jnp.exp(self.k_net(params, x1, x2))
        #k = self.k_net(params, x1, x2)
        dhdx = grad(self.h_net, argnums=1)(params, x1, x2)
        return -k*dhdx
        
    @partial(jit, static_argnums=(0,))
    def qy(self, params, x1, x2):
        k  = jnp.exp(self.k_net(params, x1, x2))
        #k = self.k_net(params, x1, x2)
        dhdy = grad(self.h_net, argnums=2)(params, x1, x2)
        return -k*dhdy
        
    @partial(jit, static_argnums=(0,))
    def res_net(self, params, x1, x2):
        dhdx2 = grad(self.qx, argnums=1)(params, x1, x2)
        dhdy2 = grad(self.qy, argnums=2)(params, x1, x2)
        return dhdx2 + dhdy2
    
    @partial(jit, static_argnums=(0,))
    def h_pred_vector(self, flat_params):
        # For HMC
        params_k = self.unravel_k(flat_params[:self.num_params_k])
        params_h = self.unravel_h(flat_params[self.num_params_k:])
        params = [params_k, params_h]
        h_pred_vec = vmap(self.h_net, (None, 0, 0))(params, self.x_h[:,0], self.x_h[:,1])
        return h_pred_vec
    
    @partial(jit, static_argnums=(0,))
    def k_pred_vector(self, flat_params):
        # For HMC
        params_k = self.unravel_k(flat_params[:self.num_params_k])
        params_h = self.unravel_h(flat_params[self.num_params_k:])
        params = [params_k, params_h]
        k_pred_vec = vmap(self.k_net, (None, 0, 0))(params, self.x_k[:,0], self.x_k[:,1])
        return k_pred_vec
    
    @partial(jit, static_argnums=(0,))
    def r_pred_vector(self, flat_params): 
        # For HMC
        params_k = self.unravel_k(flat_params[:self.num_params_k])
        params_h = self.unravel_h(flat_params[self.num_params_k:])
        params = [params_k, params_h]
        r_pred_vec = vmap(self.res_net, (None, 0, 0 ))(params, self.x_res[:,0], self.x_res[:,1])
        return r_pred_vec               
    
    @partial(jit, static_argnums=(0,))
    def dbr_pred_vector(self, flat_params): 
        # For HMC
        params_k = self.unravel_k(flat_params[:self.num_params_k])
        params_h = self.unravel_h(flat_params[self.num_params_k:])
        params = [params_k, params_h]
        dbr_pred_vec = vmap(self.h_net, (None, 0, 0))(params, self.x_dbr[:,0], self.x_dbr[:,1])
        return dbr_pred_vec 
    
    @partial(jit, static_argnums=(0,))
    def nbl_pred_vector(self, flat_params): 
        params_k = self.unravel_k(flat_params[:self.num_params_k])
        params_h = self.unravel_h(flat_params[self.num_params_k:])
        params = [params_k, params_h]
        nbl_pred_vec = vmap(self.qx, (None, 0, 0))(params, self.x_nbl[:,0], self.x_nbl[:,1])
        return nbl_pred_vec
        
    @partial(jit, static_argnums=(0,))
    def nbt_pred_vector(self, flat_params): 
        params_k = self.unravel_k(flat_params[:self.num_params_k])
        params_h = self.unravel_h(flat_params[self.num_params_k:])
        params = [params_k, params_h]
        nbt_pred_vec = vmap(self.qy, (None, 0, 0))(params, self.x_nbt[:,0], self.x_nbt[:,1])
        return nbt_pred_vec
    
    @partial(jit, static_argnums=(0,))
    def nbb_pred_vector(self, flat_params): 
        params_k = self.unravel_k(flat_params[:self.num_params_k])
        params_h = self.unravel_h(flat_params[self.num_params_k:])
        params = [params_k, params_h]
        nbb_pred_vec = vmap(self.qy, (None, 0, 0))(params, self.x_nbb[:,0], self.x_nbb[:,1])
        return nbb_pred_vec 
    
    @partial(jit, static_argnums=(0,))
    def target_log_prob_fn(self, theta):
        prior = -1/(2*self.sigma_p**2) * jnp.sum((theta)**2)
        r_likelihood = -1/(2*self.sigma_r**2) * jnp.sum((self.y_res.ravel() - self.r_pred_vector(theta))**2)
        k_likelihood = -1/(2*self.sigma_d**2) * jnp.sum((self.y_k.ravel() - self.k_pred_vector(theta))**2)
        h_likelihood = -1/(2*self.sigma_d**2) * jnp.sum((self.y_h.ravel() - self.h_pred_vector(theta))**2)
        dbr_likelihood = -1/(2*self.sigma_dbr**2) * jnp.sum((self.y_dbr.ravel() - self.dbr_pred_vector(theta))**2)
        nbl_likelihood = -1/(2*self.sigma_nbl**2) * jnp.sum((self.y_nbl.ravel() - self.nbl_pred_vector(theta))**2)
        nbb_likelihood = -1/(2*self.sigma_nbb**2) * jnp.sum((self.y_nbb.ravel() - self.nbb_pred_vector(theta))**2)
        nbt_likelihood = -1/(2*self.sigma_nbt**2) * jnp.sum((self.y_nbt.ravel() - self.nbt_pred_vector(theta))**2)
        return prior + r_likelihood + k_likelihood + h_likelihood + dbr_likelihood + nbl_likelihood + nbb_likelihood + nbt_likelihood
    
key1, key2 = random.split(random.PRNGKey(0), 2)
pinn = PINN(key2, layers_k, layers_h, dataset, lbt, ubt,  sigma_r, sigma_d, sigma_nbl, sigma_nbb, sigma_nbt, sigma_dbr, sigma_p)

new_key, *subkeys = random.split(key1, Nchains + 1)
init_state = jnp.zeros((1, pinn.num_params))
for key in subkeys:
    init_state = jnp.concatenate([init_state,random.normal(key ,(1, pinn.num_params))], axis=0)

nuts_kernel = tfp.mcmc.NoUTurnSampler(
  target_log_prob_fn = pinn.target_log_prob_fn, step_size = 0.0005, max_tree_depth=10, max_energy_diff=1000.0,
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

ts = perf_counter()
print('\nStart HMC Sampling')
states, trace = jit(vmap(run_chain, in_axes=(0, None)))(init_state, new_key)
print('\nFinish HMC Sampling')
np.save(os.path.join(path_f,'chains'), states)
#states = np.load(os.path.join(path_f,'chains.npy'))
timings = perf_counter() - ts
print(f"HMC: {timings} s")
print(f"HMC: {timings} s", file = f_rec)
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

samples = states #(Nchains, Nsamples, Nparams)

@jit
def get_h_pred(sample):
    params_k = pinn.unravel_k(sample[:pinn.num_params_k])
    params_h = pinn.unravel_h(sample[pinn.num_params_k:])
    params = [params_k, params_h]
    return pinn.h_pred_map(params,x_ref[:,0], x_ref[:,1])

@jit
def get_k_pred(sample):
    params_k = pinn.unravel_k(sample[:pinn.num_params_k])
    params_h = pinn.unravel_h(sample[pinn.num_params_k:])
    params = [params_k, params_h]
    return pinn.k_pred_map(params,x_ref[:,0], x_ref[:,1])

@jit
def get_r_pred(sample):
    params_k = pinn.unravel_k(sample[:pinn.num_params_k])
    params_h = pinn.unravel_h(sample[pinn.num_params_k:])
    params = [params_k, params_h]
    return pinn.r_pred_map(params,x_ref[:,0], x_ref[:,1])

h_pred_ens = np.array([vmap(get_h_pred)(samples[i,:,:]) for i in range(samples.shape[0])]) #(Nchains, Nsamples, 32768)
k_pred_ens = np.array([vmap(get_k_pred)(samples[i,:,:]) for i in range(samples.shape[0])]) 

# h_pred_ens = vmap(get_h_pred)(samples[:,:,:]) 
# k_pred_ens = vmap(get_k_pred)(samples[:,:,:]) 

h_pred_ens_mean = np.mean(h_pred_ens, axis = 0) #(Nchains, 32768)
h_pred_ens_std = np.std(h_pred_ens, axis = 0)
k_pred_ens_mean = np.mean(k_pred_ens, axis = 0)
k_pred_ens_std = np.std(k_pred_ens, axis = 0)

h_env = np.logical_and( (h_pred_ens_mean < h_ref.ravel() + 2*h_pred_ens_std), (h_pred_ens_mean > h_ref.ravel() - 2*h_pred_ens_std) )
k_env = np.logical_and( (k_pred_ens_mean < k_ref.ravel() + 2*k_pred_ens_std), (k_pred_ens_mean > k_ref.ravel() - 2*k_pred_ens_std) )

for i in range(Nchains):
    rl2e_h = rl2e(h_pred_ens_mean[i, :], h_ref)
    infe_h = infe(h_pred_ens_mean[i, :], h_ref)
    lpp_h = lpp(h_pred_ens_mean[i, :], h_ref, h_pred_ens_std[i, :])
    rl2e_k = rl2e(k_pred_ens_mean[i, :], k_ref)
    infe_k = infe(k_pred_ens_mean[i, :], k_ref)
    lpp_k = lpp(k_pred_ens_mean[i, :], k_ref, k_pred_ens_std[i, :])
    
    print('chains:{i}\n')
    print('h prediction:\n')
    print('Relative RL2 error: {}'.format(rl2e_h))
    print('Absolute inf error: {}'.format(infe_h))
    print('Average standard deviation: {}'.format(np.mean(h_pred_ens_std[i, :])))
    print('log predictive probability: {}'.format(lpp_h))
    print('Percentage of coverage:{}\n'.format(np.sum(h_env)/32768))
    
    print('k prediction:\n')
    print('Relative RL2 error: {}'.format(rl2e_k))
    print('Absolute inf error: {}'.format(infe_k))
    print('Average standard deviation: {}'.format(np.mean(k_pred_ens_std[i, :])))
    print('log predictive probability: {}'.format(lpp_k))
    print('Percentage of coverage:{}\n'.format(np.sum(k_env)/32768))
    
    print('chains:{i}\n', file = f_rec)
    print('h prediction:\n', file = f_rec)
    print('Relative RL2 error: {}'.format(rl2e_h), file = f_rec)
    print('Absolute inf error: {}'.format(infe_h), file = f_rec)
    print('Average standard deviation: {}'.format(np.mean(h_pred_ens_std[i, :])), file = f_rec)
    print('log predictive probability: {}'.format(lpp_h), file = f_rec)
    print('Percentage of coverage:{}\n'.format(np.sum(h_env)/32768), file = f_rec)
    
    print('k prediction:\n', file = f_rec)
    print('Relative RL2 error: {}'.format(rl2e_k), file = f_rec)
    print('Absolute inf error: {}'.format(infe_k), file = f_rec)
    print('Average standard deviation: {}'.format(np.mean(k_pred_ens_std[i, :])), file = f_rec)
    print('log predictive probability: {}'.format(lpp_k), file = f_rec)
    print('Percentage of coverage:{}\n'.format(np.sum(k_env)/32768), file = f_rec)


#Plot of k field
pcolormesh(XX, YY, k_ref.reshape(128,256), points = None, title = None, savefig = 'y_ref')
savefig = f'2D_gwf_rPINN_sigmar_{sigma_r}_Nsamples_{Nsamples}_ypred_mean'
pcolormesh(XX, YY, k_pred_ens_mean.reshape(128,256), points = None, title = None, savefig = savefig)
savefig = f'2D_gwf_rPINN_sigmar_{sigma_r}_Nsamples_{Nsamples}_ypred_std'
pcolormesh(XX, YY, k_pred_ens_std.reshape(128,256), points = x_k, title = None, savefig = savefig)
savefig = f'2D_gwf_rPINN_sigmar_{sigma_r}_Nsamples_{Nsamples}_ypred_diff'
pcolormesh(XX, YY, np.abs(k_pred_ens_mean.reshape(128,256) - k_ref.reshape(128,256)), points = x_k, title = None, savefig = savefig)
savefig = f'2D_gwf_rPINN_sigmar_{sigma_r}_Nsamples_{Nsamples}_ypred_env'
pcolormesh(XX, YY, k_env.reshape(128,256), points = x_k, title = None, savefig = savefig)

#Plot of h field
pcolormesh(XX, YY, h_ref.reshape(128,256), points = None, title = None, savefig = 'h_ref')
savefig = f'2D_gwf_rPINN_sigmar_{sigma_r}_Nsamples_{Nsamples}_hpred_mean'
pcolormesh(XX, YY, h_pred_ens_mean.reshape(128,256), points = None, title = None, savefig = savefig )
savefig = f'2D_gwf_rPINN_sigmar_{sigma_r}_Nsamples_{Nsamples}_hpred_std'
pcolormesh(XX, YY, h_pred_ens_std.reshape(128,256), points = x_h, title = None, savefig = savefig )
savefig = f'2D_gwf_rPINN_sigmar_{sigma_r}_Nsamples_{Nsamples}_hpred_diff'
pcolormesh(XX, YY, np.abs(h_pred_ens_mean.reshape(128,256) - h_ref.reshape(128,256)), points = x_h, title = None, savefig = savefig )
savefig = f'2D_gwf_rPINN_sigmar_{sigma_r}_Nsamples_{Nsamples}_hpred_env'
pcolormesh(XX, YY, h_env.reshape(128,256), points = x_h, title = None, savefig = savefig, cmap='jet', vmax = 1, vmin = 0)


rhat = tfp.mcmc.diagnostic.potential_scale_reduction(states.transpose((1,0,2)), independent_chain_ndims=1)
ess =  tfp.mcmc.effective_sample_size(states[0], filter_beyond_positive_pairs=True)

fig, ax = plt.subplots(1, 1, figsize=(4, 4), sharex='col', sharey='col', dpi = 300)
g = sns.histplot(rhat, bins = 50, kde=True, kde_kws = {'gridsize':5000})
g.tick_params(labelsize=16)
g.set_xlabel("$\hat{r}$", fontsize=18)
g.set_ylabel("Count", fontsize=18)
fig.tight_layout()
plt.savefig(os.path.join(path_fig,'rhat.png'))
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(4, 4), sharex='col', sharey='col', dpi = 300)
g = sns.histplot(ess, bins = 50, kde=True, kde_kws = {'gridsize':5000})
g.tick_params(labelsize=16)
g.set_xlabel("ESS", fontsize=18)
g.set_ylabel("Count", fontsize=18)
fig.tight_layout()
plt.savefig(os.path.join(path_fig,'ess.png'))
plt.show()

idx_low = np.argmin(rhat)
idx_high = np.argmax(rhat)
samples1 = states[:,:,idx_low]
df1 = pd.DataFrame({'chains': 'chain1', 'indice':np.arange(0, samples1[0,].shape[0], 5), 'trace':samples1[0, ::5]})
df2 = pd.DataFrame({'chains': 'chain2', 'indice':np.arange(0, samples1[1,].shape[0], 5), 'trace':samples1[1, ::5]})
df3 = pd.DataFrame({'chains': 'chain3', 'indice':np.arange(0, samples1[2,].shape[0], 5), 'trace':samples1[2, ::5]})
df = pd.concat([df1, df2, df3], ignore_index=True)
plt.figure(figsize=(4,4))
g = sns.jointplot(data=df, x='indice', y='trace', xlim=(0, 500), ylim=(-6, 6), hue='chains', joint_kws={'alpha': 0.6})
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
g = sns.jointplot(data=df, x='indice', y='trace', xlim=(0, 500), ylim=(-6, 6), hue='chains', joint_kws={'alpha': 0.6})
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

mark = [None, 'o', None]
linestyle = ['solid', 'dotted', 'dashed']
fig, ax = plt.subplots(dpi = 300, figsize = (4,4))
for i, mark in enumerate(mark):
  ax.plot(np.arange(Nsamples)[::10], target_log_prob[i,::10], marker = mark, markersize = 2, markevery= 100, markerfacecolor='None', linestyle = 'dashed', label = f'chain {i + 1}', alpha = 0.8)
ax.set_xlabel('Sample index', fontsize = 15)
ax.set_ylabel('Negative log prob', fontsize = 15)
ax.tick_params(axis='both', which = 'major', labelsize=12)
ax.set_xlim(0,Nsamples)
ax.legend(fontsize=10)
plt.savefig(os.path.join(path_fig,'target_log_prob.png'))
plt.show()

# chain0 = states[0]
# chain1 = states[1]
# chain2 = states[2]
# chain0_m = np.mean(chain0, axis = 0)
# chain1_m = np.mean(chain1, axis = 0)
# chain2_m = np.mean(chain2, axis = 0)
# hess = jax.hessian(pinn.target_log_prob_fn)
# hess_chain0 = hess(chain0_m)
# _, s0, _ = jax.scipy.linalg.svd(jax.scipy.linalg.inv(hess_chain0))
# hess_chain1 = hess(chain1_m)
# _, s1, _ = jax.scipy.linalg.svd(jax.scipy.linalg.inv(hess_chain1))
# hess_chain2 = hess(chain2_m)
# _, s2, _ = jax.scipy.linalg.svd(jax.scipy.linalg.inv(hess_chain2))

# s = np.concatenate((s0[np.newaxis, :], s1[np.newaxis, :], s2[np.newaxis, :]), axis = 0)
# np.savetxt(os.path.join(path_f,'singular_values_posterior_hessian.out'), s)

# fig, ax = plt.subplots(dpi = 300, figsize = (4,4))
# #mark = [None, 'o', None]
# linestyle = ['solid', 'dotted', 'dashed']
# for i, ls in enumerate(linestyle):
#   ax.plot(s[i], linestyle = ls, marker = None, markersize = 2, markevery= 100, markerfacecolor='None', label=f'chain{i+1}', alpha = 0.8)
# ax.set_xlabel('Index', fontsize=16)
# ax.set_ylabel('Eigenvalues', fontsize=16)
# plt.yscale('log')
# ax.tick_params(axis='both', which = 'major', labelsize=13)
# ax.legend(fontsize=8)
# plt.savefig(os.path.join(path_fig,'singular_values_posterior_hessian.png'))
# plt.show()

# fig, ax = plt.subplots(1, 1, figsize=(4, 4), sharex='col', sharey='col', dpi = 300)
# g = sns.histplot(chain0_m, bins = 50, kde=True, kde_kws = {'gridsize':5000})
# g = sns.histplot(chain1_m, bins = 50, kde=True, kde_kws = {'gridsize':5000})
# g = sns.histplot(chain2_m, bins = 50, kde=True, kde_kws = {'gridsize':5000})
# g.tick_params(labelsize=16)
# g.set_xlabel("Weight", fontsize=18)
# g.set_ylabel("Count", fontsize=18)
# fig.tight_layout()
# plt.savefig(os.path.join(path_fig,'weight.png'))
# plt.show()
