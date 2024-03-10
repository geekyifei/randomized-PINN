#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 14:09:03 2024

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

#command line argument parser
parser = argparse.ArgumentParser(description="2D Darcy with rPINN")
parser.add_argument(
    "--rand_seed",
    type=int,
    default=111,
    help="random seed")
parser.add_argument(
    "--sigma",
    type=float,
    default=1,
    help="Measurement noise level")
parser.add_argument(
    "--lambda_r",
    type=float,
    default=1,
    help="residual weight for the PINN")
parser.add_argument(
    "--lambda_nbl",
    type=float,
    default=1,
    help="left neumann boundary weight for the PINN")
parser.add_argument(
    "--lambda_nbb",
    type=float,
    default=1,
    help="lower neumann boundary weight for the PINN")
parser.add_argument(
    "--lambda_nbt",
    type=float,
    default=1,
    help="top neumann boundary weight for the PINN")
parser.add_argument(
    "--lambda_db",
    type=float,
    default=1,
    help="dirichlet boundary weight for the PINN")
parser.add_argument(
    "--lambda_d",
    type=float,
    default=1,
    help="Data weight for the PINN")
parser.add_argument(
    "--lambda_p",
    type=float,
    default=1,
    help="L2 reg weight")
parser.add_argument(
    "--Nres",
    type=int,
    default=500,
    help="Number of reisudal points")
parser.add_argument(
    "--Nsamples",
    type=int,
    default=500,
    help="Number of posterior samples")
parser.add_argument(
    "--nIter",
    type=int,
    default=50000,
    help="Number of training epochs per realization")
parser.add_argument(
    "--data_load",
    type=bool,
    default=False,
    help="If to load data")
parser.add_argument(
    "--method",
    type=str,
    default='DE',
    help="Method for Bayesian training")
args = parser.parse_args()

print(f'jax is using: {jax.devices()} \n')

#Define parameters
layers_k = [2, 60, 60, 60, 60, 1]
layers_h = [2, 60, 60, 60, 60, 1]
num_params = 22442
lbt = np.array([0., 0.])
ubt = np.array([1., 0.5])
num_print = 200
Nk = 40
Nh = 40
rand_seed = args.rand_seed
sigma = args.sigma
lambda_r = args.lambda_r
lambda_d = args.lambda_d
lambda_nbl = args.lambda_nbl
lambda_nbb = args.lambda_nbb
lambda_nbt = args.lambda_nbt
lambda_db = args.lambda_db
lambda_p = args.lambda_p
Nres = args.Nres
Nsamples = args.Nsamples
nIter = args.nIter
model_load = args.data_load
method = args.method
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

path_f   = f'2D_inverse_Nk_{Nk}_Nh_{Nh}_Nres_{Nres}_nIter_{nIter}_sigma_{sigma}_Nsamples_{Nsamples}_{method}_measurenorm'
path_fig = os.path.join(path_f,'figures')
if not os.path.exists(path_f):
    os.makedirs(path_f)
if not os.path.exists(path_fig):
    os.makedirs(path_fig)
f_rec = open(os.path.join(path_f,'record.out'), 'a+')

print(f'method:{method} rand_seed:{rand_seed} nIter:{nIter}', file = f_rec)
print(f'layers_k:{layers_k} layers_h:{layers_h}', file = f_rec)
print(f'Nk:{Nk} Nh:{Nh} Nres:{Nres}\n', file = f_rec)
print(f'sigma:{sigma} lambda_r:{lambda_r} lambda_nbl:{lambda_nbl} lambda_nbb:{lambda_nbb} lambda_nbt:{lambda_nbt} lambda_db:{lambda_db} lambda_p:{lambda_p}\n')
print(f'sigma:{sigma} lambda_r:{lambda_r} lambda_nbl:{lambda_nbl} lambda_nbb:{lambda_nbb} lambda_nbt:{lambda_nbt} lambda_db:{lambda_db} lambda_p:{lambda_p}\n', file = f_rec)
lambda_r = lambda_r*num_params/res.shape[0]
lambda_d = lambda_d*num_params/k_data.shape[0]
lambda_nbl = lambda_nbl*num_params/nbl.shape[0]
lambda_nbb = lambda_nbb*num_params/nbb.shape[0]
lambda_nbt = lambda_nbt*num_params/nbt.shape[0]
lambda_db = lambda_db*num_params/dbr.shape[0]
lambda_p = lambda_p
print(f'Normalized PINN weights: lambda_r:{lambda_r} lambda_nbl:{lambda_nbl} lambda_nbb:{lambda_nbb} lambda_nbt:{lambda_nbt} lambda_db:{lambda_db} lambda_p:{lambda_p}\n')
print(f'Normalized PINN weights: lambda_r:{lambda_r} lambda_nbl:{lambda_nbl} lambda_nbb:{lambda_nbb} lambda_nbt:{lambda_nbt} lambda_db:{lambda_db} lambda_p:{lambda_p}\n', file = f_rec)

rl2e = lambda yest, yref : spl.norm(yest - yref, 2) / spl.norm(yref, 2) 
infe = lambda yest, yref : spl.norm(yest - yref, np.inf) 
lpp = lambda h, href, sigma: np.sum( -(h - href)**2/(2*sigma**2) - 1/2*np.log( 2*np.pi) - 2*np.log(sigma))

def pcolormesh(XX, YY, Z, points = None, title = None, savefig = None, cmap='jet'):
    fig, ax = plt.subplots(dpi = 300, figsize = (6,4))
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
    if savefig is not None:
        plt.savefig(os.path.join(path_fig,f'{savefig}.png'))
    plt.show()

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
    def __init__(self, key, layers_k, layers_h, dataset, lbt, ubt, sigma, lambda_r, lambda_d, 
                 lambda_nbl, lambda_nbb, lambda_nbt, lambda_db, lambda_p):  

        self.lbt = lbt #domain lower corner
        self.ubt = ubt #domain upper corner
        self.sigma = sigma
        self.lambda_r = lambda_r 
        self.lambda_d = lambda_d
        self.lambda_nbl = lambda_nbl 
        self.lambda_nbb = lambda_nbb 
        self.lambda_nbt = lambda_nbt 
        self.lambda_db = lambda_db 
        self.lambda_p = lambda_p 
        
        self.gamma = sigma**2*lambda_d
        self.sigma_r = jnp.sqrt(self.gamma/self.lambda_r)
        self.sigma_d = jnp.sqrt(self.gamma/self.lambda_d)
        self.sigma_nbl = jnp.sqrt(self.gamma/self.lambda_nbl)
        self.sigma_nbb = jnp.sqrt(self.gamma/self.lambda_nbb)
        self.sigma_nbt = jnp.sqrt(self.gamma/self.lambda_nbt)
        self.sigma_db = jnp.sqrt(self.gamma/self.lambda_db)
        self.sigma_p = jnp.sqrt(self.gamma)
        
        # Prepare normalized training data
        self.dataset = dataset
        self.x_res, self.y_res = dataset['res'][:,0:2], dataset['res'][:,2:3]
        self.x_dbr, self.y_dbr = dataset['dbr'][:,0:2], dataset['dbr'][:,2:3]
        self.x_nbl, self.y_nbl = dataset['nbl'][:,0:2], dataset['nbl'][:,2:3]
        self.x_nbt, self.y_nbt = dataset['nbt'][:,0:2], dataset['nbt'][:,2:3]
        self.x_nbb, self.y_nbb = dataset['nbb'][:,0:2], dataset['nbb'][:,2:3]
        self.x_h, self.y_h = dataset['h_data'][:,0:2], dataset['h_data'][:,2:3]
        self.x_k, self.y_k = dataset['k_data'][:,0:2], dataset['k_data'][:,2:3]
        self.y_nb = np.hstack((self.y_nbl.ravel(), self.y_nbt.ravel(), self.y_nbb.ravel()))
        
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
        self.itercount = itertools.count()
        lr = optimizers.exponential_decay(1e-4, decay_steps=5000, decay_rate=0.9)
        self.opt_init, \
        self.opt_update, \
        self.get_params = optimizers.adam(lr)
          
        self.opt_state_k = self.opt_init(self.params_k)
        self.opt_state_h = self.opt_init(self.params_h)
        
        #Define random noise distributions
        #for h measurements
        self.h_dist = tfd.Independent(tfd.Normal(loc= jnp.zeros_like(self.y_h.ravel()),
                                   scale= self.sigma_d*jnp.ones_like(self.y_h.ravel())), reinterpreted_batch_ndims = 1)
        #for k measurements
        self.k_dist = tfd.Independent(tfd.Normal(loc= jnp.zeros_like(self.y_k.ravel()),
                                   scale= self.sigma_d*jnp.ones_like(self.y_k.ravel())), reinterpreted_batch_ndims = 1)
        # for residual term
        self.r_dist = tfd.Independent(tfd.Normal(loc= jnp.zeros_like(self.y_res.ravel()),
                                   scale= self.sigma_r*jnp.ones_like(self.y_res.ravel())), reinterpreted_batch_ndims = 1)
        #for bd measurements
        self.db_dist = tfd.Independent(tfd.Normal(loc= jnp.zeros_like(self.y_dbr.ravel()),
                                   scale= self.sigma_db*jnp.ones_like(self.y_dbr.ravel())), reinterpreted_batch_ndims = 1)
        # for Neumann boundary term
        self.nbl_dist = tfd.Independent(tfd.Normal(loc= jnp.zeros_like(self.y_nbl.ravel()),
                                   scale= self.sigma_nbl*jnp.ones_like(self.y_nbl.ravel())), reinterpreted_batch_ndims = 1)
        self.nbb_dist = tfd.Independent(tfd.Normal(loc= jnp.zeros_like(self.y_nbb.ravel()),
                                   scale= self.sigma_nbb*jnp.ones_like(self.y_nbb.ravel())), reinterpreted_batch_ndims = 1)
        self.nbt_dist = tfd.Independent(tfd.Normal(loc= jnp.zeros_like(self.y_nbt.ravel()),
                                   scale= self.sigma_nbt*jnp.ones_like(self.y_nbt.ravel())), reinterpreted_batch_ndims = 1)
        # for regularization term
        self.p_dist = tfd.Independent(tfd.Normal(loc= jnp.zeros((self.num_params,)),
                                   scale= self.sigma_p*jnp.ones((self.num_params,))), reinterpreted_batch_ndims = 1)
        
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
    
    #loss function
    @partial(jit, static_argnums=(0,))
    def loss_r(self, params, r_noise): 
        r_pred = vmap(self.res_net, (None, 0, 0))(params, self.x_res[:,0], self.x_res[:,1])
        loss_res = jnp.sum((r_pred.flatten() - self.y_res.flatten() - r_noise)**2)
        return loss_res  
  
    @partial(jit, static_argnums=(0,))
    def loss_k(self, params, k_noise): 
        k_pred = vmap(self.k_net, (None, 0, 0))(params, self.x_k[:,0], self.x_k[:,1])
        loss_k = jnp.sum((k_pred.flatten() - self.y_k.flatten() - k_noise)**2)
        return loss_k
  
    @partial(jit, static_argnums=(0,))
    def loss_h(self, params, h_noise): 
        h_pred = vmap(self.h_net, (None, 0, 0 ))(params, self.x_h[:,0], self.x_h[:,1])
        loss_h = jnp.sum((h_pred.flatten() - self.y_h.flatten()  - h_noise)**2)
        return loss_h
  
    @partial(jit, static_argnums=(0,))
    def loss_db(self, params, db_noise):
        h_pred = vmap(self.h_net, (None, 0, 0))(params, self.x_dbr[:,0], self.x_dbr[:,1])
        loss_db = jnp.sum((self.y_dbr.flatten() - h_pred.flatten() -  db_noise)**2)
        return loss_db

    @partial(jit, static_argnums=(0,))
    def loss_nbl(self, params, nbl_noise):
        loss_nbl = jnp.sum((vmap(self.qx, (None, 0, 0))(params, self.x_nbl[:,0], self.x_nbl[:,1]).flatten() - self.y_nbl.flatten() - nbl_noise)**2)
        return loss_nbl 
    
    @partial(jit, static_argnums=(0,))
    def loss_nbb(self, params, nbb_noise):
        loss_nbb = jnp.sum((vmap(self.qy, (None, 0, 0))(params, self.x_nbb[:,0], self.x_nbb[:,1]).flatten() - self.y_nbb.flatten() - nbb_noise)**2)
        return loss_nbb
    
    @partial(jit, static_argnums=(0,))
    def loss_nbt(self, params, nbt_noise):
        loss_nbt = jnp.sum((vmap(self.qy, (None, 0, 0))(params, self.x_nbt[:,0], self.x_nbt[:,1]).flatten() - self.y_nbt.flatten() - nbt_noise)**2)
        return loss_nbt
  
    @partial(jit, static_argnums=(0,))
    def l2_reg(self, params, p_noise):
        return jnp.sum((ravel_pytree(params)[0] - p_noise)**2)
    
    @partial(jit, static_argnums=(0,))  
    def loss(self, params, k_noise, h_noise, r_noise, db_noise, nbl_noise, nbb_noise, nbt_noise, p_noise):
        return 1/self.sigma_r**2*self.loss_r(params, r_noise)  + \
                  1/self.sigma_d**2*self.loss_k(params, k_noise) + 1/self.sigma_d**2*self.loss_h(params, h_noise) +\
                       1/self.sigma_db**2*self.loss_db(params, db_noise) + \
                           1/self.sigma_nbl**2*self.loss_nbl(params, nbl_noise) + 1/self.sigma_nbb**2*self.loss_nbb(params, nbb_noise) + 1/self.sigma_nbt**2*self.loss_nbt(params, nbt_noise) +\
                               1/self.sigma_p**2*self.l2_reg(params, p_noise) 

    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state_k, opt_state_h,  k_noise, h_noise, r_noise, db_noise, nbl_noise, nbb_noise, nbt_noise, p_noise):
        params_k = self.get_params(opt_state_k)
        params_h = self.get_params(opt_state_h)
        params = [params_k, params_h]
        g = grad(self.loss)(params,  k_noise, h_noise, r_noise, db_noise, nbl_noise, nbb_noise, nbt_noise, p_noise)

        return self.opt_update(i, g[0], opt_state_k), self.opt_update(i, g[1], opt_state_h)
    
    # def train(self, nIter, num_print, k_noise, h_noise, r_noise, db_noise, nbl_noise, nbb_noise, nbt_noise, p_noise):
    #     pbar = trange(nIter)
    #     # Main training loop
    #     for it in pbar:
    #         self.current_count = next(self.itercount)
    #         self.opt_state_k,  self.opt_state_h = self.step(self.current_count, self.opt_state_k, self.opt_state_h, \
    #                                                         k_noise, h_noise, r_noise, db_noise, nbl_noise, nbb_noise, nbt_noise, p_noise)
    #         if it % num_print == 0:
    #             params_k = self.get_params(self.opt_state_k)
    #             params_h = self.get_params(self.opt_state_h)
    #             params = [params_k, params_h]
                    
    #             loss_value = self.loss(params,k_noise, h_noise, r_noise, db_noise, nbl_noise, nbb_noise, nbt_noise, p_noise)
    #             loss_r_value = self.loss_r(params, r_noise)
    #             loss_k_value = self.loss_k(params, k_noise)
    #             loss_h_value = self.loss_h(params, h_noise)
                
    #             #loss_reg_value = self.l2_reg(params[0]) + self.l2_reg(params[1])

    #             pbar.set_postfix({'Loss': loss_value, 
    #                       'Loss_r': loss_r_value,
    #                       'Loss_k': loss_k_value,
    #                       'Loss_h': loss_h_value
    #                       })
                
    #     return [self.get_params(self.opt_state_k), self.get_params(self.opt_state_h)]
    
    def train(self, nIter, num_print, k_noise, h_noise, r_noise, db_noise, nbl_noise, nbb_noise, nbt_noise, p_noise):
        pbar = trange(nIter)
        # Main training loop
        for it in pbar:
            self.current_count = next(self.itercount)
            self.opt_state_k,  self.opt_state_h = self.step(self.current_count, self.opt_state_k, self.opt_state_h, \
                                                            k_noise, h_noise, r_noise, db_noise, nbl_noise, nbb_noise, nbt_noise, p_noise)
            if it % num_print == 0:
                params_k = self.get_params(self.opt_state_k)
                params_h = self.get_params(self.opt_state_h)
                params = [params_k, params_h]
                    
                loss_value = self.loss(params,k_noise, h_noise, r_noise, db_noise, nbl_noise, nbb_noise, nbt_noise, p_noise)

                pbar.set_postfix({'Loss': loss_value})
                
        return [self.get_params(self.opt_state_k), self.get_params(self.opt_state_h)]
    
    def rpinn_sample(self, Nsample, nIter, num_print, key):
        #sample with randomized PINN
        params_sample = []
        # alpha_sample = []
        # beta_sample = []
        # omega_sample = []
    
        for it in range(Nsample):
            key, *keys = random.split(key, 11)
            #key, *keys = random.split(key, 4)
            k_noise = self.k_dist.sample(1, keys[0])[0]
            h_noise = self.h_dist.sample(1, keys[1])[0]
            r_noise = self.r_dist.sample(1, keys[2])[0]
            db_noise = self.db_dist.sample(1, keys[3])[0]
            nbl_noise = self.nbl_dist.sample(1, keys[4])[0]
            nbb_noise = self.nbb_dist.sample(1, keys[5])[0]
            nbt_noise = self.nbt_dist.sample(1, keys[6])[0]
            p_noise = self.p_dist.sample(1, keys[7])[0]
      
            lr = optimizers.exponential_decay(1e-4, decay_steps=5000, decay_rate=0.9)
            self.opt_init, \
            self.opt_update, \
            self.get_params = optimizers.adam(lr)
            self.params_k = self.init_k(keys[8])
            self.params_h = self.init_k(keys[9])
            self.opt_state_k = self.opt_init(self.params_k)
            self.opt_state_h = self.opt_init(self.params_h)
            
            self.itercount = itertools.count()
            params = self.train(nIter, num_print, k_noise, h_noise, r_noise, db_noise, nbl_noise, nbb_noise, nbt_noise, p_noise)

            params_sample.append(ravel_pytree(params)[0])
            print(f'{it}-th sample finished')
        return jnp.array(params_sample)
    
    def de_sample(self, Nsample, nIter, num_print, key):
        #sample with deep ensemble
        params_sample = []
    
        for it in range(Nsample):
            key, *keys = random.split(key, 3)
            k_noise = 0
            h_noise = 0
            r_noise = 0
            db_noise = 0
            nbl_noise = 0
            nbb_noise = 0
            nbt_noise = 0
            p_noise = 0
      
            lr = optimizers.exponential_decay(1e-4, decay_steps=5000, decay_rate=0.9)
            self.opt_init, \
            self.opt_update, \
            self.get_params = optimizers.adam(lr)
            self.params_k = self.init_k(keys[0])
            self.params_h = self.init_k(keys[1])
            self.opt_state_k = self.opt_init(self.params_k)
            self.opt_state_h = self.opt_init(self.params_h)
            
            self.itercount = itertools.count()
            params = self.train(nIter, num_print, k_noise, h_noise, r_noise, db_noise, nbl_noise, nbb_noise, nbt_noise, p_noise)

            params_sample.append(ravel_pytree(params)[0])
            print(f'{it}-th sample finished')
        return jnp.array(params_sample)


key = random.PRNGKey(rand_seed)
key, subkey = random.split(key, 2)
pinn = PINN(key, layers_k, layers_h, dataset, lbt, ubt, sigma, lambda_r, lambda_d, 
             lambda_nbl, lambda_nbb, lambda_nbt, lambda_db, lambda_p)
if model_load == False:
    if method == 'rPINN':
        samples = pinn.rpinn_sample(Nsamples, nIter = nIter, 
                                              num_print = num_print, key = subkey)
    elif method == 'rPINN_metro':
        samples = pinn.rpinn_sample_metro(Nsamples, nIter = nIter, 
                                              num_print = num_print, key = subkey)
    else:
        samples = pinn.de_sample(Nsamples, nIter = nIter, num_print = num_print, key = subkey)
    
    np.savetxt(os.path.join(path_f,f'{method}_posterior_samples_Nres_{Nres}_sigma_{sigma}_Nsamples_{Nsamples}_nIter_{nIter}.out'), samples)
else:
    samples = np.loadtxt(os.path.join(path_f,f'{method}_posterior_samples_Nres_{Nres}_sigma_{sigma}_Nsamples_{Nsamples}_nIter_{nIter}.out'))

print(f'sigma_r:{pinn.sigma_r} sigma_d:{pinn.sigma_d} sigma_nbl:{pinn.sigma_nbl} sigma_nbb:{pinn.sigma_nbb} sigma_nbt:{pinn.sigma_nbt} sigma_db:{pinn.sigma_db} sigma_p:{pinn.sigma_p}\n')
print(f'sigma_r:{pinn.sigma_r} sigma_d:{pinn.sigma_d} sigma_nbl:{pinn.sigma_nbl} sigma_nbb:{pinn.sigma_nbb} sigma_nbt:{pinn.sigma_nbt} sigma_db:{pinn.sigma_db} sigma_p:{pinn.sigma_p}', file = f_rec)

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

# h_pred_ens = np.array([vmap(get_h_pred)(samples[i,:,:]) for i in range(samples.shape[0])]) #(Nchains, Nsamples, 32768)
# k_pred_ens = np.array([vmap(get_k_pred)(samples[i,:,:]) for i in range(samples.shape[0])]) 

h_pred_ens = vmap(get_h_pred)(samples[::,:]) 
k_pred_ens = vmap(get_k_pred)(samples[::,:]) 

h_pred_ens_mean = np.mean(h_pred_ens, axis = 0) #(Nchains, 32768)
h_pred_ens_std = np.std(h_pred_ens, axis = 0)
k_pred_ens_mean = np.mean(k_pred_ens, axis = 0)
k_pred_ens_std = np.std(k_pred_ens, axis = 0)

h_env = np.logical_and( (h_pred_ens_mean < h_ref.ravel() + 2*h_pred_ens_std), (h_pred_ens_mean > h_ref.ravel() - 2*h_pred_ens_std) )
k_env = np.logical_and( (k_pred_ens_mean < k_ref.ravel() + 2*k_pred_ens_std), (k_pred_ens_mean > k_ref.ravel() - 2*k_pred_ens_std) )

rl2e_h = rl2e(h_pred_ens_mean, h_ref)
infe_h = infe(h_pred_ens_mean, h_ref)
lpp_h = lpp(h_pred_ens_mean, h_ref, h_pred_ens_std)
rl2e_k = rl2e(k_pred_ens_mean, k_ref)
infe_k = infe(k_pred_ens_mean, k_ref)
lpp_k = lpp(k_pred_ens_mean, k_ref, k_pred_ens_std)

print('h prediction:\n')
print('Relative RL2 error: {}'.format(rl2e_h))
print('Absolute inf error: {}'.format(infe_h))
print('Average standard deviation: {}'.format(np.mean(h_pred_ens_std)))
print('log predictive probability: {}'.format(lpp_h))
print('Percentage of coverage:{}\n'.format(np.sum(h_env)/32768))

print('k prediction:\n')
print('Relative RL2 error: {}'.format(rl2e_k))
print('Absolute inf error: {}'.format(infe_k))
print('Average standard deviation: {}'.format(np.mean(k_pred_ens_std)))
print('log predictive probability: {}'.format(lpp_k))
print('Percentage of coverage:{}\n'.format(np.sum(k_env)/32768))

print('h prediction:\n', file = f_rec)
print('Relative RL2 error: {}'.format(rl2e_h), file = f_rec)
print('Absolute inf error: {}'.format(infe_h), file = f_rec)
print('Average standard deviation: {}'.format(np.mean(h_pred_ens_std)), file = f_rec)
print('log predictive probability: {}'.format(lpp_h), file = f_rec)
print('Percentage of coverage:{}\n'.format(np.sum(h_env)/32768), file = f_rec)

print('k prediction:\n', file = f_rec)
print('Relative RL2 error: {}'.format(rl2e_k), file = f_rec)
print('Absolute inf error: {}'.format(infe_k), file = f_rec)
print('Average standard deviation: {}'.format(np.mean(k_pred_ens_std)), file = f_rec)
print('log predictive probability: {}'.format(lpp_k), file = f_rec)
print('Percentage of coverage:{}\n'.format(np.sum(k_env)/32768), file = f_rec)

#Plot of k field
pcolormesh(XX, YY, k_ref.reshape(128,256), points = None, title = None, savefig = 'y_ref')
savefig = f'2D_gwf_rPINN_sigma_{sigma}_Nsamples_{Nsamples}_ypred_mean'
pcolormesh(XX, YY, k_pred_ens_mean.reshape(128,256), points = None, title = None, savefig = savefig)
savefig = f'2D_gwf_rPINN_sigma_{sigma}_Nsamples_{Nsamples}_ypred_std'
pcolormesh(XX, YY, k_pred_ens_std.reshape(128,256), points = x_k, title = None, savefig = savefig)
savefig = f'2D_gwf_rPINN_sigma_{sigma}_Nsamples_{Nsamples}_ypred_diff'
pcolormesh(XX, YY, np.abs(k_pred_ens_mean.reshape(128,256) - k_ref.reshape(128,256)), points = x_k, title = None, savefig = savefig)
savefig = f'2D_gwf_rPINN_sigma_{sigma}_Nsamples_{Nsamples}_ypred_env'
pcolormesh(XX, YY, k_env.reshape(128,256), points = x_k, title = None, savefig = savefig)

#Plot of h field
pcolormesh(XX, YY, h_ref.reshape(128,256), points = None, title = None, savefig = 'h_ref')
savefig = f'2D_gwf_rPINN_sigma_{sigma}_Nsamples_{Nsamples}_hpred_mean'
pcolormesh(XX, YY, h_pred_ens_mean.reshape(128,256), points = None, title = None, savefig = savefig )
savefig = f'2D_gwf_rPINN_sigma_{sigma}_Nsamples_{Nsamples}_hpred_std'
pcolormesh(XX, YY, h_pred_ens_std.reshape(128,256), points = x_h, title = None, savefig = savefig )
savefig = f'2D_gwf_rPINN_sigma_{sigma}_Nsamples_{Nsamples}_hpred_diff'
pcolormesh(XX, YY, np.abs(h_pred_ens_mean.reshape(128,256) - h_ref.reshape(128,256)), points = x_h, title = None, savefig = savefig )
savefig = f'2D_gwf_rPINN_sigma_{sigma}_Nsamples_{Nsamples}_hpred_env'
pcolormesh(XX, YY, h_env.reshape(128,256), points = x_h, title = None, savefig = savefig )

f_rec.close()