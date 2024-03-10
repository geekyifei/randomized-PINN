#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 15:11:25 2024

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
from scipy.spatial.distance import pdist, squareform
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
    default=8888,
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
    default=0.1118034,
    help="Aleotoric uncertainty to the data")
parser.add_argument(
    "--sigma_p",
    type=float,
    default=4.107919,
    help="Prior std")
parser.add_argument(
    "--Nres",
    type=int,
    default=32,
    help="Number of reisudal points")
parser.add_argument(
    "--Nsamples",
    type=int,
    default=1000,
    help="Number of Posterior samples")
parser.add_argument(
    "--nIter",
    type=int,
    default=50000,
    help="Number of Posterior samples")
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
sigma_d = args.sigma_d
sigma_p = args.sigma_p
# sigma = 0.01
# sigma_r = 0.01
# sigma_d = 0.01118
# sigma_p = 0.41079
Nsamples = args.Nsamples
nIter = args.nIter
num_print = 20
bandwidth = -1
path_f   = f'1D_nonlinear_poisson_SVGD_Nres_{Nres}_sigma_{sigma}_Nsamples_{Nsamples}_nIter_{nIter}_bandwidth_{bandwidth}'
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
    def __init__(self, key, layers, dataset, lbt, ubt, lamb, k, sigma_r, sigma_d, sigma_p): 

        self.lbt = lbt #domain lower corner
        self.ubt = ubt #domain upper corner
        self.k = k
        self.lamb = lamb
        self.scale_coe = 0.5
        self.scale = 2 * self.scale_coe / (self.ubt - self.lbt)
        self.sigma_r = sigma_r
        self.sigma_d = sigma_d
        self.sigma_p = sigma_p
    
        # Prepare normalized training data
        self.dataset = dataset
        self.X_res, self.y_res = self.normalize(dataset['res'][:,0:1]), dataset['res'][:,1:2]
        self.X_data, self.y_data = self.normalize(dataset['data'][:,0:1]), dataset['data'][:,1:2]
        
        # Initalize the network
        self.init, self.forward = FNN(layers, activation=jnp.tanh)
        self.params = self.init(key)
        raveled_params, self.unravel = ravel_pytree(self.params)
        self.num_params = raveled_params.shape[0]
        
        self.itercount = itertools.count()
        self.log_prob_log = []
        self.u_rl2e_log = []
        self.u_lpp_log = []
        self.f_rl2e_log = []
        self.f_lpp_log = []
                
        # Evaluate the network and the residual over the grid
        self.u_pred_map = vmap(self.predict_u, (None, 0)) 
        self.f_pred_map = vmap(self.predict_f, (None, 0))  
        self.rl2e = lambda yest, yref : spl.norm(yest - yref, 2) / spl.norm(yref, 2) 
        self.lpp = lambda h, href, sigma: np.sum(-(h - href)**2/(2*sigma**2) - 1/2*np.log( 2*np.pi) - 2*np.log(sigma))
        
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
    def u_pred_vector(self, params):
        u_pred_vec = vmap(self.u_net, (None, 0))(self.unravel(params), self.X_data[:,0])
        return u_pred_vec

    @partial(jit, static_argnums=(0,))
    def f_pred_vector(self, params):
        f_pred_vec = vmap(self.res_net, (None, 0))(self.unravel(params), self.X_res[:,0])
        return f_pred_vec  
    
    @partial(jit, static_argnums=(0,))
    def target_log_prob(self, theta):
        prior = jnp.sum(-(theta)**2/(2*self.sigma_p**2))
        r_likelihood = jnp.sum(-(y_r.ravel() - self.f_pred_vector(theta))**2/(2*self.sigma_r**2))
        u_likelihood = jnp.sum(-(y_data.ravel() - self.u_pred_vector(theta))**2/(2*self.sigma_d**2))
        return prior + r_likelihood + u_likelihood

    @partial(jit, static_argnums=(0,))
    def grad_log_prob(self, theta):
        return jax.value_and_grad(self.target_log_prob, argnums = 0)(theta)[1]
    
    def median_trick_h(self, theta):
        '''
        The scipy one seems even faster and memory efficient
        
        '''
        sq_dist = pdist(theta)
        pairwise_dists = squareform(sq_dist)
        h = np.median(pairwise_dists)**2  
        h = np.sqrt(0.5 * h / np.log(theta.shape[0]+1))
        return h

    @partial(jit, static_argnums=(0,))
    def rbf_kernel(self, theta1, theta2, h):
        '''
        Evaluate the rbf kernel k(x, x') = exp(-|x - x'|^2/(2h^2))
        input: theta1, theta2 are 1d array of parameters, 
                h is correlation length
        output: a scalar value of kernel evaluation 
        '''
        # here theta1 and theta2 are 1d-array of parameters
        return jnp.exp(-((theta1 - theta2)**2).sum(axis=-1) / (2 * h**2))
    
    @partial(jit, static_argnums=(0,))
    def compute_kernel_matrix(self, theta, h):
        return vmap(vmap(lambda x, y: self.rbf_kernel(x, y, h), in_axes=(None, 0)), in_axes=(0, None))(theta, theta)
    
    @partial(jit, static_argnums=(0,))
    def kernel_and_grad(self, theta, h):
        '''
        input theta: (Nsamples, Nparams)
                h is correlation length
        output: K: #(Nsamples, Nsamples)
                grad_K: #(Nsamples, Nparams)
        '''
        K = self.compute_kernel_matrix(theta, h) #(Nsamples, Nsamples)
        grad_K = jnp.sum(jnp.einsum('ijk,ij->ijk', theta - theta[:, None, :], K), axis = 0)/ (h**2) 
        return (K, grad_K)
    
    @partial(jit, static_argnums=(0,))
    def svgd_step(self, i, opt_state, h):
        theta = self.get_params(opt_state)
        grad_logprob = vmap(self.grad_log_prob)(theta)
        K, grad_K = self.kernel_and_grad(theta, h)
        phi = -(jnp.einsum('ij, jk->ik', K, grad_logprob)/theta.shape[0] + grad_K) #(Nsamples, Nparams)
        return self.opt_update(i, phi, opt_state)
     
    def svgd_train(self, key, Nsamples, nIter, num_print, bandwidth, u_ref, f_ref):
        
        new_key, subkey = random.split(key, 2)
        init_state = random.normal(subkey , (Nsamples, self.num_params))
        
        x_test = jnp.linspace(-0.7,0.7,101)
        u_pred = vmap(lambda sample: self.u_pred_map(self.unravel(sample),x_test))
        f_pred = vmap(lambda sample: self.f_pred_map(self.unravel(sample),x_test))
        u = u_ref(x_test)
        f = f_ref(x_test)
            
        lr = optimizers.exponential_decay(1e-4, decay_steps=1000, decay_rate=0.9)
        #lr = 1e-4
        self.opt_init, \
        self.opt_update, \
        self.get_params = optimizers.adam(lr)
        self.opt_state = self.opt_init(init_state)
        
        ts = perf_counter()
        pbar = trange(nIter)
        
        for it in pbar:
            self.current_count = next(self.itercount)
            theta = self.get_params(self.opt_state)
            h = bandwidth if bandwidth > 0 else self.median_trick_h(theta) 
            self.opt_state = self.svgd_step(self.current_count, self.opt_state, h)
            
            if it % num_print == 0:
    
                log_prob = jnp.mean(vmap(self.target_log_prob)(theta))
                u_test_ens = u_pred(theta)
                f_test_ens = f_pred(theta)
                u_test_mean = jnp.mean(u_test_ens, axis = 0)
                u_test_std = jnp.std(u_test_ens, axis = 0)
                rl2e_u = self.rl2e(u_test_mean, u)
                lpp_u = self.lpp(u_test_mean, u, u_test_std)
                
                f_test_mean = jnp.mean(f_test_ens, axis = 0)
                f_test_std = jnp.std(f_test_ens, axis = 0)
                rl2e_f = self.rl2e(f_test_mean, f)
                lpp_f = self.lpp(f_test_mean, f, f_test_std)
                
                self.log_prob_log.append(log_prob)
                self.u_rl2e_log.append(rl2e_u)
                self.u_lpp_log.append(lpp_u)
                self.f_rl2e_log.append(rl2e_f)
                self.f_lpp_log.append(lpp_f)
                
                pbar.set_postfix({'Log prob': log_prob, 
                          'u_rl2e': rl2e_u,
                          'u_lpp':lpp_u,
                          'f_rl2e': rl2e_f,
                          'f_lpp': lpp_f})
        
        timings = perf_counter() - ts
        print(f"SVGD: {timings} s")
        return self.get_params(self.opt_state)

key1, key2 = random.split(random.PRNGKey(0), 2)
model = PINN(key2, layers_u, dataset, lbt, ubt, lamb, k, sigma_r, sigma_d, sigma_p)
ts = perf_counter()
samples = model.svgd_train(key2, Nsamples, nIter = nIter, num_print = num_print, bandwidth = bandwidth, u_ref = u, f_ref = f)
timings = perf_counter() - ts
print(f"SVGD: {timings} s")
print(f"SVGD: {timings} s", file = f_rec)
np.savetxt(os.path.join(path_f,f'SVGD_samples_Nres_{Nres}_sigma_{sigma}_Nsamples_{Nsamples}_nIter_{nIter}.out'), samples)

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

f_rec.close()

# =============================================================================
# Plot posterior predictions
# =============================================================================

fig, ax = plt.subplots(dpi = 300, figsize = (4,4))
ax.plot(x_pred_index, u_ref, 'k-', label='Exact')
ax.plot(x_pred_index, u_pred_ens_mean, markersize = 1, markevery=2, markerfacecolor='None', label= 'SVGD mean', alpha = 0.8)
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
plt.savefig(os.path.join(path_fig,f'1D_nonlinear_poisson_SVGD_Nres_{Nres}_sigma_{sigma}_Nsamples_{Nsamples}_nIter_{nIter}_upred.png'))
plt.show()

fig, ax = plt.subplots(dpi = 300, figsize = (4,4))
ax.plot(x_pred_index, f_ref, 'k-', label='Exact')
ax.plot(x_pred_index, f_pred_ens_mean, markersize = 1, markevery=2, markerfacecolor='None', label='SVGD mean', alpha = 0.8)
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
plt.savefig(os.path.join(path_fig,f'1D_nonlinear_poisson_SVGD_Nres_{Nres}_sigma_{sigma}_Nsamples_{Nsamples}_nIter_{nIter}_fpred.png'))
plt.show()

# Log prob plot
t = np.arange(0, nIter, num_print)
fig = plt.figure(constrained_layout=False, figsize=(4, 4), dpi = 300)
ax = fig.add_subplot()
ax.plot(t, -np.array(model.log_prob_log), color='blue', label='Negative Log prob')
ax.set_yscale('log')
ax.set_ylabel('Loss',  fontsize = 16)
ax.set_xlabel('Epochs', fontsize = 16)
ax.legend(loc='upper right', fontsize = 14)
fig.tight_layout()
fig.savefig(os.path.join(path_fig,'loss.png'))

t = np.arange(0, nIter, num_print)
fig = plt.figure(constrained_layout=False, figsize=(4, 4), dpi = 300)
ax = fig.add_subplot()
ax.plot(t, np.array(model.u_rl2e_log), label='u')
ax.plot(t, np.array(model.f_rl2e_log), label='f')
ax.set_ylabel('relative L2 error',  fontsize = 16)
ax.set_xlabel('Epochs', fontsize = 16)
ax.legend(loc='upper right', fontsize = 14)
fig.tight_layout()
fig.savefig(os.path.join(path_fig,'test_rl2e.png'))

fig, ax = plt.subplots(dpi = 300, figsize = (4,4))
ax.plot(x_pred_index, u_ref, 'k-', label='Exact')
for i in range(1, 1000, 50):
    ax.plot(x_pred_index, get_u_pred(samples[i]), alpha = 0.5)
ax.fill_between(x_pred_index, u_pred_ens_mean + 2 * u_pred_ens_std, u_pred_ens_mean - 2 * u_pred_ens_std,
                alpha = 0.3, label = r'$95 \% $ CI')
ax.scatter(data[:,0], data[:,1], label='Obs' , s = 20, facecolors='none', edgecolors='b')
ax.set_xlabel('$x$', fontsize=16)
ax.set_xlim(-0.72,0.72)
ax.set_ylim(-2.5,2.5)
ax.set_ylabel('$u(x)$', fontsize=16)
ax.tick_params(axis='both', which = 'major', labelsize=13)
ax.legend(fontsize=10, loc= 'upper left')
fig.tight_layout()
fig.savefig(os.path.join(path_fig,'u_realizations.png'))
plt.show()

fig, ax = plt.subplots(dpi = 300, figsize = (4,4))
ax.plot(x_pred_index, f_ref, 'k-', label='Exact')
for i in range(1, 1000, 50):
    ax.plot(x_pred_index, get_f_pred(samples[i]), alpha = 0.5)
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
fig.savefig(os.path.join(path_fig,'f_realizations.png'))
plt.show()