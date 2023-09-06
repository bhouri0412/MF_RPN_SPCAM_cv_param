#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 17:19:00 2022

@author: mohamedazizbhouri
"""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

from jax import numpy as np
from jax import vmap, jit, random
import numpy as onp

import time

##########################################################
##########################################################
##########################################################

def leakyRELU(x):
    return np.where(x > 0, x, x * 0.15)

def MLP(layers, activation=leakyRELU): # np.tanh
    def init(rng_key):
        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            glorot_stddev = 1. / np.sqrt((d_in + d_out) / 2.)
            W = glorot_stddev*random.normal(k1, (d_in, d_out))
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

##########################################################
##########################################################
##########################################################
    
from jax import grad
from jax.example_libraries import optimizers
import itertools

def exponential_decay_loc(step_size, decay_steps, decay_rate):
  def schedule(i):
    return step_size * decay_rate ** (i / decay_steps)
  return schedule

# Define the model
class EnsembleRegression:
    def __init__(self, layers, ensemble_size, rng_key = random.PRNGKey(0)):  
        # Network initialization and evaluation functions
        self.init, self.apply = MLP(layers)
        self.init_prior, self.apply_prior = MLP(layers)
        
        # Random keys
        k1, k2, k3 = random.split(rng_key, 3)
        keys_1 = random.split(k1, ensemble_size)
        keys_2 = random.split(k2, ensemble_size)
        keys_3 = random.split(k3, ensemble_size)
        
        # Initialize
        params = vmap(self.init)(keys_1)
        params_prior = vmap(self.init_prior)(keys_2)
                
        # Use optimizers to set optimizer initialization and update functions
        lr = optimizers.exponential_decay(1e-4, decay_steps=1000, decay_rate=0.99)
        self.opt_init, \
        self.opt_update, \
        self.get_params = optimizers.adam(lr)

        self.opt_state = vmap(self.opt_init)(params)
        self.prior_opt_state = vmap(self.opt_init)(params_prior)
        self.key_opt_state = vmap(self.opt_init)(keys_3)

        # Logger
        self.itercount = itertools.count()
        self.loss_log = []

    # Define the forward pass
    def net_forward(self, params, params_prior, inputs):
        Y_pred = self.apply(params, inputs) + self.apply_prior(params_prior, inputs) 
        return Y_pred

    def loss(self, params, params_prior, batch):
        inputs, targets = batch
        # Compute forward pass
        outputs = vmap(self.net_forward, (None, None, 0))(params, params_prior, inputs)
        # Compute loss
        loss = np.mean((targets - outputs)**2)
        return loss

    # Define the update step
    def step(self, i, opt_state, prior_opt_state, key_opt_state, batch):
        params = self.get_params(opt_state)
        params_prior = self.get_params(prior_opt_state)
        g = grad(self.loss)(params, params_prior, batch)
        return self.opt_update(i, g, opt_state)
    
    def monitor_loss(self, opt_state, prior_opt_state, batch):
        params = self.get_params(opt_state)
        params_prior = self.get_params(prior_opt_state)
        loss_value = self.loss(params, params_prior, batch)
        return loss_value

    # Optimize parameters in a loop
    def train(self, nIter = 1000):
        # Define vectorized SGD step across the entire ensemble
        v_step = jit(vmap(self.step, in_axes = (None, 0, 0, 0, 0)))
        v_monitor_loss = jit(vmap(self.monitor_loss, in_axes = (0, 0, 0)))

        # Main training loop
        tt = time.time()
        for it in range(nIter):
            id_SF = (it+n_iter_prev)%nb_SF
            
            inputs0_H = batches_in_SF[0][id_SF][None,:,:]
            targets0_H = batches_out_SF[0][id_SF][None,:,:]
            for ii in range(ensemble_size-1):
                inputs0_H = onp.concatenate( (inputs0_H,batches_in_SF[ii+1][id_SF][None,:,:]), axis=0)
                targets0_H = onp.concatenate( (targets0_H,batches_out_SF[ii+1][id_SF][None,:,:]), axis=0)
                        
            batch = (inputs0_H-mu_SF_in)/sigma_SF_in, (targets0_H-mu_SF_out)/sigma_SF_out
            
            self.opt_state = v_step(it, self.opt_state, self.prior_opt_state, self.key_opt_state, batch)
            
            if it % nloss == 0:
                loss_value = v_monitor_loss(self.opt_state, self.prior_opt_state, batch)
                self.loss_log.append(loss_value)
                
                print(it, nIter, time.time() - tt, n_run_param)
                tt = time.time()
            if (it+1) % nsave == 0:
                params = vmap(self.get_params)(self.opt_state)
                for i in range(len(layers_H)-1):
                    for j in range(2):
                        np.save('SF_param/SF_param_'+str(n_run_param)+'/params_'+str(i)+'_'+str(j),params[i][j])
                
##########################################################
##########################################################
##########################################################

n_remove = 4
ind_input = np.concatenate( (np.arange(26),n_remove+26+np.arange(26-n_remove),np.array([52,53,54,55])) ) 
dim_xH = ind_input.shape[0]
dim_xL = ind_input.shape[0]

ind_output_heat = np.arange(26)
ind_output_moist = n_remove+np.arange(26-n_remove)

dim_yH = ind_output_heat.shape[0]+ind_output_moist.shape[0]
dim_yL = ind_output_heat.shape[0]+ind_output_moist.shape[0]

ensemble_size = 1
layers_H = [dim_xH, 512, 512, 512, 512, 512, 512, 512, dim_yH]

n_iter_prev = 0

n_run_param = 0
batch_size_SF = 2048
nsave = 50000
nloss = 500

nepoch = 5 

mu_SF_in = onp.load('norm/mu_X_SPCAM5.npy')[None,None,ind_input]
sigma_SF_in = onp.load('norm/sigma_X_SPCAM5.npy')[None,None,ind_input]

mu_SF_out = onp.concatenate((onp.load('norm/mu_y_heat_SPCAM5.npy')[None,None,ind_output_heat],
                             onp.load('norm/mu_y_moist_SPCAM5.npy')[None,None,ind_output_moist]),axis=2)
sigma_SF_out = onp.concatenate((onp.load('norm/sigma_y_heat_SPCAM5.npy')[None,None,ind_output_heat],
                                onp.load('norm/sigma_y_moist_SPCAM5.npy')[None,None,ind_output_moist]),axis=2)
print('loading data')

train_SF_in = onp.load('data_SPCAM5_hist/three_month_inputs.npy')[:,ind_input]
train_SF_out = onp.concatenate((onp.load('data_SPCAM5_hist/three_month_outputs_heat.npy')[:,ind_output_heat],
                                onp.load('data_SPCAM5_hist/three_month_outputs_moist.npy')[:,ind_output_moist]),axis=1)
print('data loaded')

N_tot_MF = 121098240
N_tot_SF = 29528064

N_rpn_MF = 96878592
N_rpn_SF = 23623680

nb_MF = 47304
nb_SF = 11535

batches_in_SF = []
batches_out_SF = []
for i in range(ensemble_size):
    print(i,ensemble_size)
    
    idx_SF = onp.arange(N_rpn_SF)
    
    onp.random.seed(n_run_param)
    onp.random.shuffle(idx_SF)
    
    batches_in_SF_loc = []
    batches_out_SF_loc = []
    
    for j in range(nb_SF):
        if j % (nb_SF%5000) == 0:
            print(i,'SF',j,nb_SF)
        batches_in_SF_loc.append( train_SF_in[idx_SF[:N_rpn_SF][j*batch_size_SF:(j+1)*batch_size_SF],:] )
        batches_out_SF_loc.append( train_SF_out[idx_SF[:N_rpn_SF][j*batch_size_SF:(j+1)*batch_size_SF],:] )
    batches_in_SF.append(batches_in_SF_loc)
    batches_out_SF.append(batches_out_SF_loc)

model = EnsembleRegression(layers_H, ensemble_size, rng_key = random.PRNGKey(0))#n_run_param))

params_prior = vmap(model.get_params)(model.prior_opt_state)
params_prior_H = params_prior[:len(layers_H)-1]
params_prior_L = params_prior[len(layers_H)-1:]  
print('saving parameters')
for i in range(len(layers_H)-1):
    for j in range(2):
        np.save('SF_param/SF_param_'+str(n_run_param)+'/params_prior_'+str(i)+'_'+str(j),params_prior_H[i][j])
print('finished saving')

# Train model
model.train(nIter=nepoch*max(nb_MF,nb_SF)-n_iter_prev)

params = vmap(model.get_params)(model.opt_state)
params_H = params[:len(layers_H)-1]
for i in range(len(layers_H)-1):
    for j in range(2):
        np.save('SF_param/SF_param_'+str(n_run_param)+'/params_'+str(i)+'_'+str(j),params_H[i][j])
        