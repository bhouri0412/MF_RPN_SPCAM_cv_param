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
from functools import partial
import itertools

def exponential_decay_loc(step_size, decay_steps, decay_rate):
  def schedule(i):
    return step_size * decay_rate ** (i / decay_steps)
  return schedule

# Define the model
class EnsembleRegression:
    def __init__(self, layers_H, layers_L, ensemble_size, rng_key = random.PRNGKey(0)):  
        # Network initialization and evaluation functions
        self.init_H, self.apply_H = MLP(layers_H)
        self.init_prior_H, self.apply_prior_H = MLP(layers_H)
        
        self.init_L, self.apply_L = MLP(layers_L)
        self.init_prior_L, self.apply_prior_L = MLP(layers_L)
                
        # Random keys
        k1, k2, k3 = random.split(rng_key, 3)
        keys_1 = random.split(k1, ensemble_size)
        keys_2 = random.split(k2, ensemble_size)
        keys_3 = random.split(k3, ensemble_size)
        
        # Initialize
        params = vmap(self.init_H)(keys_1)
        params_prior = vmap(self.init_prior_H)(keys_2)
        
        rng_key, _ = random.split(rng_key, 2)
        k1, k2, k3 = random.split(rng_key, 3)
        keys_1 = random.split(k1, ensemble_size)
        keys_2 = random.split(k2, ensemble_size)
        keys_3 = random.split(k3, ensemble_size)
        params = params + vmap(self.init_L)(keys_1)
        params_prior = params_prior + vmap(self.init_prior_L)(keys_2)
        
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
        self.loss_log_L = []
        self.loss_log_H = []

    # Define the forward pass
    def net_forward_H(self, params, params_prior, inputs):
        Y_pred = self.apply_H(params, inputs) + self.apply_prior_H(params_prior, inputs) 
        return Y_pred
    def net_forward_L(self, params, params_prior, inputs):
        Y_pred = self.apply_L(params, inputs) + self.apply_prior_L(params_prior, inputs) 
        return Y_pred

    def loss(self, params, params_prior, batch, batch_L):
        inputs, targets = batch
        inputs_L, targets_L = batch_L
        outputs = vmap(self.net_forward_H, (None, None, 0))(params[:len(layers_H)-1], params_prior[:len(layers_H)-1], inputs)
        outputs_L = vmap(self.net_forward_L, (None, None, 0))(params[len(layers_H)-1:], params_prior[len(layers_H)-1:], inputs_L)
        loss = np.mean((targets - outputs)**2) + alpha * np.mean((targets_L - outputs_L)**2)
        return loss
    def loss_H(self, params, params_prior, batch):
        inputs, targets = batch
        outputs = vmap(self.net_forward_H, (None, None, 0))(params, params_prior, inputs)
        loss_H = np.mean((targets - outputs)**2)
        return loss_H
    def loss_L(self, params, params_prior, batch):
        inputs, targets = batch
        outputs = vmap(self.net_forward_L, (None, None, 0))(params, params_prior, inputs)
        loss_L = np.mean((targets - outputs)**2)
        return loss_L
    
    # Define the update step
    def step(self, i, opt_state, prior_opt_state, key_opt_state, batch, batch_L):
        params = self.get_params(opt_state)
        params_prior = self.get_params(prior_opt_state)
        g = grad(self.loss)(params, params_prior, batch, batch_L)
        return self.opt_update(i, g, opt_state)
    
    def monitor_loss(self, opt_state, prior_opt_state, batch, batch_L):
        params = self.get_params(opt_state)
        params_prior = self.get_params(prior_opt_state)
        loss_value = self.loss(params, params_prior, batch, batch_L)
        return loss_value
    def monitor_loss_H(self, opt_state, prior_opt_state, batch):
        params = self.get_params(opt_state)[:len(layers_H)-1]
        params_prior = self.get_params(prior_opt_state)[:len(layers_H)-1]
        loss_value = self.loss_H(params, params_prior, batch)
        return loss_value
    def monitor_loss_L(self, opt_state, prior_opt_state, batch):
        params = self.get_params(opt_state)[len(layers_H)-1:]
        params_prior = self.get_params(prior_opt_state)[len(layers_H)-1:]
        loss_value = self.loss_L(params, params_prior, batch)
        return loss_value
    
    def predict_L(self, x):
        params = self.get_params(self.opt_state)[len(layers_H)-1:]
        params_prior = self.get_params(self.prior_opt_state)[len(layers_H)-1:]
        samples = self.posterior(params, params_prior, x)
        return samples

    # Evaluates predictions at test points  
    @partial(jit, static_argnums=(0,))
    def posterior(self, params, params_prior, inputs):
        samples = vmap(self.net_forward_L, (0, 0, 0))(params, params_prior, inputs)
        return samples
    
    def train(self, nIter = 1000):
        # Define vectorized SGD step across the entire ensemble
        v_step = jit(vmap(self.step, in_axes = (None, 0, 0, 0, 0, 0)))
        v_monitor_loss = jit(vmap(self.monitor_loss, in_axes = (0, 0, 0, 0)))
        v_monitor_loss_H = jit(vmap(self.monitor_loss_H, in_axes = (0, 0, 0)))
        v_monitor_loss_L = jit(vmap(self.monitor_loss_L, in_axes = (0, 0, 0)))

        # Main training loop
        tt = time.time()
        for it in range(nIter):
            id_SF = it%nb_SF
            id_b_MF = it%nb_MF
            pred_L = self.predict_L( (batches_in_SF[0][id_SF][None,:,:]-mu_MF_in)/sigma_MF_in )
            batch = pred_L, (batches_out_SF[0][id_SF][None,:,:]-mu_SF_out)/sigma_SF_out
            
            batch_L = (batches_in_MF[0][id_b_MF][None,:,:]-mu_MF_in)/sigma_MF_in, (batches_out_MF[0][id_b_MF][None,:,:]-mu_MF_out)/sigma_MF_out
            
            self.opt_state = v_step(it, self.opt_state, self.prior_opt_state, self.key_opt_state, batch, batch_L)
            
            if it % nloss == 0:
                loss_value = v_monitor_loss(self.opt_state, self.prior_opt_state, batch, batch_L)
                loss_value_H = v_monitor_loss_H(self.opt_state, self.prior_opt_state, batch)
                loss_value_L = v_monitor_loss_L(self.opt_state, self.prior_opt_state, batch_L)
                self.loss_log.append(loss_value)
                self.loss_log_H.append(loss_value_H)
                self.loss_log_L.append(loss_value_L)
                print(it, nIter, time.time() - tt, n_run_param)
                tt = time.time()
            if (it+1) % nsave == 0:
                params = vmap(self.get_params)(self.opt_state)
                params_H = params[:len(layers_H)-1]
                params_L = params[len(layers_H)-1:]
                for i in range(len(layers_H)-1):
                    for j in range(2):
                        np.save('MF_param/MF_param_'+str(n_run_param)+'/HF_params_'+str(i)+'_'+str(j),params_H[i][j])
                for i in range(len(layers_L)-1):
                    for j in range(2):
                        np.save('MF_param/MF_param_'+str(n_run_param)+'/LF_params_'+str(i)+'_'+str(j),params_L[i][j])
    
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

layers_H = [dim_yL, 512, 512, 512, 512, 512, 512, 512, dim_yH]
layers_L = [dim_xL, 512, 512, 512, 512, 512, 512, 512, dim_yL]

n_run_param = 0
batch_size_MF = 2048
batch_size_SF = 2048
nsave = 50000
nloss = 500
alpha = 1.0

nepoch = 5

mu_SF_out = onp.concatenate((onp.load('norm/mu_y_heat_CAM5.npy')[None,None,ind_output_heat],
                             onp.load('norm/mu_y_moist_CAM5.npy')[None,None,ind_output_moist]),axis=2)
sigma_SF_out = onp.concatenate((onp.load('norm/sigma_y_heat_CAM5.npy')[None,None,ind_output_heat],
                                onp.load('norm/sigma_y_moist_CAM5.npy')[None,None,ind_output_moist]),axis=2)

mu_MF_in = onp.load('norm/mu_X_CAM5.npy')[None,None,ind_input]
sigma_MF_in = onp.load('norm/sigma_X_CAM5.npy')[None,None,ind_input]
mu_MF_out = onp.concatenate((onp.load('norm/mu_y_heat_CAM5.npy')[None,None,ind_output_heat],
                             onp.load('norm/mu_y_moist_CAM5.npy')[None,None,ind_output_moist]),axis=2)
sigma_MF_out = onp.concatenate((onp.load('norm/sigma_y_heat_CAM5.npy')[None,None,ind_output_heat],
                                onp.load('norm/sigma_y_moist_CAM5.npy')[None,None,ind_output_moist]),axis=2)
print('loading data')

train_MF_in = onp.load('data_CAM5_8K/all_inputs.npy')[:,ind_input]
train_MF_out = onp.concatenate((onp.load('data_CAM5_8K/all_outputs_heat.npy')[:,ind_output_heat],
                                onp.load('data_CAM5_8K/all_outputs_moist.npy')[:,ind_output_moist]),axis=1)
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

batches_in_MF = []
batches_out_MF = []
batches_in_SF = []
batches_out_SF = []

for i in range(ensemble_size):
    print(i,ensemble_size)
    
    idx_SF = onp.arange(N_rpn_SF)
    idx_MF = onp.arange(N_tot_MF)
    
    onp.random.seed(n_run_param)
    onp.random.shuffle(idx_SF)

    onp.random.seed(n_run_param)
    onp.random.shuffle(idx_MF)
    
    batches_in_MF_loc = []
    batches_out_MF_loc = []
    batches_in_SF_loc = []
    batches_out_SF_loc = []
    
    for j in range(nb_MF):
        if j % (nb_MF%5000) == 0:
            print(i,'MF',j,nb_MF)
        batches_in_MF_loc.append( train_MF_in[idx_MF[:N_rpn_MF][j*batch_size_MF:(j+1)*batch_size_MF],:] )
        batches_out_MF_loc.append( train_MF_out[idx_MF[:N_rpn_MF][j*batch_size_MF:(j+1)*batch_size_MF],:] )
    for j in range(nb_SF):
        if j % (nb_SF%5000) == 0:
            print(i,'SF',j,nb_SF)
        batches_in_SF_loc.append( train_SF_in[idx_SF[:N_rpn_SF][j*batch_size_SF:(j+1)*batch_size_SF],:] )
        batches_out_SF_loc.append( train_SF_out[idx_SF[:N_rpn_SF][j*batch_size_SF:(j+1)*batch_size_SF],:] )
    batches_in_MF.append(batches_in_MF_loc)
    batches_out_MF.append(batches_out_MF_loc)
    batches_in_SF.append(batches_in_SF_loc)
    batches_out_SF.append(batches_out_SF_loc)

# Initialize model
model = EnsembleRegression(layers_H, layers_L, ensemble_size, rng_key = random.PRNGKey(0))

params_prior = vmap(model.get_params)(model.prior_opt_state)
params_prior_H = params_prior[:len(layers_H)-1]
params_prior_L = params_prior[len(layers_H)-1:]  
print('saving parameters')
for i in range(len(layers_H)-1):
    for j in range(2):
        np.save('MF_param/MF_param_'+str(n_run_param)+'/HF_params_prior_'+str(i)+'_'+str(j),params_prior_H[i][j])
for i in range(len(layers_L)-1):
    for j in range(2):
        np.save('MF_param/MF_param_'+str(n_run_param)+'/LF_params_prior_'+str(i)+'_'+str(j),params_prior_L[i][j])   
print('finished saving')

# Train model
model.train(nIter=nepoch*max(nb_MF,nb_SF))

params = vmap(model.get_params)(model.opt_state)
params_H = params[:len(layers_H)-1]
params_L = params[len(layers_H)-1:]
for i in range(len(layers_H)-1):
    for j in range(2):
        np.save('MF_param/MF_param_'+str(n_run_param)+'/HF_params_'+str(i)+'_'+str(j),params_H[i][j])
for i in range(len(layers_L)-1):
    for j in range(2):
        np.save('MF_param/MF_param_'+str(n_run_param)+'/LF_params_'+str(i)+'_'+str(j),params_L[i][j])
