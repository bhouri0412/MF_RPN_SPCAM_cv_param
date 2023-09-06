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
    
from jax.example_libraries import optimizers
from functools import partial
import itertools

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
        lr = optimizers.exponential_decay(1e-4, decay_steps=1000, decay_rate=0.999)
        lr = optimizers.exponential_decay(0.000227, decay_steps=1000, decay_rate=0.999)
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
     
    # Evaluates predictions at test points  
    @partial(jit, static_argnums=(0,))
    def posterior(self, params, inputs):
        params, params_prior = params
        samples = vmap(self.net_forward, (0, 0, 0))(params, params_prior, inputs)
        return samples
    
##########################################################
##########################################################
##########################################################

# Helper functions
normalize = lambda x, mu, std: (x-mu)/std

n_remove = 4
ind_input = np.concatenate( (np.arange(26),n_remove+26+np.arange(26-n_remove),np.array([52,53,54,55])) ) 
dim_xH = ind_input.shape[0]
dim_xL = ind_input.shape[0]

ind_output_heat = np.arange(26)
ind_output_moist = n_remove+np.arange(26-n_remove)

dim_yH = ind_output_heat.shape[0]+ind_output_moist.shape[0]
dim_yL = ind_output_heat.shape[0]+ind_output_moist.shape[0]

ensemble_size = 1
i_ensemble = 0
layers_H = [dim_yL, 512, 512, 512, 512, 512, 512, 512, dim_yH]
layers_L = [dim_xL, 512, 512, 512, 512, 512, 512, 512, dim_yL]
    
id_step = 1 # 1 or 2 for instance in case we dont have enough RAM memory to make
            # predictions for the whole test dataset at once.
n_run_param = 0

key = random.PRNGKey(n_run_param)
model_H = EnsembleRegression(layers_H, ensemble_size, key)
model_L = EnsembleRegression(layers_L, ensemble_size, key)

mu_MF_in = onp.load('norm/mu_X_CAM5.npy')[None,ind_input]
sigma_MF_in = onp.load('norm/sigma_X_CAM5.npy')[None,ind_input]

print('loading NN parameters')
params = []
params_prior = []

for i in range(len(layers_H)-1):
    params.append( ( np.load('MF_param/MF_param_'+str(n_run_param)+'/HF_params_'+str(i)+'_'+str(0)+'.npy')[i_ensemble:i_ensemble+1,:,:] , 
                       np.load('MF_param/MF_param_'+str(n_run_param)+'/HF_params_'+str(i)+'_'+str(1)+'.npy')[i_ensemble:i_ensemble+1,:] )  )   
    params_prior.append( ( np.load('MF_param/MF_param_'+str(n_run_param)+'/HF_params_prior_'+str(i)+'_'+str(0)+'.npy')[i_ensemble:i_ensemble+1,:,:] , 
                             np.load('MF_param/MF_param_'+str(n_run_param)+'/HF_params_prior_'+str(i)+'_'+str(1)+'.npy')[i_ensemble:i_ensemble+1,:] )  )     

opt_params_H = (params, params_prior)

params_L = []
params_prior_L = []

for i in range(len(layers_L)-1):
    params_L.append( ( np.load('MF_param/MF_param_'+str(n_run_param)+'/LF_params_'+str(i)+'_'+str(0)+'.npy')[i_ensemble:i_ensemble+1,:,:] , 
                       np.load('MF_param/MF_param_'+str(n_run_param)+'/LF_params_'+str(i)+'_'+str(1)+'.npy')[i_ensemble:i_ensemble+1,:] )  )   
    params_prior_L.append( ( np.load('MF_param/MF_param_'+str(n_run_param)+'/LF_params_prior_'+str(i)+'_'+str(0)+'.npy')[i_ensemble:i_ensemble+1,:,:] , 
                             np.load('MF_param/MF_param_'+str(n_run_param)+'/LF_params_prior_'+str(i)+'_'+str(1)+'.npy')[i_ensemble:i_ensemble+1,:] )  )     

opt_params_L = (params_L, params_prior_L)
    
@jit
def predict_L(x):
    # accepts and returns un-normalized data
    x = np.tile(x[np.newaxis,:,:], (ensemble_size, 1, 1))
    samples = model_L.posterior(opt_params_L, x)
    return samples

@jit
def predict_H(x):
    # accepts and returns un-normalized data
    xL = normalize(x, mu_MF_in, sigma_MF_in)
    x = predict_L(xL)
    samples = model_H.posterior(opt_params_H, x)
    return samples

##################################      
############## test ##############
##################################
print('loading test')
if id_step == 1:
    t = time.time()
    test_XH = onp.load('data_SPCAM5_4K/all_inputs.npy')[:59387904,ind_input]
    print('test loaded', 'time (s):', time.time() - t)
elif id_step == 2:
    t = time.time()
    test_XH = onp.load('data_SPCAM5_4K/all_inputs.npy')[59387904:,ind_input]
    print('test loaded', 'time (s):', time.time() - t)

print('n_run_param : ', n_run_param, 'id_step : ', id_step, 'Computing pred')
t = time.time()
samples_test_H = predict_H(test_XH)
print('Computing pred', 'time (s):', time.time() - t)
print('Saving pred')
t = time.time()
np.save('MF_param/MF_param_'+str(n_run_param)+'/test_pred_'+str(id_step)+'.npy', samples_test_H)
print('Saving pred', 'time (s):', time.time() - t)
    