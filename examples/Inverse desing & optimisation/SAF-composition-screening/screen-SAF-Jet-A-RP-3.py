# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 16:37:54 2026

@author: Rodolfo Freitas
"""

import os
import pandas as pd
import numpy as np

from datetime import datetime
from timeit import default_timer
from math import ceil
from deepfuel.scipy_optimizers import optimize_fuel
from deepfuel.io import load_model
from deepfuel.mol_featurizer import featurize_molecules
from scipy.optimize import NonlinearConstraint
import torch
import multiprocessing
#


#%% Auxiliary functions    
def predict(x):
    y_pred = model.predict(x) 
    return scaler.inverse_transform(y_pred)

def predict_TF(x):
    y_pred = model_TF.predict(x) 
    return scaler_TF.inverse_transform(y_pred)

    
def obj_fun(x, phi, target, alpha):
    # Mixing Operator
    Phi = np.matmul(x[None, :], phi)
    
    # call the compositional mapping
    y_pred = predict(Phi)
    y_pred_TF = predict_TF(Phi).reshape(1,-1)
    y_pred = np.concatenate((y_pred, y_pred_TF), axis=1)
    
    # L1 regularization (Lasso)
    l1_reg = np.linalg.norm(x, 1)

    # MSE + L1 penalty
    loss = np.mean(np.square(target - y_pred[0][:len(target)])/np.square(target)) + alpha * l1_reg 

    return loss


def jac_fun(x, phi, y_true, alpha, eps=1e-6):
    n = len(x)
    grad = np.zeros_like(x)

    f0 = obj_fun(x, phi, y_true, alpha)

    for j in range(n):
        x_perturb = x.copy()
        x_perturb[j] += eps
        f1 = obj_fun(x_perturb, phi, y_true, alpha)
        grad[j] = (f1 - f0) / eps

    return grad  

    

if __name__ == "__main__":
    date_time = datetime.now().strftime("%d-%b-%Y %H:%M:%S")
    #%% READ MOLECULES

    # read chemical space
    data = pd.read_excel('SAF-palette.xlsx')
    fuel_list = data['Name']
    SMILES = data['SMILES'].tolist()
    classes = data['Class'].tolist() 
    
    #%% Load the pre-trained compositional mapping
    model       = load_model('Pre-Trained-Models/MLP-predictor-rho-LHV-FP/best_model.joblib')
    scaler = load_model('Pre-Trained-Models/MLP-predictor-rho-LHV-FP/scalery.joblib')
    model_TF     = load_model('Pre-Trained-Models/MLP-predictor-TF/best_model.joblib')
    scaler_TF = load_model('Pre-Trained-Models/MLP-predictor-TF/scalery.joblib')
    
    phi = featurize_molecules(featurizer='Mol2VecFingerprint', smiles_list=SMILES)

    # ---- Nonlinear property constraints ----
    # ASTM D1655 (Jet A/A-1) and GB 6537(RP-3)
    # density - 0.775 -- 0.840 / 0.775 -- 0.830
    # Net Heat of Combustion - min 42.8  
    # Flash Point - min 38
    # Freezing point - max -40 (Jet A) or -47 (Jet A-1) / max -47   
    cons = [NonlinearConstraint(lambda x: predict(np.matmul(x[None,:],phi))[0][0], 0.775, 0.840),
            NonlinearConstraint(lambda x: (predict(np.matmul(x[None, :], phi)))[0][1] - 42.8, 0, np.inf),
            NonlinearConstraint(lambda x: (predict(np.matmul(x[None, :], phi)))[0][2] - 38.0, 0, np.inf),
            NonlinearConstraint(lambda x: - (predict_TF(np.matmul(x[None, :], phi)))[0] - 40.0, 0, np.inf)]

    # ---- Add your sum-to-one (mixture) constraint ----
    cons.append({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
    
    # ---- enforce a minimum aromatic content of 8% (0.08) according to ASTM D7566, and maximum 25% ASTM D1655
    is_aromatic = np.array([c.lower() == 'aromatic' for c in classes])
    cons.append( NonlinearConstraint(lambda x: np.sum(x[is_aromatic]),
                                     0.08,   # ASTM D7566 minimum = 8%
                                     0.25    # upper limit (keep this if you still want it)
                                     ))
    
    # target density fuel
    '''
    Screening SAF blends that emulate Jet-A and RP-3 jet fuels.
    Target Properties Source: https://doi.org/10.1016/j.paerosci.2024.101054
    Jet-A, 0.806, 42.8, 47, -49
    RP-3, 0.788, 42.8, 43, -47
    '''
    fuel_target = np.array([0.806, 42.8, 47])
    fuel = 	'SAF-Jet-A-rho-LHV-FP'
    alpha = [0.1, 0.5, 1.0]
    K = 1000
        
    # Save diretory
    results_dir = "results/" + fuel
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    
    # Optimization
    num_cores = multiprocessing.cpu_count()
    for idx in range(len(alpha)):
        output_filename = results_dir + '/output_optimization_fuel_{}_lasso_{}_samples_{}.txt'.format(fuel,alpha[idx],K)
        print(output_filename)
    
        f = open(output_filename, "w")
        f.write(">>>>>>>>>>>>>>>>>>>>  Output File for Inverse fuel design with scipy minimize  <<<<<<<<<<<<<<<<\n\n")
        f.write(f"Rodolfo Freitas      {date_time}\n")
        f.write(f"GPU name: {torch.cuda.get_device_name(model._device())}\n")
        f.write(f"number of cpus: {num_cores}\n")
        f.write(f"Target fuel: {fuel_target}\n")
        f.write("-------------------------------------------------------------------------------------------------------\n\n")
        f.close()
        
    
        
        time_start = default_timer()
        best_x, best_sol, solutions = optimize_fuel(obj_fun, 
                                                    args=(phi, fuel_target, alpha[idx]),
                                                    optimizer='SLSQP',
                                                    n_features=len(SMILES),
                                                    constraints=cons,
                                                    jac=jac_fun,
                                                    n_starts=K,
                                                    n_jobs=num_cores)
        time_end = default_timer()
        
        mins, secs = divmod(ceil(time_end-time_start), 60)
        hours, mins = divmod(mins, 60)
        days, hours = divmod(hours, 24)
    
        
        
        f = open(output_filename, "a")
        f.write("-------------------------------------------------------------------------------------------------------\n\n")
        f.write(f"Total Optimization time: {days:02d}-{hours:02d}:{mins:02d}:{secs:02d}\n")
        f.write("-------------------------------------------------------------------------------------------------------\n\n")
        f.close()
        
        
        # save
        filename = f"{results_dir}/compositions_optimization_fuel_{fuel}_lasso_{alpha[idx]}_samples_{K}.npz"
        xs = [s[0] for s in solutions]
        sols = [s[1] for s in solutions]
        np.savez(filename, xs=xs, sols=sols)
