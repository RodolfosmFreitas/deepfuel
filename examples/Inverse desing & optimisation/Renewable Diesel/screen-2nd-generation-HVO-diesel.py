# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from datetime import datetime
from timeit import default_timer
from math import ceil
from deepfuel.hybrid_optim import HybridOptimization
from deepfuel.ga import FuelOptimizer
from deepfuel.io import load_model
from deepfuel.mol_featurizer import featurize_molecules
import matplotlib.pyplot as plt
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

#


#%% Auxiliary functions    
def predict_CN(x):
    y_pred = model_CN.predict(x) 
    return scaler_CN.inverse_transform(y_pred)

def predict_rho(x):
    y_pred = model_rho.predict(x) 
    return scaler_rho.inverse_transform(y_pred)

def predict_LHV(x):
    y_pred = model_LHV.predict(x) 
    return scaler_LHV.inverse_transform(y_pred)

def predict_FP(x):
    y_pred = model_FP.predict(x) 
    return scaler_FP.inverse_transform(y_pred)

def predict_BP(x):
    y_pred = model_BP.predict(x) 
    return scaler_BP.inverse_transform(y_pred)

def predict_YSI(x):
    y_pred = model_YSI.predict(x) 
    return scaler_YSI.inverse_transform(y_pred)


def project_simplex_threshold(x, tol=1e-2):
    x = np.array(x, dtype=float).copy()
    
    # Zero-out small components
    x[x < tol] = 0.0
    
    total = x.sum()
    
    # Avoid division by zero (edge case: everything removed)
    if total <= 1e-12:
        # fallback: keep the largest original component
        idx = np.argmax(x)
        x[idx] = 1.0
        return x
    
    # Renormalize
    x /= total
    
    return x

def obj_fun(x):
    
    x = project_simplex_threshold(x)
    Phi = np.matmul(x[None, :], phi)
    obj_values = []
    
    for obj in objectives:
        if obj == 'CN':
            val = predict_CN(Phi)
            obj_values.append(-val)  # maximize CN
        elif obj == 'LHV':
            val = predict_LHV(Phi)
            obj_values.append(-val)  # maximize LHV
        elif obj == 'FP':
            val = predict_FP(Phi)
            obj_values.append(-val)  # maximize FP
        elif obj == 'YSI':
            val = predict_YSI(Phi)
            obj_values.append(val)   # minimize YSI
        else:
            raise ValueError(f"Unknown objective: {obj}")
    
    return obj_values


def constraints_fun(x):
    
    x = project_simplex_threshold(x)
    
    Phi = np.matmul(x[None, :], phi)
    CN  = np.asarray(predict_CN(Phi)).squeeze().item()
    rho = np.asarray(predict_rho(Phi)).squeeze().item()
    FP  = np.asarray(predict_FP(Phi)).squeeze().item()
    BP  = np.asarray(predict_BP(Phi)).squeeze().item()
    LHV = np.asarray(predict_LHV(Phi)).squeeze().item()
    
    g = [
        70. - CN,         # CN ≥ 70
        CN - 90,        # CN  ≤ 90
        0.770 - rho,     # rho ≥ 0.770
        rho - 0.790,     # rho ≤ 0.790
        61. - FP,         # FP ≥ 61
        180. - BP,        # BP ≥ 180
        42. - LHV         # LHV ≥ 42 
    ]
        
    if include_omex:
        omex_fraction = float(np.sum(x[is_ethers]))
        g += [omex_fraction - 0.15,   # OMEx ≤ 15%
              0.05 - omex_fraction    # OMEx ≥ 5%
              ]
    
    return np.array(g, dtype=float)
    
    

if __name__ == "__main__":
    date_time = datetime.now().strftime("%d-%b-%Y %H:%M:%S")
    #%% READ MOLECULES

    # read chemical space
    data = pd.read_excel('Diesel-palette.xlsx')
    classes = data['Class'].tolist() 
    
    
    
    #%% Load the pre-trained compositional mapping
    model_CN        = load_model('Models/MLP-predictor-DCN/best_model.joblib')
    scaler_CN       = load_model('Models/MLP-predictor-DCN/scalery.joblib')
    model_YSI       = load_model('Models/MLP-predictor-YSI/best_model.joblib')
    scaler_YSI      = load_model('Models/MLP-predictor-YSI/scalery.joblib')
    model_rho       = load_model('Models/MLP-predictor-rho/best_model.joblib')
    scaler_rho      = load_model('Models/MLP-predictor-rho/scalery.joblib')
    model_FP        = load_model('Models/MLP-predictor-FP/best_model.joblib')
    scaler_FP       = load_model('Models/MLP-predictor-FP/scalery.joblib')
    model_LHV       = load_model('Models/MLP-predictor-LHV/best_model.joblib')
    scaler_LHV      = load_model('Models/MLP-predictor-LHV/scalery.joblib')
    model_BP        = load_model('Models/MLP-predictor-BP/best_model.joblib')
    scaler_BP       = load_model('Models/MLP-predictor-BP/scalery.joblib')
    
    
    '''
    property constraints
    EN 15940 (Paraffinic diesel standard)
    https://www.crownoil.co.uk/fuel-specifications/bs-en-15940/
    Density at 15°C	kg/m3  -- 0.770 – 0.790
    Flash point °C (EN ISO 2719) -- min 61
    Initial Boiling Point °C -- min 180 
    Cetane Number -- min 70 (typical values 70-90)
    Net Heat of Combustion	MJ/kg -- min 42	
    '''
    
    optimizer = 'Hybrid'
    n_gen = 200
    n_pop = 100
    n_cycles = 3
    objectives = ['CN', 'LHV', 'YSI']
    include_omex = True
    
    is_ethers = np.array([c.lower() == 'ethers' for c in classes])
    if include_omex == False:
        fuel = 	'HVO-Diesel'
        fuel_list = data['Name'][~is_ethers]
        SMILES = data['SMILES']
        SMILES = SMILES[~is_ethers].tolist()
        phi = featurize_molecules(featurizer='Mol2VecFingerprint', smiles_list=SMILES)
        n_comp = len(fuel_list)
    else:
        fuel = 	'HVO-Diesel-OMEn'
        fuel_list = data['Name']
        SMILES = data['SMILES'].tolist()
        phi = featurize_molecules(featurizer='Mol2VecFingerprint', smiles_list=SMILES)
        n_comp = len(fuel_list)
        
    
    # Optimization
    # Save diretory
    results_dir = f"results/{optimizer}-{fuel}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
     
    objective_str = "-".join(objectives)
    if optimizer == 'GA': 
        output_filename = f"{results_dir}/output-{fuel}-{objective_str}-optimization-{optimizer}-generation-{n_gen}-population-{n_pop}.txt" 
    else:
        output_filename = f"{results_dir}/output-{fuel}-{objective_str}-optimization-{optimizer}-generation-{n_gen}-population-{n_pop}-cycles-{n_cycles}.txt" 
        
    f = open(output_filename, "w")
    f.write(">>>>>>>>>>>>>>>>>>>>  Output File for HVO fuel design <<<<<<<<<<<<<<<<\n\n")
    f.write(f"Rodolfo Freitas      {date_time}\n")
    f.write(f"Target fuel: {fuel}\n")
    f.write(f"Optimization: {optimizer}\n")
    f.write("----------------------------------------------------------------------\n\n")
    f.close()

    
    time_start = default_timer()
    seeds = [0, 42, 123, 456, 789]
    if optimizer == 'GA':
        all_results = []
        all_hv = []
        for seed in seeds:
            opt = FuelOptimizer(obj_fun, 
                                n_comp, 
                                n_obj=len(objectives), 
                                bounds_oper=None,
                                constraints_fun=constraints_fun, 
                                pop_size=n_pop, 
                                n_gen=n_gen, 
                                seed=seed).run()
        
            
            all_results.append(opt['result'])
            all_hv.append(opt['hv_history'])
    
    elif optimizer== 'Hybrid':
        all_results = []
        all_hv = []
        for seed in seeds:
            hybrid_opt = HybridOptimization(n_comp=n_comp,
                                            bounds_oper=None,
                                            obj_fun=obj_fun,
                                            n_obj=len(objectives),
                                            constraints_fun=constraints_fun,
                                            n_cycles=n_cycles,
                                            ga_params={'pop_size': n_pop, 
                                                       'n_gen': n_gen,
                                                       #'hv_ref_point':np.array([80, 100.0]),
                                                       'seed': seed},
                                            rl_params={'method': 'SAC', 
                                                       'timesteps': 10000, 
                                                       'verbose': 1, 
                                                       'early_stop': True,
                                                       'tol': 1e-3,
                                                       'window': 20})
            opt, model = hybrid_opt.run()
            all_results.append(opt['result'])
            all_hv.append(opt['hv_history'])
        
    time_end = default_timer()
     
    mins, secs = divmod(ceil(time_end-time_start), 60)
    hours, mins = divmod(mins, 60)
    days, hours = divmod(hours, 24)
 
    f = open(output_filename, "a")
    f.write("-------------------------------------------------------------------------------------------------------\n\n")
    f.write(f"Total Optimization time: {days:02d}-{hours:02d}:{mins:02d}:{secs:02d}\n")
    f.write("-------------------------------------------------------------------------------------------------------\n\n")
    f.close()
    
    # ALIGN HYPERVOLUME ACROSS RUNS
    max_len = max(len(h) for h in all_hv)
    hv_mat = np.array([np.pad(h, (0, max_len - len(h)), constant_values=np.nan) for h in all_hv])
    hv_mean = np.nanmean(hv_mat, axis=0)
    hv_std  = np.nanstd(hv_mat, axis=0)
    
    plt.plot(hv_mean, label="Mean HV")
    plt.fill_between(
        range(len(hv_mean)),
        hv_mean - hv_std,
        hv_mean + hv_std,
        alpha=0.3,
        label="Std"
    )
    
    plt.xlabel("Generation")
    plt.ylabel("Hypervolume")
    plt.title("DeepFuel Convergence (Multi-seed)")
    plt.legend()
    plt.show()
    
    # Robust Concensus Pareto front
    F_all = np.vstack([r.F for r in all_results])
    X_all = np.vstack([r.X for r in all_results])
    
    

    nds = NonDominatedSorting().do(F_all, only_non_dominated_front=True)

    F_consensus = F_all[nds]
    X_consensus = X_all[nds]
    _, unique_idx = np.unique(F_consensus, axis=0, return_index=True)
    F_consensus = F_consensus[unique_idx]
    X_consensus = X_consensus[unique_idx]
    
    # save
    filename = f"{results_dir}/compositions-optimization-{fuel}-{optimizer}-{objective_str}.npz"
    if optimizer == 'GA':
        X = [project_simplex_threshold(X_consensus[s]) for s in range(X_consensus.shape[0])]
        np.savez(filename, x=X, F=F_consensus)
        np.savez(f"{results_dir}/HV-optimization-{fuel}-{optimizer}-{objective_str}.npz", 
                 mean=hv_mean, F=hv_std)
    elif optimizer== 'Hybrid':
        X = [project_simplex_threshold(X_consensus[s]) for s in range(X_consensus.shape[0])]
        np.savez(filename, x=X, F=F_consensus)
        np.savez(f"{results_dir}/HV-optimization-{fuel}-{optimizer}-{objective_str}.npz", 
                 mean=hv_mean, F=hv_std)
        
        # # save the pareto cycles
        # np.savez(f"{results_dir}/cycles-Pareto-{fuel}-{optimizer}-{objective_str}.npz",
        #           cycles_F=np.array(hybrid_opt.cycles_F, dtype=object),
        #           cycles_X=np.array(hybrid_opt.cycles_X, dtype=object))
        
    
        
