# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 10:26:40 2026

@author: Rodolfo Freitas
"""

from matplotlib import pyplot as plt
import time
import argparse
import numpy as np
import os
import pandas as pd
import torch
from deepfuel.data_utils import prepare_data, preprocess_features, recursive_feature_elimination
from deepfuel.models import get_model
from deepfuel.metrics import regression_metrics
from deepfuel.hpo import tune
from deepfuel.io import save_model, load_model
from deepfuel.mol_featurizer import featurize_molecules

# Reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Publication-style formatting
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "lines.linewidth": 1.5,
    "savefig.dpi": 600,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# Train
parser = argparse.ArgumentParser(description='Compositional mapping')
parser.add_argument('--data-dir', type=str, default="../database", help='data directory')
parser.add_argument('--property', nargs='+', default=['DCN'], 
                    help='List of properties to property: BP | rho | LHV | FP | ST | TF | YSI | DCN' )
parser.add_argument('--train-size', type=float, default=0.80, help='amount of the data used to train')
parser.add_argument('--model', type=str, default='DeepNN', help='Model to construct the compositional mapping')
parser.add_argument('--featurizer', type=str, default='Mol2VecFingerprint', 
                    help='Molecular representation: RDKitDescriptors | MordredDescriptors | MorganFingerprint | Mol2VecFingerprint| ChemBERTa| Mol2Graph')

args = parser.parse_args()

# Check if cuda is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('------------ Arguments -------------')
print("Torch device:{}-{}".format(device,torch.cuda.get_device_name(0)))
for k, v in sorted(vars(args).items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')



#%% Preparing the data for model (pure compounds)
data            = pd.read_excel('{}/data-fuel-properties.xlsx'.format(args.data_dir))
fuel_list_pure  = data["Name"]
SMILES          = data["SMILES"].to_numpy()
y               = data[args.property].to_numpy()  

# mask out NaNs (sparse data we don't have all properties)
mask    = ~np.isnan(y)
X       = SMILES[mask[:,0]].tolist()
y       = y[mask] 


tic = time.time()
# Generate the molecule features
X = featurize_molecules(featurizer=args.featurizer, smiles_list=X)

if args.featurizer in ('RDKitDescriptors','MordredDescriptors'):
    # removes low-variance features and highly correlated features.
    X , X_idx = preprocess_features(X, y, var_threshold=1e-6, corr_threshold=0.8, corr_method='pearson')
    print("Processed shape:", X.shape)
    # Perform RFECV to automatically select the optimal number of features.
    X, idx_selected, selector = recursive_feature_elimination(X, y, n_features_to_select=10)
    print("Shape after RFE:", X.shape)

# split and scaling the data
X_train, X_test, y_train, y_test, scalerX, scalery = prepare_data(X, 
                                                                  y, 
                                                                  train_size=args.train_size,  
                                                                  scaler_X=None, 
                                                                  scaler_y="mad")



#%% Get the model and tunning the hyperparams
# Save diretory
model_dir = f"Models/MLP_Featurizer_{args.featurizer}_Property_{args.property[0]}"
    
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model = get_model(args.model, epochs=300, batch_size=64, verbose=False)

def suggest_params(trial, hidden_dims_list):

    return {
        "hidden_dims": trial.suggest_categorical("hidden_dims", hidden_dims_list),
        "hidden_activation": trial.suggest_categorical("hidden_activation", ['relu', 'prelu', 'leakyrelu', 'elu', 'gelu', 'tanh', None]),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    }

# hidden dimensions 
min_layers = 1
max_layers = 4
neuron_choices = [16, 32, 64, 128, 256, 512]

# Generate all unique tuples
hidden_dims_list = []

for n_layers in range(min_layers, max_layers+1):
    for neurons in neuron_choices:
        hidden_dims_list.append((neurons,) * n_layers)

study, best_model = tune(model, 
                          lambda trial: suggest_params(trial, hidden_dims_list), 
                          X_train, 
                          y_train, 
                          metric="r2",
                          n_splits=5,
                          save_dir=model_dir,
                          n_trials=100)

tic2 = time.time()
print(f"Done training in {tic2 - tic} seconds")

#%% Make predictions

y_pred_train = scalery.inverse_transform(best_model.predict(X_train))
y_pred_test  = scalery.inverse_transform(best_model.predict(X_test))
y_train      = scalery.inverse_transform(y_train)
y_test       = scalery.inverse_transform(y_test)

np.save(f"{model_dir}/{args.property[0]}_predictions_train_dataset.npy", y_pred_train)
np.save(f"{model_dir}/{args.property[0]}_predictions_test_dataset.npy", y_pred_test)
np.save(f"{model_dir}/{args.property[0]}_train_dataset.npy", y_train)
np.save(f"{model_dir}/{args.property[0]}_test_dataset.npy", y_test)

accuracy_metrics_train = regression_metrics(y_train, y_pred_train)
accuracy_metrics_test = regression_metrics(y_test, y_pred_test)

# creata a dataframe for metrics
df = {'Train': accuracy_metrics_train, #
      'Test': accuracy_metrics_test}

metrics = pd.DataFrame(data=df)
metrics.to_excel(model_dir + '/metrics.xlsx')

#%% plotting

# Scatter plot
colors = ["#ff7f0e", "#1f77b4"]

min_data  = np.minimum(np.minimum(np.amin(y_train),np.amin(y_pred_train)), np.minimum(np.amin(y_test), np.amin(y_pred_test)))
max_data = np.maximum(np.maximum(np.amax(y_train),np.amax(y_pred_train)), np.maximum(np.amax(y_test),np.amax((y_pred_test))))

plt.figure(figsize=(3.5,2.8))

plt.scatter(np.reshape(y_pred_train,-1),np.reshape(y_train,-1), 
                  s=10, marker='o', color = colors[0],  alpha=0.7, label=r'Train')
plt.scatter(np.reshape(y_pred_test,-1),np.reshape(y_test,-1), 
                  s=10 ,marker='o', color = colors[1], alpha=0.7, label=r'Test')
xlim = plt.xlim(min_data, max_data)
ylim = plt.ylim(min_data, max_data)
plt.plot(xlim,ylim,'k--')
if args.property[0] == 'LHV': 
    plt.xlabel(r'Predicted LHV [MJ/kg]')
    plt.ylabel(r'Measured LHV [MJ/kg]')
    ticks = np.linspace(10, 50, 5)
elif args.property[0] == 'rho':
    plt.xlabel(r'Predicted Density [g/cm$^3$]')
    plt.ylabel(r'Measured Density [g/cm$^3$]')
    ticks = np.linspace(0.5, 2, 4)
elif args.property[0] == 'TF':
    plt.xlabel(r'Predicted T$_F$ [$^\circ$C]')
    plt.ylabel(r'Measured T$_F$ [$^\circ$C]')
    ticks = np.linspace(-200, 200, 5)
elif args.property[0] == 'BP':
    plt.xlabel(r'Predicted BP [$^\circ$C]')
    plt.ylabel(r'Measured BP [$^\circ$C]')
    ticks = np.linspace(-200, 500, 5)
elif args.property[0] == 'FP':
    plt.xlabel(r'Predicted FP [$^\circ$C]')
    plt.ylabel(r'Measured FP [$^\circ$C]')
    ticks = np.linspace(-200, 500, 5)
elif args.property[0] == 'YSI':
    plt.xlabel(r'Predicted YSI')
    plt.ylabel(r'Measured YSI')
    ticks = np.linspace(0, 1400, 5)
    
plt.xticks(ticks)
plt.yticks(ticks)
plt.legend(loc='best', frameon=False, prop={'weight': 'extra bold'})
plt.savefig(f"{model_dir}/parity_{args.property[0]}.pdf", bbox_inches='tight')
plt.show()

colors = ["#9DEEB1",          
    "#41CEBA",   
    "#017FA6",
    "#01558E", 
    "#011959"]

plt.figure(figsize=(3.5,2.8))

xlim = plt.xlim(min_data, max_data)
ylim = plt.ylim(min_data, max_data)
plt.plot(xlim,ylim,'k--')
if args.property[0] == 'LHV': 
    plt.scatter(np.reshape(y_pred_test,-1),np.reshape(y_test,-1), 
                      s=10 ,marker='o', color = colors[1], alpha=0.75)
    plt.xlabel(r'Predicted LHV [MJ/kg]')
    plt.ylabel(r'Measured LHV [MJ/kg]')
    ticks = np.linspace(10, 50, 5)
elif args.property[0] == 'rho':
    plt.scatter(np.reshape(y_pred_test,-1),np.reshape(y_test,-1), 
                      s=10 ,marker='o', color = colors[0], alpha=0.75)
    plt.xlabel(r'Predicted Density [g/cm$^3$]')
    plt.ylabel(r'Measured Density [g/cm$^3$]')
    ticks = np.linspace(0.5, 2, 4)
elif args.property[0] == 'TF':
    plt.scatter(np.reshape(y_pred_test,-1),np.reshape(y_test,-1), 
                      s=10 ,marker='o', color = colors[2], alpha=0.75)
    plt.xlabel(r'Predicted T$_F$ [$^\circ$C]')
    plt.ylabel(r'Measured T$_F$ [$^\circ$C]')
    ticks = np.linspace(-200, 200, 5)
elif args.property[0] == 'FP':
    plt.scatter(np.reshape(y_pred_test,-1),np.reshape(y_test,-1), 
                      s=10 ,marker='o', color = colors[3], alpha=0.75)
    plt.xlabel(r'Predicted FP [$^\circ$C]')
    plt.ylabel(r'Measured FP [$^\circ$C]')
    ticks = np.linspace(-200, 500, 5)
elif args.property[0] == 'YSI':
    plt.scatter(np.reshape(y_pred_test,-1),np.reshape(y_test,-1), 
                      s=10 ,marker='o', color = colors[4], alpha=0.75)
    plt.xlabel(r'Predicted YSI')
    plt.ylabel(r'Measured YSI')
    ticks = np.linspace(0, 1400, 5)
    
plt.xticks(ticks)
plt.yticks(ticks)
plt.legend(loc='best', frameon=False, prop={'weight': 'extra bold'})
plt.savefig(f"{model_dir}/parity_test_{args.property[0]}.pdf", bbox_inches='tight')
plt.show()