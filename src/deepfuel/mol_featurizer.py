# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 12:23:49 2025

@author: Rodolfo Freitas
"""

import deepchem as dc
import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import DataStructs, rdFingerprintGenerator

if not hasattr(np, "product"):
    np.product = np.prod

import torch
from torch_geometric.data import Data

# Optional: Hugging Face Transformers for ChemBERTa
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------

def featurizer_rdkit(smiles_list):
    """
    ~200 descriptors (Physical–chemical)
    """
    print('Featurizing with RDKit Descriptors...')
    featurizer = dc.feat.RDKitDescriptors()
    return featurizer.featurize(smiles_list)

def featurizer_mordred(smiles_list, ignore_3D=True):
    """
    Moriwaki, Hirotomo, et al. “Mordred: a molecular descriptor calculator.” 
    Journal of cheminformatics 10.1 (2018): 4.
    ~1800 descriptors 
    """
    print('Featurizing with Mordred Descriptors...')
    featurizer = dc.feat.MordredDescriptors(ignore_3D=ignore_3D)
    return featurizer.featurize(smiles_list)

def featurizer_Morgan(smiles_list, radius=2, n_bits=1024):
    '''
    Morgan fingerprint (same as ECFP).
    radius: Neighborhood radius (ECFP4 = 2-default for most ML applications, ECFP6 = 3).
    n_bits: Length of fingerprint vector (Fingerprint length (1024–4096 typical))
    '''
    print('Featurizing with Morgan Fingerprints...')
    
    featurizer = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    X = np.zeros((len(smiles_list), n_bits), dtype=int)
    
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            # Handle invalid SMILES
            X[i, :] = np.zeros(n_bits)
            continue
        fp = featurizer.GetFingerprint(mol)
        arr = np.zeros(n_bits, dtype=int)
        DataStructs.ConvertToNumpyArray(fp, arr)
        X[i, :] = arr
    return X
  

def featurizer_mol2vec(smiles_list, model_path=None):
    '''
    Jaeger, Sabrina, Simone Fulle, and Samo Turk. 
    “Mol2vec: unsupervised machine learning approach with chemical intuition.” 
    Journal of chemical information and modeling 58.1 (2018): 27-35.
    The default model was trained on 20 million compounds downloaded from ZINC 
    using the following paramters.
        - radius 1
        - UNK to replace all identifiers that appear less than 4 times
        - skip-gram and window size of 10
        - embeddings size 300
    '''
    if model_path is not None:
        featurizer = dc.feat.Mol2VecFingerprint(model_path=model_path)
        print('Featurizing with Mol2Vec pretrained embeddings from {model_path}')
    else:
        print('Featurizing with Mol2Vec pretrained embeddings from https://github.com/samoturk/mol2vec/')
        featurizer = dc.feat.Mol2VecFingerprint()
    
    return featurizer.featurize(smiles_list)


def featurizer_chemberta(smiles_list, model_path=None, device=None):
    """
    Convert a list of SMILES strings into ChemBERTa embeddings.

    Args:
        smiles_list (list of str): SMILES strings of molecules
        model_name (str): HuggingFace model name 
        device (str or torch.device, optional): "cuda" or "cpu". Defaults to GPU if available.

    Returns:
        np.Array: Embeddings [num_molecules, hidden_dim]
    """
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    
    # Load model
    if model_path is not None:
        print('Featurizing with pretrained ChemBERTa from {model_path}')
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path, use_safetensors=True).to(device)
    else:
        print('Featurizing with pretrained ChemBERTa from seyonec/ChemBERTa-zinc-base-v1')
        tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1", use_safetensors=True).to(device)
    
    model.eval()
        
    # Tokenize SMILES
    tokens = tokenizer(smiles_list, padding=True, truncation=True, return_tensors="pt").to(device)
    
    # Generate embeddings
    with torch.no_grad():
        outputs = model(**tokens)
        featurizer = outputs.pooler_output  # [batch_size, hidden_dim]
    
    return featurizer.detach().cpu().numpy()


def filter_valid_smiles(smiles, labels=None, verbose=True):
    keep_idx = []
    failed_smiles = []

    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumAtoms() <= 1:
            failed_smiles.append(smi)
        else:
            keep_idx.append(i)

    smiles_filt = [smiles[i] for i in keep_idx]

    if labels is None:
        labels_filt = None
    else:
        labels = np.asarray(labels)
        labels_filt = labels[keep_idx]

    if verbose and failed_smiles:
        print(f"Failed featurization ({len(failed_smiles)} molecules):")
        print(failed_smiles)
        print("More than one atom should be present in the molecule for this featurizer to work")

    return smiles_filt, labels_filt

def featurizer_mol_to_graph(smiles_list,labels=None, 
                            use_edges=True, 
                            use_chirality=False, 
                            use_partial_charge=False,
                            mode='graph',
                            verbose=False):
    """
    Converts one or more SMILES strings into PyTorch Geometric Data objects
    using DeepChem's MolGraphConvFeaturizer ready for GCNConv, GATConv, GraphConv, and D-MPNN.
    Kearnes, Steven, et al. “Molecular graph convolutions: moving beyond fingerprints.” 
    Journal of computer-aided molecular design 30.8 (2016):595-608.

    Parameters
    ----------
    smiles_list : list of str
        List of SMILES strings
    labels : list or np.array
        Optional target labels (regression or classification)
    use_edges: bool, default False
        Whether to use edge features or not.
    use_chirality: bool, default False
        Whether to use chirality information or not.
        If True, featurization becomes slow.
    use_partial_charge: bool, default False
        Whether to use partial charge data or not.
        If True, this featurizer computes gasteiger charges.
        Therefore, there is a possibility to fail to featurize for some molecules
        and featurization becomes slow.
    mode : {'graph', 'dmpnn'}, default='graph'
        - 'graph': standard undirected molecular graph.
        - 'dmpnn': directed bond graph with reverse edge mapping.

    Returns
    -------
    List[torch_geometric.data.Data]
    List of PyG Data objects.

    """
    """
    
    
    Args:
        smiles_list (list of str): List of SMILES strings
        labels (): Optional target labels (regression or classification)
        use_edges (bool): Include bond types as edge features for PyG
    
    Returns: list of torch_geometric.data.Data
        
    """

    pyg_graphs = []
    
    if mode == "graph":
        # --- Standard graph for GraphConv, MFConv, GATConv ---
        if verbose == True:
            print("Featurizing with MolGraphConvFeaturizer...")
        
        # DeepChem molecular graph featurizer
        graphconv_featurizer = dc.feat.MolGraphConvFeaturizer(
            use_edges=use_edges,
            use_chirality=use_chirality,
            use_partial_charge=use_partial_charge,
        )
        
        # Filter SMILES before featurization
        smiles_filter, labels_filter = filter_valid_smiles(smiles_list, labels)

        # Apply featurization
        graphs = graphconv_featurizer.featurize(smiles_filter)
        
        # --- Determine max edge feature dimension ---
        max_edge_dim = 0 
        if use_edges: 
            for g in graphs: 
                if getattr(g, "edge_features", None) is not None: 
                    max_edge_dim = max(max_edge_dim, g.edge_features.shape[1])
        
        for i, g in enumerate(graphs):
            
            # Node features
            x = torch.tensor(g.node_features, dtype=torch.float)
            
            # Edge indices
            edge_index = torch.tensor(g.edge_index, dtype=torch.long).contiguous()

            data_kwargs = {"x": x, "edge_index": edge_index}
            
            # Edge features
            if use_edges and getattr(g, "edge_features", None) is not None:
                edge_attr = torch.tensor(g.edge_features, dtype=torch.float)
                data_kwargs["edge_attr"] = edge_attr
                
                
            
            # Labels    
            if labels_filter is not None:
                label_array = np.array(labels_filter[i], dtype=np.float32).reshape(1, -1)
                data_kwargs["y"] = torch.tensor(label_array, dtype=torch.float)

            pyg_graphs.append(Data(**data_kwargs))

    elif mode == "dmpnn":
        # --- Directed Message Passing Neural Network (D-MPNN) ---
        if verbose == True:
            print("Featurizing for D-MPNN...")

        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smi}")

            # Atom features
            atom_feats = []
            for atom in mol.GetAtoms():
                atom_feats.append([
                    atom.GetAtomicNum(),
                    atom.GetTotalDegree(),
                    int(atom.GetIsAromatic()),
                ])
            x = torch.tensor(atom_feats, dtype=torch.float)

            # Edge features and directed edges
            edge_index_list, edge_attr_list = [], []
            for bond in mol.GetBonds():
                a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                f = [
                    bond.GetBondTypeAsDouble(),
                    float(bond.GetIsConjugated()),
                    float(bond.IsInRing())
                ]
                # Add both directions
                edge_index_list += [[a, b], [b, a]]
                edge_attr_list += [f, f]

            # Build tensors
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)


            data_kwargs = {
                "x": x,
                "edge_index": edge_index,
                "edge_attr": edge_attr
            }
            
            # Labels
            if labels is not None:
                label_array = np.array(labels[i], dtype=np.float32).reshape(1, -1)
                data_kwargs["y"] = torch.tensor(label_array, dtype=torch.float)

            pyg_graphs.append(Data(**data_kwargs))

    else:
        raise ValueError("Invalid mode. Choose 'graph' or 'dmpnn'")

    return pyg_graphs


# ---------------------------------------------------------------
# General Unified Featurization Function
# ---------------------------------------------------------------

def featurize_molecules(featurizer: str, **kwargs):
    """
    Generate molecular features using multiple featurizers.

    Parameters
    ----------
    featurizer : str
        Name of the featurizer
    smiles_list : list
        List of SMILES strings
    **kwargs : TYPE
        keyword arguments of the featurizer.

    Returns
    -------
    X : np.ndarray: Concatenated feature matrix [n_mols, n_features_total]
    
    """
    
    featurizers = {
        "RDKitDescriptors": featurizer_rdkit,
        "MordredDescriptors": featurizer_mordred,
        "MorganFingerprint": featurizer_Morgan,
        "Mol2VecFingerprint": featurizer_mol2vec,
        "ChemBERTa": featurizer_chemberta,
        "Mol2Graph": featurizer_mol_to_graph,
    }
    
    if featurizer not in featurizers:
        raise ValueError(f"Unknown featurizer '{featurizer}'. Available: {list(featurizers.keys())}")
    
    return featurizers[featurizer](**kwargs) 

# ---------------------------------------------------------------
# Example
# ---------------------------------------------------------------

# if __name__ == "__main__":
    
#     smiles = ["CCO", "CCN", "c1ccccc1O"]
#     labels = np.array(([0.5, 1.0, 0.9], [10, 7, 4])).T
    
#     # Molecular Featurisation for traditional ML models (inputs to outputs)
    
#     X_rdkit = featurize_molecules(featurizer='RDKitDescriptors', smiles_list=smiles)
#     X_mordred = featurize_molecules(featurizer='MordredDescriptors', smiles_list=smiles, ignore_3D=True)
#     X_morgan = featurize_molecules(featurizer='MorganFingerprint', smiles_list=smiles, n_bits=1024, radius=2)
#     X_mol2vec = featurize_molecules(featurizer='Mol2VecFingerprint', smiles_list=smiles, model_path=None)
#     X_chemberta = featurize_molecules(featurizer='ChemBERTa', smiles_list=smiles, 
#                                       model_path='seyonec/ChemBERTa-zinc-base-v1', device=None)
#     X_mol2graph = featurize_molecules(featurizer='Mol2Graph', smiles_list=smiles, 
#                                       labels=labels, 
#                                       use_edges=True, mode='graph')
    

