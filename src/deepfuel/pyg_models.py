# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 14:50:22 2025

@author: Rodolfo Freitas
"""

from typing import List, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    GraphConv,
    Linear,
    MFConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.typing import Adj

from deepfuel.mol_featurizer import featurize_molecules


class DMPNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
      
        """
        Directed Message Passing Neural Network (D-MPNN) layer for graph data.
    
        This layer implements a single message-passing step where messages are computed 
        from neighboring nodes and edge features, then aggregated to update node representations.
    
        Parameters
        ----------
        in_channels : int
            Number of input node features per node.
        out_channels : int
            Number of output node features after message passing.
    
        Forward Inputs
        --------------
        x : torch.Tensor
            Node feature matrix of shape [num_nodes, in_channels].
        edge_index : torch_geometric.typing.Adj
            Edge indices in COO format with shape [2, num_edges]. 
            The first row contains target nodes, the second row contains source nodes.
        edge_attr : torch.Tensor
            Edge feature matrix of shape [num_edges, edge_in_channels].
    
        Returns
        -------
        out : torch.Tensor
            Updated node feature matrix of shape [num_nodes, out_channels].
    
        Notes
        -----
        - This layer concatenates the sender node features with edge features and applies
          a two-layer MLP to generate messages.
        - Messages are aggregated to target nodes using `index_add`, which sums all messages
          for each target node.
        - Edge feature projection is applied if `edge_in_channels` differs from `in_channels`.
        """
      
        # Edge MLP: combine sender node and edge features
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_channels + in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
    
    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor):
        row, col = edge_index  # col -> row
        # project edge_attr to match node dim if needed
        if edge_attr.shape[1] != x.shape[1]:
            edge_attr = nn.Linear(edge_attr.shape[1], x.shape[1]).to(x.device)(edge_attr)

        messages = torch.cat([x[col], edge_attr], dim=-1)
        messages = self.edge_mlp(messages)

        # Initialize out with shape [num_nodes, hidden_channels]
        out = torch.zeros(x.size(0), messages.size(1), device=x.device)
        out = out.index_add(0, row, messages)
        return out




class GNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_graph_layers: int = 3,
        conv_layer: str = 'GraphConv',
        num_mlp_layers: int = 2,
        out_features: int = 1,
        activation: Optional[str] = 'relu',
        dropout: float = 0.0,
        pooling: Literal['mean','max','add'] = 'mean',
        use_residual: bool = False
    ):
        
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_graph_layers = num_graph_layers
        self.num_mlp_layers = num_mlp_layers
        self.out_features = out_features
        self.conv_layer = conv_layer
        self.dropout_rate = dropout
        self.use_residual = use_residual

        # Activation selection
        activations = {
            None: nn.Identity(),
            'relu': nn.ReLU(),
            'prelu': nn.PReLU(),
            'leakyrelu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh()
        }
        self.activation = activations.get(
            activation.lower() if isinstance(activation, str) else activation, nn.ReLU()
        )

        # Dropout
        self.dropout = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()

        # Pooling selection
        pool_dict = {
            'mean': global_mean_pool,
            'max': global_max_pool,
            'add': global_add_pool
        }
        self.pool = pool_dict.get(pooling, global_mean_pool)

        # Build GNN layers
        conv_dict = {
            'GraphConv': GraphConv,
            'GCNConv': GCNConv,
            'MFConv': MFConv,
            'GATConv': GATConv,
            'DMPNN': DMPNNLayer
        }
        conv_class = conv_dict.get(conv_layer)
        if conv_class is None:
            raise ValueError(f"Unknown conv_layer: {conv_layer}")

        self.GNN = nn.ModuleList()
        for i in range(self.num_graph_layers):
            in_ch = self.in_channels if i == 0 else self.hidden_channels
            self.GNN.append(conv_class(in_ch, self.hidden_channels))

        # Build MLP layers
        self.MLP = nn.ModuleList()
        for _ in range(self.num_mlp_layers):
            self.MLP.append(Linear(self.hidden_channels, self.hidden_channels))

        # Output layer
        self.out = Linear(self.hidden_channels, self.out_features)

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass of the GNN.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape [num_nodes, in_channels].
        edge_index : torch_geometric.typing.Adj
            Graph connectivity in COO format with shape [2, num_edges].
        edge_attr : torch.Tensor, optional
            Edge feature matrix of shape [num_edges, edge_features].
            Used only for 'MFConv' and 'GATConv'.
        batch : torch.Tensor, optional
            Batch vector mapping each node to a graph in the batch. Required for
            global pooling.

        Returns
        -------
        torch.Tensor
            Graph-level feature tensor of shape [num_graphs, out_features].
        """
        # Apply GNN layers
        for conv in self.GNN:
            x_res = x
            if isinstance(conv, (GATConv, DMPNNLayer)) and edge_attr is not None:
                x = conv(x, edge_index, edge_attr)
            
            else:
                x = conv(x, edge_index)
            x = self.activation(x)
            x = self.dropout(x)
            if self.use_residual:
                x = x + x_res

        # Global pooling
        x = self.pool(x, batch)

        # MLP
        for lin in self.MLP:
            x = self.activation(lin(x))
            x = self.dropout(x)

        # Output layer
        x = self.out(x)
        return x


class GNNRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        hidden_channels: int = 128,
        num_graph_layers: int = 3,
        num_mlp_layers: int = 1,
        out_dim: int = None,
        conv_layer: str = 'MFConv',
        activation: str = 'relu',
        dropout: float = 0.0,
        pooling: str = 'mean',
        use_residual: bool = False,
        loss: str= 'mse',
        lr: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 50,
        use_cuda=True,
        verbose=False,
    ):
        """
        Graph Neural Network (GNN) for molecular or graph-structured data.

        This model supports multiple types of graph convolution layers and can be
        configured with a flexible number of GNN and MLP layers, activations,
        dropout, residual connections, and pooling methods.

        Parameters
        ----------
        in_channels : int
            Number of input node features per node (e.g., atom features).
        hidden_channels : int
            Number of hidden units in each GNN and MLP layer.
        num_graph_layers : int, default=3
            Number of graph convolution layers.
        num_mlp_layers : int, default=2
            Number of linear layers in the MLP after graph convolutions.
        out_dim : int, default=None
            Number of output features (e.g., regression targets or classes).
        conv_layer : str, default='MFConv'
            Type of graph convolution layer. Options:
            'GraphConv', 'GCNConv', 'MFConv', 'GATConv',
        activation : str or None, default='relu'
            Activation function for hidden layers. Options:
            'relu', 'prelu', 'leakyrelu', 'elu', 'gelu', 'tanh', or None.
         loss : str, default='mse'
             Loss function: 'mse', 'mae', 'huber'.
        dropout : float, default=0.0
            Dropout probability applied after each layer. Set 0.0 to disable.
        pooling : {'mean','max','add'}, default='mean'
            Global pooling method to aggregate node features into a graph-level embedding.
        use_residual : bool, default=False
            If True, adds residual connections between GNN layers.

        Methods
        -------
        forward(x, edge_index, edge_attr=None, batch=None)
            Performs a forward pass through the network.

        Notes
        -----
        - Supports edge features for 'MFConv' and 'GATConv'.
        - Designed for molecular graphs but can be applied to general graphs.
        - Residual connections can improve training stability for deep GNNs.
        """
        
        self.hidden_channels = hidden_channels
        self.num_graph_layers = num_graph_layers
        self.num_mlp_layers = num_mlp_layers
        self.out_dim = out_dim
        self.conv_layer = conv_layer
        self.activation = activation
        self.dropout = dropout
        self.pooling = pooling
        self.use_residual = use_residual
        self.loss = loss
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.verbose = verbose
        
        # --- automatically set the featurizer mode according to conv_layer ---
        if conv_layer.upper() == 'DMPNN':
            self.mode = 'dmpnn'
        else:
            self.mode = 'graph'
    
    def _device(self):
        return torch.device("cuda") if self.use_cuda else torch.device("cpu")
        

    def fit(self, X:List[str], y:np.ndarray):
        
        # Ensure train_y is 2D for multi-output support
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        _, n_targets = y.shape
        if self.out_dim is None:
            self.out_dim = n_targets
        elif self.out_dim != n_targets:
            raise ValueError(f"output_dim mismatch: expected {self.out_dim}, got {n_targets}")
        
        device = self._device()
        
        dataset = featurize_molecules(featurizer='Mol2Graph', 
                                      smiles_list=X, 
                                      labels=y,
                                      use_edges=True,
                                      mode=self.mode)

   
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        
        # Model 
        self.model = GNN(in_channels=dataset[0].x.shape[1],
                          hidden_channels=self.hidden_channels,
                          num_graph_layers=self.num_graph_layers,
                          num_mlp_layers=self.num_mlp_layers,
                          out_features=self.out_dim,
                          conv_layer=self.conv_layer,
                          activation=self.activation,
                          dropout=self.dropout,
                          pooling=self.pooling,
                          use_residual=self.use_residual
                          ).to(device)
        
        # Loss function
        loss_map = {'mse': nn.MSELoss, 'mae': nn.L1Loss, 'huber': nn.SmoothL1Loss}
        self.criterion = loss_map.get(self.loss.lower(), nn.MSELoss)()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0
            for batch in data_loader:
                
                optimizer.zero_grad()
                
                batch = batch.to(device)
                
                out = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                
                loss = self.criterion(out, batch.y)
                
                loss.backward()
                
                optimizer.step()
                
                total_loss += loss.item()
                
            if self.verbose and (epoch % max(1, self.epochs // 10) == 0 or epoch == 1 or epoch == self.epochs):    
                print(f"Epoch {epoch}/{self.epochs} - Loss: {total_loss / len(data_loader):.4f}")
        return self

    def predict(self, X):
        check_is_fitted(self, 'model')
        self.model.eval() 
        device = self._device()
        with torch.no_grad():
            dataset = featurize_molecules(featurizer='Mol2Graph', 
                                          smiles_list=X, 
                                          use_edges=True,
                                          mode=self.mode)
            
            batch = Batch.from_data_list(dataset).to(device)
            out = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        return out.detach().cpu().numpy()
    
#%% Example

# if __name__=='__main__':
    
#     # 'GraphConv', 'GCNConv': GCNConv, 'MFConv', 'GATConv', 'DMPNN'

#     model = GNNRegressor(conv_layer='DMPNN', epochs=100, batch_size=32, verbose=True)
#     smiles = ["CCO", "CCN", "c1ccccc1"]
#     labels = np.array(([0.5, 1.0, 0.9], [10, 7, 4])).T

#     model.fit(smiles, labels)
#     preds = model.predict(smiles)
#     print(preds.flatten())