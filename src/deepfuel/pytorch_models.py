# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 09:56:07 2025

@author: Rodolfo Freitas
"""
import math

import gpytorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted

# --- Set default dtype to float32 globally ---
torch.set_default_dtype(torch.float32)


#%% Neural Network

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 64), output_dim=1, activation_cls=nn.ReLU):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation_cls())  # instantiate per layer
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)  # returns (batch, output_dim)


class DeepNet(BaseEstimator, RegressorMixin):

    def __init__(
        self,
        input_dim=None,
        hidden_dims=(128, 64),
        hidden_activation='tanh',
        out_dim=None,
        lr=1e-3,
        epochs=200,
        loss='mse',
        batch_size=64,
        use_cuda=True,
        verbose=False,
    ):
        """
           Deep feedforward neural network regressor using PyTorch.
        
           This class provides a scikit-learn compatible wrapper around a PyTorch fully 
           connected network for regression tasks.
        
           Parameters
           ----------
           input_dim : int, optional
               Number of input features. If None, inferred from training data.
           hidden_dims : tuple of int, default=(128,64)
               Number of units in each hidden layer.
           hidden_activation : str, default='tanh'
               Activation function for hidden layers. Options: 
               'relu', 'prelu', 'leakyrelu', 'elu', 'gelu', 'tanh', None.
           out_dim : int, default=1
               Number of output dimensions.
           lr : float, default=1e-3
               Learning rate for optimizer.
           epochs : int, default=200
               Number of training epochs.
           loss : str, default='mse'
               Loss function: 'mse', 'mae', 'huber'.
           batch_size : int, default=32
               Mini-batch size for training.
           use_cuda : bool, default=True
               Whether to use GPU if available.
           verbose : bool, default=True
               Print training progress.
        
           Methods
           -------
           fit(X, y)
               Train the network on input features X and targets y.
           predict(X)
               Predict targets for input features X.
        """
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.lr = lr
        self.epochs = epochs
        self.hidden_activation = hidden_activation
        self.batch_size = batch_size
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.verbose = verbose
        self.loss = loss

        # Map string to activation class
        activation_map = {
            None: nn.Identity,
            'relu': nn.ReLU,
            'prelu': nn.PReLU,
            'leakyrelu': nn.LeakyReLU,
            'elu': nn.ELU,
            'gelu': nn.GELU,
            'tanh': nn.Tanh
        }
        self.activation_cls = activation_map.get(hidden_activation.lower() if hidden_activation else None, nn.Tanh)

        # placeholders
        self._is_fitted = False

    def _device(self):
        return torch.device("cuda") if self.use_cuda else torch.device("cpu")

    def fit(self, X, y):
        X = check_array(X)
        y = check_array(y, ensure_2d=False, ensure_all_finite=False)  # allow 1D targets
        
        # Ensure train_y is 2D for multi-output support
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            
        n_samples, n_features = X.shape
        _, n_targets = y.shape
        if self.input_dim is None:
            self.input_dim = n_features
        elif self.input_dim != n_features:
            raise ValueError(f"input_dim mismatch: expected {self.input_dim}, got {n_features}")
        
        if self.out_dim is None:
            self.out_dim = n_targets
        elif self.out_dim != n_targets:
            raise ValueError(f"output_dim mismatch: expected {self.out_dim}, got {n_targets}")

        device = self._device()

        # Convert to torch tensors
        train_x = torch.tensor(X, dtype=torch.float32, device=device)
        train_y = torch.tensor(y, dtype=torch.float32, device=device)

        dataset = torch.utils.data.TensorDataset(train_x, train_y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Build model
        self.model = Net(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.out_dim,
            activation_cls=self.activation_cls
        ).to(device)

        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Loss function
        loss_map = {'mse': nn.MSELoss, 'mae': nn.L1Loss, 'huber': nn.SmoothL1Loss}
        self.criterion = loss_map.get(self.loss.lower(), nn.MSELoss)()
        
      

        # Training loop
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                y_pred = self.model(batch_x)
                loss = self.criterion(y_pred, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch_x.size(0)

            if self.verbose and (epoch == 1 or epoch == self.epochs or epoch % max(1, self.epochs // 10) == 0):
                print(f"Epoch {epoch}/{self.epochs} - Loss: {total_loss / n_samples:.4f}")

        
        self._is_fitted = True
        return self

    def predict(self, X):
        check_is_fitted(self, "_is_fitted")
        X = check_array(X)
        device = self._device()
        test_x = torch.tensor(X, dtype=torch.float32, device=device)

        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(test_x).cpu().numpy()
        # Return 1D if single output
        if self.out_dim == 1:
            y_pred = y_pred.ravel()
        return y_pred

#%% Probabilistic conditional generative model
class CADGM(BaseEstimator, RegressorMixin):

    def __init__(
        self,
        input_dim=None,
        hidden_dims_AE=(100, 100, 100),
        hidden_dims_T=(100, 100),
        hidden_activation='tanh',
        latent_dim=1,
        lambda_=1.5,
        beta=0.5,
        out_dim=None,
        lr=1e-3,
        epochs=200,
        batch_size=64,
        n_discriminator_steps=1,
        use_cuda=True,
        verbose=False,
    ):
        """
        Conditional Adversarial Deep Generative Model (CADGM) for regression with uncertainty estimation.
    
        This model combines ideas from Variational Autoencoders (VAE) and Generative Adversarial Networks (GANs)
        to learn a conditional generative distribution p(y|x) for regression tasks. The model consists of:
    
            - Encoder: q(z|x,y), maps input x and output y to latent space z
            - Decoder: p(y|x,z), reconstructs y from x and latent z
            - Discriminator: distinguishes real vs generated y conditioned on x
    
        Supports multi-output regression and Monte Carlo sampling for predictive uncertainty.
    
        Parameters
        ----------
        input_dim : int, optional
            Number of input features. If None, inferred from training data.
        hidden_dims_AE : tuple of int, default=(100,100,100)
            Hidden layer sizes for Encoder and Decoder networks.
        hidden_dims_T : tuple of int, default=(100,100)
            Hidden layer sizes for Discriminator network.
        hidden_activation : str, default='tanh'
            Activation function for hidden layers. Options: 'relu', 'prelu', 'leakyrelu', 'elu', 'gelu', 'tanh', None.
        latent_dim : int, default=1
            Dimensionality of latent space z.
        lambda_ : float, default=1.5
            Weighting factor for entropic regularization term in generator loss.
        beta : float, default=0.5
            Weighting factor for reconstruction loss in generator loss.
        out_dim : int, optional
            Output dimensionality. If None, inferred from training data.
        lr : float, default=1e-3
            Learning rate for Adam optimizer.
        epochs : int, default=200
            Number of training epochs.
        batch_size : int, default=32
            Mini-batch size for training.
        n_discriminator_steps : int, default=1
            Number of discriminator updates per generator update (multi-step training).
        use_cuda : bool, default=True
            Whether to use GPU if available.
        verbose : bool, default=False
            If True, prints training progress.
    
        Attributes
        ----------
        Encoder : torch.nn.Module
            Encoder network q(z|x,y).
        Decoder : torch.nn.Module
            Decoder network p(y|x,z).
        Discriminator : torch.nn.Module
            Discriminator network distinguishing real vs fake (x,y) pairs.
        _is_fitted : bool
            Flag indicating if the model has been fitted.
    
        Methods
        -------
        fit(X, y)
            Train the CADGM on input data X and targets y.
        compute_G_loss(x, y, z)
            Compute generator loss (KL + entropic + reconstruction).
        compute_T_loss(x, y, z)
            Compute discriminator loss.
        sampling(X, n_samples=1000, batch_size=128)
            Draw Monte Carlo samples from p(y|x) given inputs X.
        predict(X, n_samples=1000, return_std=False)
            Compute predictive mean (and optional standard deviation) for inputs X.
        """
        
        self.input_dim = input_dim
        self.hidden_dims_AE = hidden_dims_AE
        self.hidden_dims_T = hidden_dims_T
        self.latent_dim = latent_dim
        self.lambda_ = lambda_
        self.beta = beta
        self.out_dim = out_dim
        self.lr = lr
        self.epochs = epochs
        self.hidden_activation = hidden_activation
        self.batch_size = batch_size
        self.n_discriminator_steps = n_discriminator_steps  
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.verbose = verbose
        

        # Map string to activation class
        activation_map = {
            None: nn.Identity,
            'relu': nn.ReLU,
            'prelu': nn.PReLU,
            'leakyrelu': nn.LeakyReLU,
            'elu': nn.ELU,
            'gelu': nn.GELU,
            'tanh': nn.Tanh
        }
        self.activation_cls = activation_map.get(hidden_activation.lower() if hidden_activation else None, nn.Tanh)

        # placeholders
        self._is_fitted = False

    def _device(self):
        return torch.device("cuda") if self.use_cuda else torch.device("cpu")
    
    # Compute generator loss
    def compute_G_loss(self, x, y, z): 
        # Prior: p(z)
        z_prior = z
        # Decoder: p(y|x,z)
        y_pred = self.Decoder(torch.cat((x, z_prior),dim=1))        
        # Encoder: q(z|x,y)
        z_encoder = self.Encoder(torch.cat((x, y_pred),dim=1))
        # Discriminator output
        T_pred = self.Discriminator(torch.cat((x, y_pred),dim=1))
        # Estimated KL-divergence 
        KL = torch.mean(T_pred)
        # Entropic regularization
        log_q = -torch.mean(torch.square(z_prior-z_encoder)) / z.shape[1]  # scale by latent dim
        # Reconsturction Loss
        log_p = torch.mean(torch.square(y - y_pred))
        # Generator loss 
        loss_G = KL + (1.0-self.lambda_)*log_q + self.beta * log_p
        
        return loss_G, KL, log_p

    # Compute discriminator loss
    def compute_T_loss(self, x, y, z): 
        # Prior: p(z)
        z_prior = z
        # Decoder: p(y|x,z)
        y_pred = self.Decoder(torch.cat((x, z_prior),dim=1))               
        
        # Discriminator loss
        T_real = torch.sigmoid(self.Discriminator(torch.cat((x, y), dim=1)))
        T_fake = torch.sigmoid(self.Discriminator(torch.cat((x, y_pred.detach()), dim=1)))
        
        T_loss = -torch.mean(torch.log(1.0 - T_real + 1e-8) + torch.log(T_fake + 1e-8))
        
        return T_loss

    def fit(self, X, y):
        X = check_array(X)
        y = check_array(y, ensure_2d=False)  # allow 1D targets
      
        # Ensure train_y is 2D for multi-output support
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_samples, n_features = X.shape
        _, n_targets = y.shape
        if self.input_dim is None:
            self.input_dim = n_features
        elif self.input_dim != n_features:
            raise ValueError(f"input_dim mismatch: expected {self.input_dim}, got {n_features}")
        
        if self.out_dim is None:
            self.out_dim = n_targets
        elif self.out_dim != n_targets:
            raise ValueError(f"output_dim mismatch: expected {self.out_dim}, got {n_targets}")

        device = self._device()

        # Convert to torch tensors
        train_x = torch.tensor(X, dtype=torch.float32, device=device)
        train_y = torch.tensor(y, dtype=torch.float32, device=device)

        # Ensure train_y is 2D for multi-output support
        if train_y.ndim == 1:
            train_y = train_y.view(-1, self.out_dim)

        dataset = torch.utils.data.TensorDataset(train_x, train_y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Build model
        # Encoder: q(z|x,y)
        self.Encoder = Net(input_dim=self.input_dim + self.out_dim,
                    hidden_dims=self.hidden_dims_AE,
                    output_dim=self.latent_dim,
                    activation_cls=self.activation_cls).to(device)
        
        # Decoder: p(y|x,z)
        self.Decoder = Net(input_dim=self.input_dim + self.latent_dim,
                    hidden_dims=self.hidden_dims_AE,
                    output_dim=self.out_dim,
                    activation_cls=self.activation_cls).to(device)
        
        # Discriminator (P|x,y)
        self.Discriminator = Net(input_dim=self.input_dim + self.out_dim,
                    hidden_dims=self.hidden_dims_T,
                    output_dim=1,
                    activation_cls=self.activation_cls).to(device)

        # Optimizer
        optimizer_T = torch.optim.Adam(self.Discriminator.parameters(), lr=self.lr)
        optimizer_G = torch.optim.Adam(list(self.Decoder.parameters())+list(self.Encoder.parameters()), lr=self.lr)

        # Training loop
        for epoch in range(1, self.epochs + 1):
            self.Encoder.train()
            self.Decoder.train()
            self.Discriminator.train()
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                Z = torch.randn(batch_x.shape[0], self.latent_dim, device=device)

                # Multi-step discriminator updates
                for _ in range(self.n_discriminator_steps):
                    optimizer_T.zero_grad()
                    T_loss = self.compute_T_loss(batch_x, batch_y, Z)
                    T_loss.backward()
                    optimizer_T.step()

                # Update generator
                optimizer_G.zero_grad()
                G_loss, KL_loss, mse_loss = self.compute_G_loss(batch_x, batch_y, Z)
                G_loss.backward()
                optimizer_G.step()
                

            if self.verbose and (epoch == 1 or epoch == self.epochs or epoch % max(1, self.epochs // 10) == 0):
                print(f"Epoch {epoch}/{self.epochs} , Generator Loss: {G_loss:.4f}, MSE: {mse_loss:.4f}, Discriminator Loss: {T_loss:.4f}")

        # Save fitted model
        self._is_fitted = True
        return self
    
    def sampling(self, X, n_samples=1000, batch_size=128):
        check_is_fitted(self, "_is_fitted")
        X = check_array(X)
        device = self._device()
        self.Decoder.eval()

        test_x = torch.tensor(X, dtype=torch.float32, device=device)
        samples = []

        for start in range(0, n_samples, batch_size):
            curr_batch = min(batch_size, n_samples - start)
            Z = torch.randn(test_x.shape[0], curr_batch, self.latent_dim, device=device)
            Z = Z.permute(1, 0, 2)  # [batch, n_samples, latent_dim]
            batch_samples = [self.Decoder(torch.cat((test_x, z), dim=1)).detach() for z in Z]
            samples.extend(batch_samples)

        samples = torch.stack(samples)
        return samples.cpu().numpy()

    def predict(self, X, n_samples=1000, return_std=False):
        samples = self.sampling(X, n_samples) 
        predictive_mean = samples.mean(0)
        predictive_std = samples.std(0)

        if return_std:
            return predictive_mean, predictive_std
        return predictive_mean



#%% Bayesian Neural Network
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_std=1.0):
        super().__init__()

        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_logvar = nn.Parameter(torch.full((out_features, in_features), -5.0))

        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_logvar = nn.Parameter(torch.full((out_features,), -5.0))

        self.prior_std = prior_std

    def forward(self, x):
        w_std = torch.exp(0.5 * self.weight_logvar)
        b_std = torch.exp(0.5 * self.bias_logvar)

        weight = self.weight_mu + w_std * torch.randn_like(w_std)
        bias = self.bias_mu + b_std * torch.randn_like(b_std)

        return F.linear(x, weight, bias)

    def kl_divergence(self):
        prior_var = self.prior_std ** 2
        prior_logvar = 2 * math.log(self.prior_std)
        kl_w = 0.5 * torch.sum(
            (self.weight_mu**2 + torch.exp(self.weight_logvar)) / prior_var
            - 1 - self.weight_logvar + prior_logvar
        )

        kl_b = 0.5 * torch.sum(
            (self.bias_mu**2 + torch.exp(self.bias_logvar)) / prior_var
            - 1 - self.bias_logvar + 2 * math.log(self.prior_std)
        )

        return kl_w + kl_b

class BayesianNet(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims=(128, 64),
        output_dim=1,
        activation_cls=nn.ReLU,
        prior_std=1.0
    ):
        super().__init__()

        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(BayesianLinear(in_dim, h, prior_std))
            layers.append(activation_cls())
            in_dim = h

        self.net = nn.Sequential(*layers)
        self.mean_head = BayesianLinear(in_dim, output_dim, prior_std)
        self.logvar_head = BayesianLinear(in_dim, output_dim, prior_std)

    def forward(self, x):
        x = self.net(x)
        mean = self.mean_head(x)
        log_var = torch.clamp(self.logvar_head(x), -10, 5)
        var = torch.exp(log_var)
        return mean, var

    def kl_divergence(self):
        kl = 0.0
        for m in self.modules():
            if isinstance(m, BayesianLinear):
                kl += m.kl_divergence()
        return kl


class BayesianNNRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, 
                 input_dim=None, 
                 hidden_dims=(128,64), 
                 hidden_activation='tanh',
                 out_dim=None, 
                 lr=1e-3, 
                 epochs=100, 
                 batch_size=64, 
                 use_cuda=True, 
                 verbose=False):
        
        
        """
        Scikit-learn wrapper for a Bayesian neural network regressor.
    
        Supports predictive uncertainty estimation by returning predictive mean and
        standard deviation.
    
        Parameters
        ----------
        input_dim : int, optional
            Number of input features. If None, inferred from training data.
        hidden_dims : tuple of int, default=(128,64)
            Hidden layer sizes.
        hidden_activation : str, default='tanh'
            Activation function for hidden layers.
        out_dim : int, default=None
            Number of regression targets.
        lr : float, default=1e-3
            Learning rate.
        epochs : int, default=100
            Number of training epochs.
        batch_size : int, default=32
            Training batch size.
        use_cuda : bool, default=True
            Use GPU if available.
        verbose : bool, default=True
            Print training progress.
    
        Methods
        -------
        fit(X, y)
            Train the Bayesian neural network.
        predict(X, n_samples=1000, return_std=False)
            Predict target values for X. If return_std=True, also returns predictive standard deviation.
        """
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.hidden_activation = hidden_activation
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.verbose = verbose

        activation_map = {
            None: nn.Identity,
            'relu': nn.ReLU,
            'prelu': nn.PReLU,
            'leakyrelu': nn.LeakyReLU,
            'elu': nn.ELU,
            'gelu': nn.GELU,
            'tanh': nn.Tanh
        }
        self.activation_cls = activation_map.get(hidden_activation.lower() if hidden_activation else None, nn.Tanh)

        self._is_fitted = False

    def _device(self):
        return torch.device("cuda") if self.use_cuda else torch.device("cpu")

    def fit(self, X, y):
        X = check_array(X)
        y = check_array(y, ensure_2d=False)
        
        # Ensure train_y is 2D for multi-output support
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_samples, n_features = X.shape
        _, n_targets = y.shape
        if self.input_dim is None:
            self.input_dim = n_features
        elif self.input_dim != n_features:
            raise ValueError(f"input_dim mismatch: expected {self.input_dim}, got {n_features}")
        
        if self.out_dim is None:
            self.out_dim = n_targets
        elif self.out_dim != n_targets:
            raise ValueError(f"output_dim mismatch: expected {self.out_dim}, got {n_targets}")

        device = self._device()
        train_x = torch.tensor(X, dtype=torch.float32, device=device)
        train_y = torch.tensor(y, dtype=torch.float32, device=device)

        dataset = torch.utils.data.TensorDataset(train_x, train_y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = BayesianNet(self.input_dim, 
                            self.hidden_dims, 
                            self.out_dim, 
                            self.activation_cls).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.GaussianNLLLoss()
        
        # scaling factor in front of the KL divergence in the ELBO
        self.beta = 1.0 / len(dataloader.dataset)
        
        for epoch in range(1, self.epochs+1):
            total_loss = 0
            self.model.train()
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                y_mean, y_var = self.model(batch_x)
                loss = self.criterion(y_mean, batch_y, y_var) + self.beta * self.model.kl_divergence()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch_x.size(0)
            if self.verbose and (epoch==1 or epoch==self.epochs or epoch % max(1, self.epochs//10)==0):
                print(f"Epoch {epoch}/{self.epochs} - ELBO Loss: {total_loss/n_samples:.4f}")

        self._is_fitted = True
        return self
    
    def sampling(self, X, n_samples=1000):
        check_is_fitted(self, "_is_fitted")
        X = check_array(X)
        device = self._device()
        self.model.eval()

        test_x = torch.tensor(X, dtype=torch.float32, device=device)
        samples_mean = []
        samples_var = []

        for _ in range(n_samples):
            mean, var = self.model(test_x)
            samples_mean.append(mean.detach())
            samples_var.append(var.detach())

        samples_mean = torch.stack(samples_mean)
        samples_var = torch.stack(samples_var)
        return samples_mean.cpu().numpy(), samples_var.cpu().numpy()

    def predict(self, X, n_samples=1000, return_std=False):
        means, vars_ = self.sampling(X, n_samples)
        predictive_mean = means.mean(0)
        aleatoric = vars_.mean(0)
        epistemic = means.var(0)
        predictive_std = np.sqrt(aleatoric + epistemic)

        if return_std:
            return predictive_mean, predictive_std
        return predictive_mean

    
#%% Deep Kernel Learning
# -------------------------
# GPyTorch Exact GP model
# -------------------------

class DKLExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, feature_extractor, ard_num_dims=None, kernel='RBF'):
        super().__init__(train_x, train_y, likelihood)
        
        if train_y.ndim == 1:
            self.batch_size = 1
        else:
            _, self.batch_size = train_y.shape
        
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([self.batch_size]))
        
        # Covariance module
        if kernel == 'RBF':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims, batch_shape=torch.Size([self.batch_size]))
            )
        elif kernel == 'Matern_05':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(ard_num_dims=ard_num_dims, batch_shape=torch.Size([self.batch_size]),nu=0.5)
            )
        elif kernel == 'Matern_15':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(ard_num_dims=ard_num_dims, batch_shape=torch.Size([self.batch_size]),nu=1.5)
            )
        elif kernel == 'Matern_25':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(ard_num_dims=ard_num_dims, batch_shape=torch.Size([self.batch_size]),nu=2.5)
            )
        else:
            raise ValueError(f"Unsupported kernel: {kernel}")
        
        self.feature_extractor = feature_extractor
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.scale_to_bounds(features)
        mean_x = self.mean_module(features)
        covar_x = self.covar_module(features)
        if self.batch_size == 1:
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        else: 
            return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(gpytorch.distributions.MultivariateNormal(mean_x, covar_x)) 


# -------------------------
# Sklearn-style DKL Regressor
# -------------------------
class DKLRegressor(BaseEstimator, RegressorMixin):

    def __init__(self,
                 input_dim=None,
                 feature_hidden_dims=(128, 64),
                 feature_out_dim=32,
                 hidden_activation='tanh',
                 kernel='RBF',
                 lr=1e-3,
                 epochs=200,
                 use_cuda=True,
                 verbose=False,
                 jitter=1e-6):
        
        self.input_dim = input_dim
        self.feature_hidden_dims = feature_hidden_dims
        self.feature_out_dim = feature_out_dim
        self.hidden_activation = hidden_activation
        self.kernel = kernel
        self.lr = lr
        self.epochs = epochs
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.verbose = verbose
        self.jitter = jitter
        
        """
        Scikit-learn wrapper for Deep Kernel Learning regression using GPyTorch.
    
        Uses a neural network feature extractor combined with an exact GP.
    
        Parameters
        ----------
        input_dim : int, optional
            Input feature dimension. Inferred if None.
        feature_hidden_dims : tuple, default=(128,64)
            Hidden layers for feature extractor network.
        feature_out_dim : int, default=32
            Output dimension of feature extractor (input to GP kernel).
        hidden_activation : str, default='tanh'
            Activation function for feature extractor.
        kernel : str, default='RBF'
            Kernel type for GP ('RBF' or 'Matern').
        lr : float, default=1e-3
            Learning rate for optimizer.
        epochs : int, default=200
            Number of training epochs.
        use_cuda : bool, default=True
            Use GPU if available.
        verbose : bool, default=True
            Print training progress.
        jitter : float, default=1e-6
            Jitter added to covariance for numerical stability.
    
        Methods
        -------
        fit(X, y)
            Fit the DKL model to training data.
        predict(X, return_std=False)
            Predict outputs for X. If return_std=True, returns predictive standard deviation.
        sample_from_posterior(X, n_samples=1000)
            Sample from the posterior distribution over X.
        predict_std(X)
            Shortcut to get predictive standard deviation only.
        """
        # Map string to activation class
        activation_map = {
            None: nn.Identity,
            'relu': nn.ReLU,
            'prelu': nn.PReLU,
            'leakyrelu': nn.LeakyReLU,
            'elu': nn.ELU,
            'gelu': nn.GELU,
            'tanh': nn.Tanh
        }
        self.activation_cls = activation_map.get(hidden_activation.lower() if hidden_activation else None, nn.Tanh)
        
        self._is_fitted = False

    def _device(self):
        return torch.device("cuda") if self.use_cuda else torch.device("cpu")

    def fit(self, X, y):
        X = check_array(X)
        y = check_array(y, ensure_2d=False)
        

        n_samples, n_features = X.shape
        if self.input_dim is None:
            self.input_dim = n_features
        elif self.input_dim != n_features:
            raise ValueError(f"input_dim mismatch: expected {self.input_dim}, got {n_features}")
        
        device = self._device()

        train_x = torch.tensor(X, dtype=torch.float32, device=device)
        train_y = torch.tensor(y, dtype=torch.float32, device=device)

        # Build feature extractor
        feature_extractor = Net(
            input_dim=self.input_dim,
            hidden_dims=self.feature_hidden_dims,
            output_dim=self.feature_out_dim,
            activation_cls=self.activation_cls
        ).to(device)
        
        # Define the likelihood for multi-output support
        if train_y.ndim == 1:
            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        else:
            _, n_targets = train_y.shape
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=n_targets).to(device)
        
        
        model = DKLExactGPModel(train_x, 
                                train_y, 
                                likelihood,
                                feature_extractor,
                                ard_num_dims=self.feature_out_dim,
                                kernel=self.kernel
                                ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        # Training loop
        for epoch in range(1, self.epochs + 1):
            model.train()
            likelihood.train()
            optimizer.zero_grad()
            output = model(train_x)
            loss = -self.criterion(output, train_y)
            loss.backward()
            optimizer.step()

            if self.verbose and (epoch == 1 or epoch == self.epochs or epoch % max(1, self.epochs // 10) == 0):
                try:
                    lengthscale = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
                except Exception:
                    lengthscale = None
                noise = likelihood.noise.detach().cpu().numpy() if hasattr(likelihood, "noise") else None
                print(f"Epoch {epoch}/{self.epochs} - Loss: {loss.item():.6f} - noise: {noise} - lengthscale shape: {None if lengthscale is None else lengthscale.shape}")

        self.model = model
        self.likelihood_ = likelihood
        self.feature_extractor_ = feature_extractor
        self.train_X_ = X
        self.train_y_ = y
        self._is_fitted = True
        return self

    def predict(self, X, return_std=False):
        check_is_fitted(self, "_is_fitted")
        X = check_array(X)
        device = self._device()
        test_x = torch.tensor(X, dtype=torch.float32, device=device)

        self.model.eval()
        self.likelihood_.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(self.jitter):
            pred_dist = self.likelihood_(self.model(test_x))
            mean = pred_dist.mean.cpu().numpy()
            var = pred_dist.variance.cpu().numpy()
        
        # Ensure sklearn compatibility    
        if self.train_y_.ndim == 1:
            mean = mean.reshape(-1, 1)
            var = var.reshape(-1, 1)
        if return_std:
            return mean, np.sqrt(var)
        return mean

    def sample_from_posterior(self, X, n_samples=1000):
        check_is_fitted(self, "_is_fitted")
        X = check_array(X)
        device = self._device()
        test_x = torch.tensor(X, dtype=torch.float32, device=device)
        self.model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(self.jitter):
            pred_dist = self.model(test_x)
            samples = pred_dist.rsample(torch.Size([n_samples]))
        return samples.cpu().numpy()

    def predict_std(self, X):
        _, std = self.predict(X, return_std=True)
        return std