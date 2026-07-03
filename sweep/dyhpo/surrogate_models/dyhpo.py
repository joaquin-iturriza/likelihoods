# Vendored from https://github.com/releaunifreiburg/DyHPO (MIT licence)
# Patches applied on top of original:
#   - budget_dim configurable (was hardcoded +1 scalar, then +2); set via surrogate_config
#   - CNN learning-curve encoder replaced by DeepSets observation encoder:
#       obs_encoder MLP: (budget_dim+1,) → obs_embed_dim, then mean-pool per HP config
#       forward(x, budgets, context_embs) — context_embs pre-computed by encode_contexts()

from copy import deepcopy
import logging
import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import cat

import gpytorch


class FeatureExtractor(nn.Module):
    def __init__(self, configuration):
        super(FeatureExtractor, self).__init__()

        self.configuration = configuration
        self.nr_layers = configuration['nr_layers']
        self.act_func = nn.LeakyReLU()

        # budget_dim = N+1 for N datasets + t_steps (injected by DyHPOSampler)
        initial_features = configuration['nr_initial_features'] + configuration.get('budget_dim', 2)
        self.fc1 = nn.Linear(initial_features, configuration['layer1_units'])
        self.bn1 = nn.BatchNorm1d(configuration['layer1_units'])

        for i in range(2, self.nr_layers):
            setattr(
                self,
                f'fc{i + 1}',
                nn.Linear(configuration[f'layer{i - 1}_units'], configuration[f'layer{i}_units']),
            )
            setattr(
                self,
                f'bn{i + 1}',
                nn.BatchNorm1d(configuration[f'layer{i}_units']),
            )

        # DeepSets observation encoder: φ(budget_vec, val_loss) → obs_embed_dim, then mean-pool
        self.obs_embed_dim = configuration.get('obs_embed_dim', 16)
        obs_input_dim = configuration.get('budget_dim', 2) + 1   # budget_vec concat val_loss
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_input_dim, self.obs_embed_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(self.obs_embed_dim * 2, self.obs_embed_dim),
        )

        setattr(
            self,
            f'fc{self.nr_layers}',
            nn.Linear(
                configuration[f'layer{self.nr_layers - 1}_units'] + self.obs_embed_dim,
                configuration[f'layer{self.nr_layers}_units'],
            ),
        )

    def encode_contexts(self, contexts, device):
        """
        Encode variable-length observation sets via DeepSets (mean-pool after shared MLP).

        Parameters
        ----------
        contexts : list of lists of (budget_norm_tuple, val_loss) pairs
            One list per training/test point. Empty list → zero embedding (no prior obs).
        device : torch.device

        Returns
        -------
        Tensor of shape (batch, obs_embed_dim)
        """
        embs = []
        for obs_list in contexts:
            if obs_list:
                inp = torch.tensor(
                    [[*bvec, vl] for bvec, vl in obs_list],
                    dtype=torch.float32, device=device,
                )
                emb = self.obs_encoder(inp).mean(dim=0)   # (obs_embed_dim,)
            else:
                emb = torch.zeros(self.obs_embed_dim, device=device)
            embs.append(emb)
        return torch.stack(embs, dim=0)   # (batch, obs_embed_dim)

    def forward(self, x, budgets, context_embs):
        # budgets: (N, budget_dim), context_embs: (N, obs_embed_dim) — pre-computed
        x = cat((x, budgets), dim=1)
        x = self.fc1(x)
        x = self.act_func(self.bn1(x))

        for i in range(2, self.nr_layers):
            x = self.act_func(
                getattr(self, f'bn{i}')(
                    getattr(self, f'fc{i}')(x)
                )
            )

        x = cat((x, context_embs), dim=1)
        x = self.act_func(getattr(self, f'fc{self.nr_layers}')(x))
        return x


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DyHPO:
    def __init__(self, configuration, device, dataset_name='unknown', output_path='.', seed=11):
        super(DyHPO, self).__init__()
        self.feature_extractor = FeatureExtractor(configuration)
        self.batch_size = configuration['batch_size']
        self.nr_epochs = configuration['nr_epochs']
        self.early_stopping_patience = configuration['nr_patience_epochs']
        self.refine_epochs = 50
        self.dev = device
        self.seed = seed
        self.model, self.likelihood, self.mll = self.get_model_likelihood_mll(
            configuration[f'layer{self.feature_extractor.nr_layers}_units']
        )

        self.model.to(self.dev)
        self.likelihood.to(self.dev)
        self.feature_extractor.to(self.dev)

        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters(), 'lr': configuration['learning_rate']},
            {'params': self.feature_extractor.parameters(), 'lr': configuration['learning_rate']},
        ])

        self.configuration = configuration
        self.initial_nr_points = 10
        self.iterations = 0
        self.restart = True
        self.logger = logging.getLogger(__name__)

        self.checkpoint_path = os.path.join(output_path, 'dyhpo_surrogate_checkpoints', f'{dataset_name}', f'{seed}')
        os.makedirs(self.checkpoint_path, exist_ok=True)
        self.checkpoint_file = os.path.join(self.checkpoint_path, 'checkpoint.pth')

    def restart_optimization(self):
        self.feature_extractor = FeatureExtractor(self.configuration).to(self.dev)
        self.model, self.likelihood, self.mll = self.get_model_likelihood_mll(
            self.configuration[f'layer{self.feature_extractor.nr_layers}_units'],
        )
        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters(), 'lr': self.configuration['learning_rate']},
            {'params': self.feature_extractor.parameters(), 'lr': self.configuration['learning_rate']},
        ])

    def get_model_likelihood_mll(self, train_size):
        train_x = torch.ones(train_size, train_size).to(self.dev)
        train_y = torch.ones(train_size).to(self.dev)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.dev)
        model = GPRegressionModel(train_x=train_x, train_y=train_y, likelihood=likelihood).to(self.dev)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model).to(self.dev)
        return model, likelihood, mll

    def train_pipeline(self, data, load_checkpoint=False):
        self.iterations += 1
        weights_changed = False

        if load_checkpoint:
            try:
                self.load_checkpoint()
            except FileNotFoundError:
                self.logger.error(f'No checkpoint at {self.checkpoint_file}')

        self.model.train()
        self.likelihood.train()
        self.feature_extractor.train()

        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters(), 'lr': self.configuration['learning_rate']},
            {'params': self.feature_extractor.parameters(), 'lr': self.configuration['learning_rate']},
        ])

        X_train = data['X_train']
        train_budgets  = data['train_budgets']   # shape (N, budget_dim)
        train_contexts = data['train_contexts']  # list of lists of (budget_tuple, val_loss)
        y_train = data['y_train']

        initial_state = self.get_state()
        training_errored = False

        if self.restart:
            self.restart_optimization()
            nr_epochs = self.nr_epochs
            if self.initial_nr_points <= self.iterations:
                self.restart = False
        else:
            nr_epochs = self.refine_epochs

        mse = 0.0
        for epoch_nr in range(0, nr_epochs):
            nr_examples_batch = X_train.size(dim=0)
            if nr_examples_batch == 1:
                continue

            self.optimizer.zero_grad()
            context_embs = self.feature_extractor.encode_contexts(train_contexts, self.dev)
            projected_x = self.feature_extractor(X_train, train_budgets, context_embs)
            self.model.set_train_data(projected_x, y_train, strict=False)
            output = self.model(projected_x)

            try:
                loss = -self.mll(output, self.model.train_targets)
                mse = gpytorch.metrics.mean_squared_error(output, self.model.train_targets)
                loss.backward()
                self.optimizer.step()
            except Exception as training_error:
                self.logger.error(f'Training error: {training_error}')
                self.restart = True
                training_errored = True
                break

        if training_errored:
            self.save_checkpoint(initial_state)
            self.load_checkpoint()

    def predict_pipeline(self, train_data, test_data):
        self.model.eval()
        self.feature_extractor.eval()
        self.likelihood.eval()

        with torch.no_grad():
            train_ctx_embs = self.feature_extractor.encode_contexts(
                train_data['train_contexts'], self.dev
            )
            projected_train_x = self.feature_extractor(
                train_data['X_train'],
                train_data['train_budgets'],
                train_ctx_embs,
            )
            self.model.set_train_data(inputs=projected_train_x, targets=train_data['y_train'], strict=False)
            test_ctx_embs = self.feature_extractor.encode_contexts(
                test_data['test_contexts'], self.dev
            )
            projected_test_x = self.feature_extractor(
                test_data['X_test'],
                test_data['test_budgets'],
                test_ctx_embs,
            )
            preds = self.likelihood(self.model(projected_test_x))

        means = preds.mean.detach().to('cpu').numpy().reshape(-1,)
        stds = preds.stddev.detach().to('cpu').numpy().reshape(-1,)
        return means, stds

    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_file)
        self.model.load_state_dict(checkpoint['gp_state_dict'])
        self.feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
        self.likelihood.load_state_dict(checkpoint['likelihood_state_dict'])

    def save_checkpoint(self, state=None):
        if state is None:
            torch.save(self.get_state(), self.checkpoint_file)
        else:
            torch.save(state, self.checkpoint_file)

    def get_state(self):
        return {
            'gp_state_dict': deepcopy(self.model.state_dict()),
            'feature_extractor_state_dict': deepcopy(self.feature_extractor.state_dict()),
            'likelihood_state_dict': deepcopy(self.likelihood.state_dict()),
        }
