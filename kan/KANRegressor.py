import torch
import numpy as np
from .MultKAN import KAN
from .loss_functions import kanMSELoss
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


class KANRegressor(BaseEstimator): # I could probably make a parent class KANEstimator for KANClassifier and KANRegressor, but I don't think it wouldn't change anything significant 
    # Still not usable
    def __init__(self, width=None, grid=3, k=3, mult_arity = 2, noise_scale=0.3, scale_base_mu=0.0, scale_base_sigma=1.0, base_fun='silu', symbolic_enabled=True, affine_trainable=False, grid_eps=0.02, grid_range=[-1, 1], sp_trainable=True, sb_trainable=True, random_state=1, save_act=True, sparse_init=False, auto_save=False, first_init=True, ckpt_path='./model', state_id=0, round=0, device='cpu', opt="LBFGS", steps=20, loss_fn=kanMSELoss(), lr=1.) -> None:
       self.model = KAN(width=width, grid=grid, k=k, mult_arity=mult_arity, noise_scale=noise_scale, scale_base_mu=scale_base_mu, scale_base_sigma=scale_base_sigma, base_fun=base_fun, symbolic_enabled=symbolic_enabled, affine_trainable=affine_trainable, grid_eps=grid_eps, grid_range=grid_range, sp_trainable=sp_trainable, sb_trainable=sb_trainable, seed=random_state, save_act=save_act, sparse_init=sparse_init, auto_save=auto_save, first_init=first_init, ckpt_path=ckpt_path, state_id=state_id, round=round, device=device)
       self.optimizer = opt
       self.steps = steps
       self.loss_fn = loss_fn
       self.learning_rate = lr
       self.random_state = random_state
       #self.model = model
       self._estimator_type = "regressor"
       self.data = {}
       self.results = {}
       self.mse = None
       self.rmse = None
       self.mae = None
       self.r2 = None

    def get_params(self, deep=False):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"width": self.model.width, "grid": self.model.grid, "k": self.model.k,
                "random_state": self.random_state,
                "mult_arity": self.model.mult_arity, "base_fun": self.model.base_fun_name,
                "symbolic_enabled": self.model.symbolic_enabled, "affine_trainable": self.model.affine_trainable,
                "grid_eps": self.model.grid_eps, "grid_range": self.model.grid_range,
                "sp_trainable": self.model.sp_trainable, "sb_trainable": self.model.sb_trainable,
                "device": self.model.device, "save_act": self.model.save_act,
                "auto_save": self.model.auto_save, "round": self.model.round}
    
    # Transformação de dataset:
    def __dt4kan(self, datarray):
        if isinstance(datarray, np.ndarray):
            return torch.from_numpy(datarray).float()
        elif isinstance(datarray, torch.Tensor):
            return datarray
        return torch.from_numpy(np.array(datarray)).float()
    
    # Métricas
    def train_rmse(self):
        pass

    def test_rmse(self):
        pass

    def train_mae(self):
        pass

    def test_mae(self):
        pass

    def train_r2(self):
        pass

    def test_r2(self):
        pass

    # Fit
    def fit(self, dataset):
        self.is_fitted_ = True
        for key in ['train_input', 'train_label', 'test_input', 'test_label']:
            self.data[key] = self.__dt4kan(dataset[key])
        #self.classes_ = self.data['train_label'].unique()

        self.results = self.model.fit(self.data,
                                      opt=self.optimizer,
                                      steps=self.steps,
                                      metrics=(self.train_rmse, self.test_rmse,
                                               self.train_mae, self.test_mae,
                                               self.train_r2, self.test_r2),
                                      loss_fn=self.loss_fn,
                                      lr=self.learning_rate)
        #self.mse, self.rmse, self.mae, self.r2 = 
        return self