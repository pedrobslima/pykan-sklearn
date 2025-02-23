import torch
import numpy as np
from .MultKAN import KAN
from .loss_functions import *
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class KANClassifier(BaseEstimator):
    '''Classe para o modelo KAN, para que se pareça mais com outras classes de
    modelos de AM do scikit-learn

    Parâmetros:
    width : list of int
        Without multiplication nodes: :math:`[n_0, n_1, .., n_{L-1}]` specify the number of neurons in each layer (including inputs/outputs)
        With multiplication nodes: :math:`[[n_0,m_0=0], [n_1,m_1], .., [n_{L-1},m_{L-1}]]` specify the number of addition/multiplication nodes in each layer (including inputs/outputs)
    grid : int
        number of grid intervals. Default: 3.
    k : int
        order of piecewise polynomial. Default: 3.
    mult_arity : int, or list of int lists
        multiplication arity for each multiplication node (the number of numbers to be multiplied)
    noise_scale : float
        initial injected noise to spline.
    base_fun : str
        the residual function b(x). Default: 'silu'
    symbolic_enabled : bool
        compute (True) or skip (False) symbolic computations (for efficiency). By default: True. 
    affine_trainable : bool
        affine parameters are updated or not. Affine parameters include node_scale, node_bias, subnode_scale, subnode_bias
    grid_eps : float
        When grid_eps = 1, the grid is uniform; when grid_eps = 0, the grid is partitioned using percentiles of samples. 0 < grid_eps < 1 interpolates between the two extremes.
    grid_range : list/np.array of shape (2,))
        setting the range of grids. Default: [-1,1]. This argument is not important if fit(update_grid=True) (by default updata_grid=True)
    sp_trainable : bool
        If true, scale_sp is trainable. Default: True.
    sb_trainable : bool
        If true, scale_base is trainable. Default: True.
    device : str
        device
    random_state : int
        random seed
    save_act : bool
        indicate whether intermediate activations are saved in forward pass
    sparse_init : bool
        sparse initialization (True) or normal dense initialization. Default: False.
    auto_save : bool
        indicate whether to automatically save a checkpoint once the model is modified
    state_id : int
        the state of the model (used to save checkpoint)
    ckpt_path : str
        the folder to store checkpoints. Default: './model'
    round : int
        the number of times rewind() has been called
    device : str
    opt : str
        "LBFGS" or "Adam"
    steps : int
        training steps
    loss_fn : function
                loss function

    Exp.: KANClassifier(width=[2,5,2], grid=5, k=3, random_state=1, opt="Adam", steps=20)
    '''
    def __init__(self, width=None, grid=3, k=3, mult_arity = 2, noise_scale=0.3, scale_base_mu=0.0, scale_base_sigma=1.0, base_fun='silu', symbolic_enabled=True, affine_trainable=False, grid_eps=0.02, grid_range=[-1, 1], sp_trainable=True, sb_trainable=True, random_state=1, save_act=True, sparse_init=False, auto_save=False, first_init=True, ckpt_path='./model', state_id=0, round=0, device='cpu', opt="LBFGS", steps=20, loss_fn=kanCELoss(), lr=1.) -> None:
       self.model = KAN(width=width, grid=grid, k=k, mult_arity=mult_arity, noise_scale=noise_scale, scale_base_mu=scale_base_mu, scale_base_sigma=scale_base_sigma, base_fun=base_fun, symbolic_enabled=symbolic_enabled, affine_trainable=affine_trainable, grid_eps=grid_eps, grid_range=grid_range, sp_trainable=sp_trainable, sb_trainable=sb_trainable, seed=random_state, save_act=save_act, sparse_init=sparse_init, auto_save=auto_save, first_init=first_init, ckpt_path=ckpt_path, state_id=state_id, round=round, device=device)
       self.optimizer = opt
       self.steps = steps
       self.loss_fn = loss_fn
       self.learning_rate = lr
       self.random_state = random_state
       self._estimator_type = "classifier"
       self.data = {}
       self.results = {}
       self.accuracy = None
       self.precision = None
       self.recall = None

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
    def train_acc(self):
        return torch.mean((torch.argmax(self.model(self.data['train_input']), dim=1) == self.data['train_label']).float())

    def test_acc(self):
        return torch.mean((torch.argmax(self.model(self.data['test_input']), dim=1) == self.data['test_label']).float())

    def train_prec(self, lbl = 1):
        p_hat = (torch.argmax(torch.softmax(self.model(self.data['train_input']), dim=1), dim=1) == lbl)
        vp = (p_hat & (self.data['train_label'].float() == lbl))
        return (vp.sum()/p_hat.sum()).float()

    def test_prec(self, lbl = 1):
        p_hat = (torch.argmax(torch.softmax(self.model(self.data['test_input']), dim=1), dim=1) == lbl)
        vp = (p_hat & (self.data['test_label'].float() == lbl))
        return (vp.sum()/p_hat.sum()).float()

    def train_recall(self):
        p = (self.data['train_label'] == 1)
        vp = (p & (torch.argmax(torch.softmax(self.model(self.data['train_input']), dim=1), dim=1) == 1))
        return (vp.sum()/p.sum()).float()
    
    def test_recall(self):
        p = (self.data['test_label'] == 1)
        vp = (p & (torch.argmax(torch.softmax(self.model(self.data['test_input']), dim=1), dim=1) == 1))
        return (vp.sum()/p.sum()).float()
    
    def train_f1(self, lbl=1):
        return (self.train_prec(lbl) + self.train_recall())/2
    
    def test_f1(self, lbl=1):
        return (self.test_prec(lbl) + self.test_recall())/2

    # Fit
    # TODO: 
    # - Include all arguments of the original fit function from MultKAN
    # - Add possibility of adding other metrics (?)
    def fit(self, dataset):#X_train, y_train, X_test, y_test): 
        '''Função de treinamento do modelo KAN

        Parâmetros:
        - dataset: <dict[Tensor]> que deve ter as seguintes chaves:
            - "train_input"
            - "train_label"
            - "test_input"
            - "test_label"
        '''
        #if(sorted(list(dataset.keys())) == ['test_input', 'test_label', 'train_input', 'train_label']):
        #    raise KeyError("The provided dataset needs to have the keys: 'train_input', 'train_label', 'test_input', 'test_label'")
        self.is_fitted_ = True
        for key in ['train_input', 'train_label', 'test_input', 'test_label']:
            self.data[key] = self.__dt4kan(dataset[key])
        self.classes_ = self.data['train_label'].unique()

        self.results = self.model.fit(self.data,
                                      opt=self.optimizer,
                                      steps=self.steps,
                                      metrics=(self.train_acc, self.test_acc,
                                               self.train_prec, self.test_prec,
                                               self.train_recall, self.test_recall,
                                               self.train_f1, self.test_f1),
                                      loss_fn=self.loss_fn,
                                      lr=self.learning_rate)
        self.accuracy, self.precision, self.recall, self.f1 = self.results['test_acc'][-1], self.results['test_prec'][-1], self.results['test_recall'][-1], self.results['test_f1'][-1]
        self.classes_ = np.array([i for i in range(self.predict_proba(self.data['test_input'][:2]).shape[1])]) # isso é necessário? n acho q faça mt sentido
        return self

    # Predições:
    def predict(self, new_data:torch.Tensor | np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        new_data = self.__dt4kan(new_data)
        return torch.argmax(torch.softmax(self.model(new_data), dim=1), dim=1).detach().numpy()

    def predict_proba(self, new_data:torch.Tensor | np.ndarray) -> np.ndarray:
        new_data = self.__dt4kan(new_data)
        return torch.softmax(self.model(new_data), dim=1).detach().numpy()

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    # Plotting:
    def plot(self, beta=100):
        self.model.plot(beta=beta) # não sei para que serve o beta

    # TODO: Change this function to return the figure, instead of just showing it right away
    def plot_metric(self, metric:str):
        if(metric not in ('loss', 'train_loss', 'test_loss', 'acc', 'train_acc', 'test_acc', 'prec', 'train_prec', 'test_prec', 'recall', 'train_recall', 'test_recall', 'f1', 'train_f1', 'test_f1')):
            raise ValueError(f"'{metric}' isn't a valid plottable metrics, which are: 'loss', 'train_loss', 'test_loss', 'acc', 'train_acc', 'test_acc', 'prec', 'train_prec', 'test_prec', 'recall', 'train_recall', 'test_recall', 'f1', 'train_f1', 'test_f1'")

        if(len(metric) < 7):
            plot0 = [float(x) for x in self.results['train_'+metric]]
            plot1 = [float(x) for x in self.results['test_'+metric]]
            plt.plot(plot0, label='Training data')
            plt.plot(plot1, label='Test data')
            plt.legend()
        else:
            plot0 = [float(x) for x in self.results[metric]]
            plt.plot(plot0)
        

        title = metric
        if(title[-2:] == 'f1'):
            title += ' score'
        elif(title[-3:] == 'acc'):
            title += 'uracy'
        elif(title[-4:] == 'prec'):
            title += 'ision'
        if(title[0] == 't'):
            i = title.index('_')
            title = f'{title[i+1:]} ({title[:i]})'
        plt.title(title.capitalize())

        plt.show()

