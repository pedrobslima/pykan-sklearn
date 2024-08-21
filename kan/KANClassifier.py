import torch
import numpy as np
from .MultKAN import KAN
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Função de perda:
class kanLoss(torch.nn.CrossEntropyLoss):
  def forward(self, input, target):
        return super().forward(input.type(torch.float64), target.type(torch.long))

# Classe do modelo KAN: (TODO: Maybe change the declare args so the base KAN model is declared inside the KANClassifier?)
class KANClassifier(BaseEstimator):
    '''Classe para o modelo KAN, para que se pareça mais com outras classes de
    modelos de AM do scikit-learn

    Parâmetros:
    - model: modelo original da classe MultKAN() | KAN()

    Exp.: KANnet(KAN(width=[2,2], grid=3, k=3))
    '''
    def __init__(self, model:KAN) -> None:
       self.model = model
       self._estimator_type = "classifier"
       self.data = {}
       self.results = {}
       self.accuracy = None
       self.precision = None
       self.recall = None

    def get_params(self, deep=False):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"width": self.model.width, "grid": self.model.grid, "k": self.model.k}

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

    def test_prec(self, lbl = 1):
        p_hat = (torch.argmax(torch.softmax(self.model(self.data['test_input']), dim=1), dim=1) == lbl)
        vp = (p_hat & (self.data['test_label'].float() == lbl))
        return (vp.sum()/p_hat.sum()).float()

    def test_recall(self):
        p = (self.data['test_label'] == 1)
        vp = (p & (torch.argmax(torch.softmax(self.model(self.data['test_input']), dim=1), dim=1) == 1))
        return (vp.sum()/p.sum()).float()

    # Fit
    # TODO: 
    # - Include all arguments of the original fit function from MultKAN
    # - Add possibility of adding other metrics (?)
    def fit(self, dataset:dict, opt="LBFGS", steps:int=20, loss_fn=kanLoss()):
        '''Função de treinamento do modelo KAN

        Parâmetros:
        - dataset: <dict[Tensor]> que deve ter as seguintes chaves:
            - "train_input"
            - "train_label"
            - "test_input"
            - "test_label"

        - opt: ...
        - steps: ...
        - loss_fn: ...
        '''
        #if(sorted(list(dataset.keys())) == ['test_input', 'test_label', 'train_input', 'train_label']):
        #    raise KeyError("The provided dataset needs to have the keys: 'train_input', 'train_label', 'test_input', 'test_label'")
        self.is_fitted_ = True
        self.data = dataset
        self.classes_ = self.data['train_label'].unique()

        self.results = self.model.fit(self.data,
                                      opt=opt,
                                      steps=steps,
                                      metrics=(self.train_acc,
                                               self.test_acc,
                                               self.test_prec,
                                               self.test_recall),
                                      loss_fn=loss_fn)
        self.accuracy, self.precision, self.recall = self.results['test_acc'][-1], self.results['test_prec'][-1], self.results['test_recall'][-1]
        self.classes_ = np.array([i for i in range(self.predict_proba(self.data['test_input'][:2]).shape[1])])
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
        if(metric not in ('loss', 'train_loss', 'test_loss', 'acc', 'train_acc', 'test_acc', 'test_prec', 'test_recall')):
            raise ValueError(f"'{metric}' isn't a valid plottable metrics, which are: 'loss', 'train_loss', 'test_loss', 'acc', 'train_acc', 'test_acc', 'test_prec', 'test_recall'")

        if((metric == 'loss') or (metric == 'acc')):
            plot0 = [float(x) for x in self.results['train_'+metric]]
            plot1 = [float(x) for x in self.results['test_'+metric]]
            sns.lineplot(y=plot0, x=range(1, len(plot0) + 1))
        else:
            plot1 = [float(x) for x in self.results[metric]]
        sns.lineplot(y=plot1, x=range(1, len(plot1) + 1))

        title = metric
        if(title[-3:] == 'acc'):
            title += 'uracy'
        if(title[0] == 't'):
            if(title[-4:] == 'prec'):
                title += 'ision'
            i = title.index('_')
            title = f'{title[i+1:]} ({title[:i]})'
        plt.title(title.capitalize())

        plt.show()

