import numpy as np
from scipy import stats
import pandas as pd
from statsmodels.api import families
from statsmodels.formula.api import glm
import warnings
import locfdr.Rfunctions as rf

class PoissonDensity():
    """
    Fit a density distribution using Poisson regression
    """
    def __init__(self, num_breaks = 120, df = 7, fit_type = "natural_spline"):
        assert fit_type in ['natural_spline', 'polynomial']
        self.num_breaks = num_breaks
        self.df = df
        self.fit_type = fit_type
        
    def _poisson_deviance(self, fitted, y, df):
    
        l = np.log(fitted)
        Fl = fitted.cumsum()
        Fr = fitted[::-1].cumsum()
        dev = ((y - fitted) / np.sqrt((fitted + 1)))
        dev = sum(np.power(dev[1:(len(y)-1)], 2)) / (len(y) - 2 - df)

        return dev

    def _basis_transform(self, x, df):
        if self.fit_type == 'natural_spline':
            basismatrix = rf.ns(x, df)
        elif self.fit_type == 'polynomial':
            basismatrix = rf.poly(x, df)
        else:
            raise NotImplementdError
        return basismatrix
    
    def fit(self, data):
        breaks = np.linspace(min(data), max(data), self.num_breaks)
        x = (breaks[1:] + breaks[0:-1]) / 2.
        y = np.histogram(data, bins = len(breaks) - 1)[0]

        basismatrix = self._basis_transform(x, self.df)
        self.fit_ = glm("y ~ basismatrix", data = dict(y=np.matrix(y).transpose(), basismatrix=basismatrix), 
                family=families.Poisson()).fit()

        self.poisson_dev_ = self._poisson_deviance(self.fit_.fittedvalues, y, self.df)

        if self.poisson_dev_ > 1.5:
            warnings.warn("f(z) misfit = " + str(round(self.poisson_dev_, 1)) + ". Rerun with larger df.")
        
        return self
            
    def score_samples(self, data):
        basismatrix = self._basis_transform(data, self.df)
        density = self.fit_.predict(dict(basismatrix=basismatrix))
        return density