# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 23:27:46 2022

@author: Hogan
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import sklearn.linear_model as lm

#%% Cubic Spline Smoothing Tools for No-Arbitrage Smoothing

class CubicSplineSmoother():
    def __init__(self, knot_range=None, n_knots=5, knots_distribution='even'):
        self.n_knots = n_knots
        self.knots_distribution = knots_distribution
        self.knot_range = knot_range
        
    def _knots(self, X):
        if self.knot_range is None:
            if self.knots_distribution == 'quantile':
                quantiles = np.linspace(0.0, 1.0, num=(self.n_knots + 2))[1:-1]
                self.knots = np.quantile(X, quantiles)
            elif self.knots_distribution == 'even':
                self.knots = np.linspace(min(X), max(X), num=(self.n_knots + 2))[1:-1]
        else:
            if self.knots_distribution == 'quantile':
                quantiles = np.linspace(0.0, 1.0, num=(self.n_knots + 2))[1:-1]
                self.knots = np.quantile(X, quantiles)
            elif self.knots_distribution == 'even':
                self.knots = np.linspace(min(self.knot_range), max(self.knot_range), num=(self.n_knots + 2))[1:-1]        
        return self.knots
    
    def _spline_names(self, X):
        name1 = "{}.spline_linear".format(X.name)
        name2 = "{}.spline_quadratic".format(X.name)
        name3 = "{}.spline_cubic".format(X.name)
        rest_names = ["{}.spline.{}".format(X.name, idx) for idx in range(self.n_knots)]
        spline_names = [name1, name2, name3] + rest_names
        return spline_names
    
    def _X_splines(self, X):
        X_spl = np.zeros((len(X), self.n_knots + 3))
        X_spl[:, 0] = X
        X_spl[:, 1] = X_spl[:, 0] * X_spl[:, 0]
        X_spl[:, 2] = X_spl[:, 1] * X_spl[:, 0]
        
        for i, knot in enumerate(self._knots(X), start=3):
            X_spl[:, i] = np.maximum(0, (X - knot)*(X - knot)*(X - knot))
        
        X_splines = pd.DataFrame(X_spl, columns=self._spline_names(X), index=X.index)
        
        return X_splines
    
    def fit(self, X, Y):
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        if isinstance(Y, pd.DataFrame):
            Y = X.iloc[:, 0]
        self.X, self.Y = X.copy(), Y.copy()
        X_splines = self._X_splines(X)
        self.regression = lm.LinearRegression(fit_intercept=True)
        self.regression.fit(X_splines, Y)
        self.Y_smoothed = pd.Series(self.regression.predict(X_splines))
        return
    
    def predict(self, X):
        X_splines = self._X_splines(X)
        return pd.Series(self.regression.predict(X_splines))
    
    def plot(self, figure_size=(16,12)):
        plt.style.use('seaborn')
        plt.rcParams['figure.figsize'] = figure_size
        fig=plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(self.X, self.Y, label='Actual')
        ax.plot(self.X, self.Y_smoothed, label='Smoothed')
        ax.legend()
        
        ax.set_xlabel(self.X.name)
        ax.set_ylabel(self.Y.name)
        
        plt.show()
        return fig


class NaturalCubicSplineSmoother():
    def __init__(self, n_knots=5, knots_distribution='quantile'):
        self.n_knots = n_knots
        self.knots_distribution = knots_distribution
        
    def _knots(self, X):
        if self.knots_distribution == 'quantile':
            quantiles = np.linspace(0.0, 1.0, num=(self.n_knots + 2))[1:-1]
            self.knots = np.quantile(X, quantiles)
        elif self.knots_distribution == 'even':
            self.knots = np.linspace(min(X), max(X), num=(self.n_knots + 2))[1:-1]
        elif self.knots_distribution == 'all':
            self.knots = X[1:-1]
            self.n_knots = len(self.knots)
        else:
            raise TypeError("Incorrect knots distribution type")
        return self.knots
    
    def _spline_names(self, X):
        first_name = "{}_spline_linear".format(X.name)
        rest_names = ["{}_spline_{}".format(X.name, idx)
                     for idx in range(self.n_knots - 2)]
        return [first_name] + rest_names
    
    def _X_splines(self, X):
        X_spl = np.zeros((len(X), self.n_knots - 1))
        X_spl[:, 0] = X
        
        def d(knot_idx, x):
            ppart = lambda t: np.maximum(0, t)
            cube = lambda t: t*t*t
            numerator = (cube(ppart(x - self._knots(X)[knot_idx]))
                            - cube(ppart(x - self._knots(X)[self.n_knots - 1])))
            denominator = self._knots(X)[self.n_knots - 1] - self._knots(X)[knot_idx]
            return numerator / denominator
        
        
        for i in range(0, self.n_knots - 2):
            X_spl[:, i+1] = (d(i, X) - d(self.n_knots - 2, X))
        
        self.X_splines = pd.DataFrame(X_spl, columns=self._spline_names(X), index=self.X.index)
        
        return self.X_splines
    
    
    def fit(self, X, Y):
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        if isinstance(Y, pd.DataFrame):
            Y = X.iloc[:, 0]
        self.X, self.Y = X, Y
        self._X_splines(X)
        self.regression = lm.LinearRegression(fit_intercept=True)
        self.regression.fit(self.X_splines, Y)
        self.Y_smoothed = pd.Series(self.regression.predict(self.X_splines))
        return
    
    def plot(self, figure_size=(16,12)):
        plt.style.use('seaborn')
        plt.rcParams['figure.figsize'] = figure_size
        fig=plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(self.X, self.Y, label='Actual')
        ax.plot(self.X, self.Y_smoothed, label='Smoothed')
        ax.legend()
        
        ax.set_xlabel(self.X.name)
        ax.set_ylabel(self.Y.name)
        
        plt.show()
        return fig