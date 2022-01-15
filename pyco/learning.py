from sklearn.linear_model import Lasso, LinearRegression, Ridge
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import itertools

class GroupLearner:
    
    def __init__(self, t=2):
        self.t = t
        self.coefs = None
    
    def _fit_identity(self, X, y):
        lm = LinearRegression()
        lm.fit(X, y)
        return lm.coef_, lm.intercept_

    
    def fit(self, X, Y, n_groups):
        x = np.identity(n_groups)
        runs = int(X.shape[0] / n_groups)
        
        
        self.coefs = {}
        for combo in itertools.combinations(np.arange(X.shape[1]), 1):
            self.coefs[combo] = []
        #for combo in itertools.combinations(np.arange(X.shape[1]), 2):
        #    self.coefs[combo] = []
        
        # fit model to group
        for i in range(runs):
            y = Y[i * n_groups:(i+1) * n_groups]
            slopes = self._fit_identity(x, y)
            
            for j in range(n_groups):
                ij = i*n_groups + j
                where = np.where(X[ij,:] == 1)[0] 
                not_where = np.where(X[ij,:] == 0)[0] 
                
                w = where[np.random.choice(np.arange(where.shape[0]), size=int(0.9*len(where)))]
                w = np.append(w, np.random.choice(not_where, size=int(0.1*len(where))))
                
                for c in self.coefs:
                    if set(c).issubset(set(w)):
                        self.coefs[c].append(slopes[0][j])
                        
        
            coefs = {c: np.mean(self.coefs[c]) for c in self.coefs}
            cc = {c: 0 if np.isnan(coefs[c]) else coefs[c] for c in coefs}
            plt.bar(['Factor ' + str(c) for c in cc.keys()], [cc[c] for c in cc.keys()])
            plt.xticks(rotation=90)
            plt.axhline(0)
            plt.show()
