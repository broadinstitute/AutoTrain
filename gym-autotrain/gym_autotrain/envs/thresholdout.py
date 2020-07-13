import nupmy as np

class Thresholdout:
    
    def __init__(self, train, holdout, tolerance=0.01/4, scale_factor=4, keep_log=True):
        self.tolerance = tolerance
        self.T = 4*tolerance
        
        self.eps = lambda: np.random.normal(0, 2*self.tolerance, 1)[0]
        self.gamma = lambda: np.random.normal(0, 4*self.tolerance, 1)[0]
        self.eta = lambda: np.random.normal(0, 8*self.tolerance, 1)[0]

        self.train = train
        self.holdout = holdout
        
        
    def verify(self, phi):
        train_val = phi(self.train)
        holdout_val = phi(self.holdout)
                
        delta = abs(train_val - holdout_val)
        
        if delta > self.T + self.eta():
            return holdout_val + self.eps(), True
        else:
            return train_val, False