from torch.distributions.laplace import Laplace
import pandas as pd

class Thresholdout:
    def __init__(self, train, holdout, tolerance=0.01/4, scale_factor=4, keep_log=True):
        self.tolerance = tolerance
        
        self.laplace_eps = Laplace(torch.tensor([0.0]), torch.tensor([2*self.tolerance]))
        self.laplace_gamma = Laplace(torch.tensor([0.0]), torch.tensor([4*self.tolerance]))
        self.laplace_eta = Laplace(torch.tensor([0.0]), torch.tensor([8*self.tolerance]))

        self.train = train
        self.holdout = holdout
        
        self.T = 4*tolerance + self.noise(self.laplace_gamma)
        # self.budget = ???
        
        self.keep_log = keep_log
        if keep_log:
            self.log = pd.DataFrame(columns=['GlobStep', 'threshold', 'delta', 'phi_train', 'phi_holdout', 'estimate', 'overfit'])
        
        
    def noise(self, dist):
        return dist.sample().item()
        
    def verify_statistic(self, phi, glob_step=None):
        """
            - phi(dataset) -> statistic: 
              function returns the average of some statistic
        """
        
        train_val = phi(self.train)
        holdout_val = phi(self.holdout)
                
        delta = abs(train_val - holdout_val)
        thresh = self.T + self.noise(self.laplace_eta)
        
        if delta > thresh:
            self.T += self.noise(self.laplace_gamma)
            estimate = holdout_val + self.noise(self.laplace_eps)
        else:
            estimate = train_val
            
        if self.keep_log:
            if glob_step is None: 
                raise ValueException('please provide glob step if logging is on')
            self.log.loc[len(self.log)] = [glob_step, thresh, delta, train_val, holdout_val, estimate, delta > thresh]
            
        return estimate
            