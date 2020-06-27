from torch.distributions.laplace import Laplace



class Thresholdout:
    def __init__(self, train, holdout, tolerance=0.01/4, scale_factor=4):
        self.tolerance = tolerance
        
        self.laplace_eps = Laplace(torch.tensor([0.0]), torch.tensor([2*self.tolerance]))
        self.laplace_gamma = Laplace(torch.tensor([0.0]), torch.tensor([4*self.tolerance]))
        self.laplace_eta = Laplace(torch.tensor([0.0]), torch.tensor([8*self.tolerance]))

        self.train = train
        self.holdout = holdout
        
        self.T = 4*tolerance + self.noise(self.laplace_gamma)
        
    def noise(self, dist):
        return dist.sample().item()
        
    def verify_statistic(self, phi):
        """
            - phi(dataset) -> statistic: 
              function returns the average of some statistic
        """
        
        train_val = phi(self.train)
        holdout_val = phi(self.holdout)
                
        delta = abs(train_val - holdout_val)
        
        if delta > self.T + self.noise(self.laplace_eta):
            self.T += self.noise(self.laplace_gamma)
            return holdout_val + self.noise(self.laplace_eps), delta, False
        else:
            return train_val, delta, True