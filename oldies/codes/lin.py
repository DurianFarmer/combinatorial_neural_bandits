import numpy as np
import torch
from .ucb_ts import UCB_TS


class Lin(UCB_TS):
    """LinUCB or LinTS.
    """
    def __init__(self,
                 ucb_ts, ## A string. "UCB" for UCB, "TS" for TS
                 bandit,
                 reg_factor=1.0,
                 delta=0.01,
                 bound_theta=1.0,
                 confidence_scaling_factor=1, ## for UCB                 
                 exploration_variance=1, ## for TS
                 throttle=int(1e2),
                 device=torch.device('cpu')
                ):

        # range of the linear predictors
        self.bound_theta = bound_theta
        
        super().__init__(ucb_ts, 
                         bandit, 
                         reg_factor=reg_factor,
                         confidence_scaling_factor=confidence_scaling_factor, ## for UCB
                         exploration_variance=exploration_variance, ## for TS
                         delta=delta,
                         throttle=throttle,
                        )
        self.device = device

    @property
    def approximator_dim(self):
        """Number of parameters used in the approximator.
        """
        return self.bandit.n_features
    
    def update_output_gradient(self):
        """For linear approximators, simply returns the features.
        """
        self.grad_approx = self.bandit.features[self.iteration]
    
    def reset(self):
        """Return the internal estimates
        """
        self.reset_upper_confidence_bounds() ## for UCB
        self.reset_sample_rewards() ## for TS
        self.reset_regrets()
        self.reset_actions()
        self.reset_A_inv()
        self.reset_grad_approx()
        self.iteration = 0

        # randomly initialize linear predictors within their bounds        
        self.theta = np.random.uniform(-1, 1, self.bandit.n_features) * self.bound_theta        

        # initialize reward-weighted features sum at zero
        self.b = np.zeros(self.bandit.n_features)

    @property
    def confidence_multiplier(self):
        """Use exploration variance (nu) instead of confidence scaling factor (gamma)
        """
        return self.confidence_scaling_factor
    

    def train(self):
        """Update linear predictor theta.
        """        
        self.theta = torch.matmul(self.A_inv, self.b)                      
        self.b += torch.sum(torch.Tensor(\
                                  [ self.bandit.rewards[self.iteration][i]*self.bandit.features[self.iteration, self.action][i] \
                                   for i in range(0, self.bandit.n_assortment) ] \
                                 ), dim = 0)                      
            
    def predict(self):
        """Predict reward.
        """
        self.mu_hat[self.iteration] = torch.Tesnor(
            [
                torch.dot(self.bandit.features[self.iteration, a], self.theta) for a in self.bandit.arms
            ]
        )
