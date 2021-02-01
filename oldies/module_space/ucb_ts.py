import numpy as np
import abc
from tqdm import tqdm

from .utils import inv_sherman_morrison_iter

class UCB_TS(abc.ABC):
    """Base class for UCB and TS methods.
    """
    def __init__(self,
                 ucb_ts, ## A string. "UCB" for UCB, "TS" for TS
                 bandit,
                 reg_factor=1.0,
                 confidence_scaling_factor=1, ## for UCB, gamma
                 exploration_variance=1, ## for TS, nu
                 delta=0.1,
                 training_period=1,
                 throttle=int(1e2),
                ):
        ## select whether UCB or TS
        self.ucb_ts = ucb_ts
        # bandit object, contains features and generated rewards
        self.bandit = bandit
        # L2 regularization strength
        self.reg_factor = reg_factor
        # Confidence bound with probability 1-delta
        self.delta = delta

        # multiplier for the confidence bound            
        self.confidence_scaling_factor = confidence_scaling_factor

        # exploration variance for TS
        self.exploration_variance = exploration_variance
        
        # train approximator only few rounds
        self.training_period = training_period
        
        # throttle tqdm updates
        self.throttle = throttle
        
        self.reset()
    
    ## for UCB
    def reset_upper_confidence_bounds(self):
        """Initialize upper confidence bounds and related quantities.
        """
        if self.ucb_ts == "UCB":
            self.exploration_bonus = np.empty((self.bandit.T, self.bandit.n_arms))
            self.mu_hat = np.zeros((self.bandit.T, self.bandit.n_arms)) 
            self.upper_confidence_bounds = np.ones((self.bandit.T, self.bandit.n_arms))
        else:
            pass

    ## for TS
    def reset_sample_rewards(self):
        """Initialize sample rewards and related quantities.
        """
        if self.ucb_ts == "TS":
            self.sigma_square = np.ones((self.bandit.T, self.bandit.n_arms))
            self.mu_hat = np.zeros((self.bandit.T, self.bandit.n_arms)) 
            self.sample_rewards = np.zeros((self.bandit.T, self.bandit.n_arms, self.bandit.n_samples))
            self.optimistic_sample_rewards = np.zeros((self.bandit.T, self.bandit.n_arms))
        else:
            pass
    
    def reset_regrets(self):
        """Initialize regrets.
        """
        self.regrets = np.empty(self.bandit.T)

    def reset_actions(self):
        """Initialize cache of actions (actions: played set of arms of each round).
        """
        self.actions = np.empty((self.bandit.T, self.bandit.n_assortment)).astype('int')
    
    def reset_A_inv(self):
        """Initialize n_arms square matrices representing the inverses
        of exploration bonus matrices.
        """
        self.A_inv = np.eye(self.approximator_dim)/self.reg_factor        
    
    def reset_grad_approx(self):
        """Initialize the gradient of the approximator w.r.t its parameters.
        """
        self.grad_approx = np.zeros((self.bandit.n_arms, self.approximator_dim))        
        
    def sample_action(self):        
        """Return the action (set of arms) to play based on current estimates
        """
        ## for UCB
        if self.ucb_ts == "UCB":
            a = self.upper_confidence_bounds[self.iteration]        
        ## for TS
        if self.ucb_ts == "TS":
            a = self.optimistic_sample_rewards[self.iteration]

        ind = np.argpartition(a, -1*self.bandit.n_assortment)[-1*self.bandit.n_assortment:]
        s_ind = ind[np.argsort(a[ind])][::-1].astype('int')
        return s_ind                

    @abc.abstractmethod
    def reset(self):
        """Initialize variables of interest.
        To be defined in children classes.
        """
        pass

    @property
    @abc.abstractmethod
    def approximator_dim(self):
        """Number of parameters used in the approximator.
        """
        pass
    
    @property
    @abc.abstractmethod
    def confidence_multiplier(self):
        """Multiplier for the confidence exploration bonus.
        To be defined in children classes.
        """
        pass
    
    @abc.abstractmethod
    def update_confidence_bounds(self):
        """Update the confidence bounds for all arms at time t.
        To be defined in children classes.
        """
        pass

    @abc.abstractmethod
    def update_output_gradient(self):
        """Compute output gradient of the approximator w.r.t its parameters.
        """
        pass
    
    @abc.abstractmethod
    def train(self):
        """Update approximator.
        To be defined in children classes.
        """
        pass
    
    @abc.abstractmethod
    def predict(self):
        """Predict rewards based on an approximator.
        To be defined in children classes.
        """
        pass
    
    ## for UCB
    def update_confidence_bounds(self):
        """Update confidence bounds and related quantities for all set of arms.
        """
        # update self.grad_approx
        self.update_output_gradient()
        
        # UCB exploration bonus
        self.exploration_bonus[self.iteration] = np.array(
            [
                self.confidence_multiplier * np.sqrt(np.dot(self.grad_approx[a], np.dot(self.A_inv, self.grad_approx[a].T))) for a in self.bandit.arms
            ]
        )        
        
        # update reward prediction mu_hat
        self.predict()
        
        # estimated combined bound for reward
        self.upper_confidence_bounds[self.iteration] = self.mu_hat[self.iteration] + self.exploration_bonus[self.iteration]

    ## for TS
    def update_sample_rewards(self):
        """Update sample rewards and related quantities for all set of arms.
        """        
        # update self.grad_approx
        self.update_output_gradient() 
        
        # update sigma_square        
        self.sigma_square[self.iteration] = [self.reg_factor * \
                                             np.dot(self.grad_approx[a], np.dot(self.A_inv, self.grad_approx[a].T)) \
                                             for a in self.bandit.arms]
                
        # update reward prediction mu_hat
        self.predict()
        
        # update sample reward
        self.sample_rewards[self.iteration] = [np.random.normal(loc = self.mu_hat[self.iteration, a], \
                                                                scale = (self.exploration_variance**2) * self.sigma_square[self.iteration, a], \
                                                                size = self.bandit.n_samples) \
                                               for a in self.bandit.arms]        
        
        # update optimistic sample reward for each arm
        self.optimistic_sample_rewards[self.iteration] = np.max(self.sample_rewards[self.iteration], axis=-1)
        
    def update_A_inv(self):
        """Update A_inv by using an iteration of Sherman_Morrison formula
        """
        self.A_inv = inv_sherman_morrison_iter(
            self.grad_approx[self.action],
            self.A_inv
        )               
        
    def run(self):
        """Run an episode of bandit.
        """
        postfix = {
            'total regret': 0.0,
            '% optimal set of arms': 0.0,
        }
        with tqdm(total=self.bandit.T, postfix=postfix) as pbar:
            for t in range(self.bandit.T):                
                ## for UCB
                if self.ucb_ts == "UCB":
                    # update confidence of all set of arms based on observed features at time t
                    self.update_confidence_bounds()
                ## for TS
                if self.ucb_ts == "TS":
                    ## update sample rewards of all set of arms based on observed features at time t
                    self.update_sample_rewards()                
                
                # pick action (set of arm) with the highest boosted estimated reward
                self.action = self.sample_action()
                self.actions[t] = self.action
                # update approximator                          
                self.train() ### lin and neural training are different
                # update exploration indicator A_inv
                self.update_A_inv()
                
                ## compute regret
                self.regrets[t] = self.bandit.best_round_reward[t] - self.bandit.round_reward_function(self.bandit.rewards[t, self.action])                 
                
                # increment counter
                self.iteration += 1
                
                # log
                postfix['total regret'] += self.regrets[t]
                n_optimal_arm = np.sum(
                    np.prod(
                        (self.actions[:self.iteration]==self.bandit.best_super_arm[:self.iteration])*1, 
                        axis=1)                                                          
                )
                postfix['% optimal set of arms'] = '{:.2%}'.format(n_optimal_arm / self.iteration)
                
                if t % self.throttle == 0:
                    pbar.set_postfix(postfix)
                    pbar.update(self.throttle)                      