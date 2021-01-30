import numpy as np
import itertools
SEED = 5678
np.random.seed(SEED)

class ContextualBandit():
    def __init__(self,
                 T,
                 n_arms,                 
                 n_features,                                  
                 h,                                                   
                 noise_std=1.0,                 
                 n_assortment=1,
                 n_samples=1,
                 round_reward_function=sum,
                ):
        # number of rounds
        self.T = T
        # number of arms
        self.n_arms = n_arms
        # number of features for each arm
        self.n_features = n_features        
        
        # average reward function
        # h : R^d -> R
        self.h = h

        # standard deviation of Gaussian reward noise
        self.noise_std = noise_std
        
        # number of assortment (top-K)
        self.n_assortment = n_assortment                
        
        # (TS) number of samples for each round and arm
        self.n_samples = n_samples
        
        # round reward function
        self.round_reward_function = round_reward_function                
        
        # generate random features
        self.reset()
        

    @property
    def arms(self):
        """Return [0, ...,n_arms-1]
        """
        return range(self.n_arms)
        
    def reset(self):
        """Generate new features and new rewards.
        """
        self.reset_features()
        self.reset_rewards()

    def reset_features(self):
        """Generate normalized random N(0,1) features.
        """
        x = np.random.randn(self.T, self.n_arms, self.n_features)
        x /= np.repeat(np.linalg.norm(x, axis=-1, ord=2), self.n_features).reshape(self.T, self.n_arms, self.n_features)
        self.features = x

    def reset_rewards(self):
        """Generate rewards for each arm and each round,
        following the reward function h + Gaussian noise.
        """
        self.rewards = np.array(
            [
                self.h(self.features[t, k]) + self.noise_std*np.random.randn()\
                for t,k in itertools.product(range(self.T), range(self.n_arms))
            ]
        ).reshape(self.T, self.n_arms)

        ## to be used only to compute regret, NOT by the algorithm itself
        ## self.best_rewards_oracle = np.max(self.rewards, axis=1)
        ## self.best_actions_oracle = np.argmax(self.rewards, axis=1)
        
        ## to be used only to compute regret, NOT by the algorithm itself        
        a = self.rewards
        ind = np.argpartition(a, -1*self.n_assortment, axis=1)[:,-1*self.n_assortment:]        
        s_ind = np.array([list(ind[i][np.argsort(a[i][ind[i]])][::-1]) for i in range(0, np.shape(a)[0])])
        
        self.best_super_arm = s_ind
        self.best_rewards = np.array([a[i][s_ind[i]] for i in range(0,np.shape(a)[0])])
        self.best_round_reward = self.round_reward_function(self.best_rewards)
        