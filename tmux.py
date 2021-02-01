#!/usr/bin/env python
# coding: utf-8

# # Settings

# In[ ]:


import numpy as np
import torch
import os
import itertools

from tqdm import tqdm
import abc

import torch.nn as nn
# import torch.nn.functional as F

# from codes import *

if not os.path.exists('regrets'):
    os.mkdir('regrets')
    

## Hidden function
SEED = 777
torch.manual_seed(SEED)
np.random.seed(SEED)


# # Experiment descirption
# ### Hidden functions
# - Linear: $h_{1}(\mathbf{x}_{t,i}) = \mathbf{x}_{t,i}^{\top}\mathbf{a}$
# - Quadratic: $h_{2}(\mathbf{x}_{t,i}) = (\mathbf{x}_{t,i}^{\top}\mathbf{a})^{2}$
# - Non-linear: $h_{3}(\mathbf{x}_{t,i}) = \cos(\pi \mathbf{x}_{t,i}^{\top}\mathbf{a})$
# - where $\mathbf{a} \sim N(0,1)$ and then regularized
# 
# ### For each hidden function, compare the following algorithms
# - CombLinUCB
# - CombLinTS
# - CN-UCB
# - CN-TS(1): single reward sample
# - CN-TS(30): optimistic sampling, sample size = 30 (default sample size is 1)
# 
# ### Ablation study of feature dimension *d* and neural network width *m*
# - Default value: $d = 20, m = 20$
# - $d = \{20, 40\}$  for all algorithms

# # Experiment settings

# ### Classes and functions

# In[ ]:


def inv_sherman_morrison_iter(a, A_inv):
    """Inverse of a matrix for combinatorial case.
    """
    temp = A_inv    
    for u in a:                     
        Au = torch.matmul(temp, u)
        temp = temp - torch.ger(Au, Au)/(1+torch.matmul(u.T, Au))    
    return temp       

class Model(nn.Module):
    """Template for fully connected neural network for scalar approximation.
    """
    def __init__(self, 
                 input_size=1, 
                 hidden_size=2,
                 n_layers=1,
                 activation='ReLU',
                 p=0.0,
                ):
        super(Model, self).__init__()
        
        self.n_layers = n_layers
        
        if self.n_layers == 1:
            self.layers = [nn.Linear(input_size, 1)]                        
        else:
            size  = [input_size] + [hidden_size,] * (self.n_layers-1) + [1]
            ##
            self.layers = [nn.Linear(size[i], size[i+1], bias=False)                            for i in range(self.n_layers)]
        self.layers = nn.ModuleList(self.layers)
        
        # dropout layer
        self.dropout = nn.Dropout(p=p)
        
        # activation function
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(negative_slope=0.1)
        else:
            raise Exception('{} not an available activation'.format(activation))
                    
    def forward(self, x):
        for i in range(self.n_layers-1):
            x = self.dropout(self.activation(self.layers[i](x)))
        x = self.layers[-1](x)
        return x   


# In[ ]:


class Bandit():
    def __init__(self,
                 T,
                 n_arms,                 
                 n_features,                                  
                 h,                                                   
                 noise_std=1.0,                 
                 n_assortment=1,
                 n_samples=1,
                 round_reward_function=sum,
                 device=torch.device('cpu')
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
        
        # device
        self.device = device
        
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
        self.features = torch.from_numpy(x).to(self.device)

    def reset_rewards(self):
        """Generate rewards for each arm and each round,
        following the reward function h + Gaussian noise.
        """            
        self.rewards = torch.Tensor(
            [
                self.h( self.features[t, k] ) + self.noise_std*torch.randn(1).to(self.device)\
                for t,k in itertools.product(range(self.T), range(self.n_arms))
            ]
        ).reshape(self.T, self.n_arms)

        ## to be used only to compute regret, NOT by the algorithm itself        
        a = self.rewards.to('cpu').numpy()
        ind = np.argpartition(a, -1*self.n_assortment, axis=1)[:,-1*self.n_assortment:]        
        s_ind = np.array([list(ind[i][np.argsort(a[i][ind[i]])][::-1]) for i in range(0, np.shape(a)[0])])
        
        self.best_super_arm = torch.from_numpy(s_ind).to(self.device)
        self.best_rewards = torch.Tensor([a[i][s_ind[i]] for i in range(0,np.shape(a)[0])]).to(self.device)
        self.best_round_reward = self.round_reward_function(self.best_rewards).to(self.device)


# In[ ]:


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
                 device=torch.device('cpu')
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
        
        # device
        self.device = device
        
        self.reset()
    
    ## for UCB
    def reset_upper_confidence_bounds(self):
        """Initialize upper confidence bounds and related quantities.
        """
        if self.ucb_ts == "UCB":
            self.exploration_bonus = torch.zeros((self.bandit.T, self.bandit.n_arms)).to(self.device)
            self.mu_hat = torch.zeros((self.bandit.T, self.bandit.n_arms)).to(self.device)
            self.upper_confidence_bounds = torch.ones((self.bandit.T, self.bandit.n_arms)).to(self.device)
        else:
            pass

    ## for TS
    def reset_sample_rewards(self):
        """Initialize sample rewards and related quantities.
        """
        if self.ucb_ts == "TS":
            self.sigma_square = torch.ones((self.bandit.T, self.bandit.n_arms)).to(self.device)
            self.mu_hat = torch.zeros((self.bandit.T, self.bandit.n_arms)).to(self.device)
            self.sample_rewards = torch.zeros((self.bandit.T, self.bandit.n_arms, self.bandit.n_samples)).to(self.device)
            self.optimistic_sample_rewards = torch.zeros((self.bandit.T, self.bandit.n_arms)).to(self.device)
        else:
            pass
    
    def reset_regrets(self):
        """Initialize regrets.
        """
        self.regrets = torch.zeros(self.bandit.T).to(self.device)

    def reset_actions(self):
        """Initialize cache of actions (actions: played set of arms of each round).
        """
        self.actions = torch.zeros((self.bandit.T, self.bandit.n_assortment)).to(self.device)
    
    def reset_A_inv(self):
        """Initialize n_arms square matrices representing the inverses
        of exploration bonus matrices.
        """
        self.A_inv = torch.eye(self.approximator_dim).to(self.device)/self.reg_factor        
    
    def reset_grad_approx(self):
        """Initialize the gradient of the approximator w.r.t its parameters.
        """
        self.grad_approx = torch.zeros((self.bandit.n_arms, self.approximator_dim)).to(self.device)
        
    def sample_action(self):        
        """Return the action (set of arms) to play based on current estimates
        """
        ## for UCB
        if self.ucb_ts == "UCB":
            a = self.upper_confidence_bounds[self.iteration].to('cpu').numpy()
        ## for TS
        if self.ucb_ts == "TS":
            a = self.optimistic_sample_rewards[self.iteration].to('cpu').numpy()

        ind = np.argpartition(a, -1*self.bandit.n_assortment)[-1*self.bandit.n_assortment:]
        s_ind = ind[np.argsort(a[ind])][::-1]
        return torch.Tensor(s_ind.copy()).to(self.device)               

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
        self.exploration_bonus[self.iteration] = torch.Tensor(
            [
                self.confidence_multiplier * torch.sqrt(torch.dot(self.grad_approx[a], torch.matmul(self.A_inv, self.grad_approx[a].T))) for a in self.bandit.arms
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
        self.sigma_square[self.iteration] = torch.Tensor([self.reg_factor *                                              torch.dot(self.grad_approx[a], torch.matmul(self.A_inv, self.grad_approx[a].T))                                              for a in self.bandit.arms]).to(self.device)
                
        # update reward prediction mu_hat
        self.predict()
        
        # update sample reward
        self.sample_rewards.to('cpu')
        for a in self.bandit.arms:
            for j in range(self.bandit.n_samples):
                self.sample_rewards[self.iteration][a][j] = np.random.normal(loc = self.mu_hat[self.iteration, a].to('cpu'),
                                                                             scale = (self.exploration_variance**2) * self.sigma_square[self.iteration, a].to('cpu')
                                                                            )                                                                                                                                           
        self.sample_rewards.to(self.device)
        
        # update optimistic sample reward for each arm
        for a in self.bandit.arms:
            self.optimistic_sample_rewards[self.iteration][a] = torch.max(self.sample_rewards[self.iteration][a])
        
    def update_A_inv(self):
        """Update A_inv by using an iteration of Sherman_Morrison formula
        """
        self.A_inv = inv_sherman_morrison_iter(
            self.grad_approx[self.action.to('cpu').numpy()],
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
                self.regrets[t] = self.bandit.best_round_reward[t] - self.bandit.round_reward_function(self.bandit.rewards[t, self.action.to('cpu').numpy()])                 
                
                # increment counter
                self.iteration += 1
                
                # log
                postfix['total regret'] += self.regrets[t].to('cpu').numpy()
                n_optimal_arm = np.sum(
                    np.prod(
                        (self.actions[:self.iteration].to('cpu').numpy()==self.bandit.best_super_arm[:self.iteration].to('cpu').numpy())*1, 
                        axis=1)                                                          
                )
                postfix['% optimal set of arms'] = '{:.2%}'.format(n_optimal_arm / self.iteration)
                
                if t % self.throttle == 0:
                    pbar.set_postfix(postfix)
                    pbar.update(self.throttle)


# In[ ]:


class Neural(UCB_TS):
    """CN-UCB or CN-TS.
    """
    def __init__(self,
                 ucb_ts, ## A string. "UCB" for UCB, "TS" for TS
                 bandit,
                 hidden_size=20,
                 n_layers=2,
                 reg_factor=1.0,
                 delta=0.01,
                 confidence_scaling_factor=1, ## for UCB
                 exploration_variance=1, ## for TS
                 training_window=100,
                 p=0.0,
                 learning_rate=0.01,
                 epochs=1,
                 training_period=1,
                 throttle=1,
                 device=torch.device('cpu'),
                ):

        # hidden size of the NN layers
        self.hidden_size = hidden_size
        # number of layers
        self.n_layers = n_layers
        
        # number of rewards in the training buffer
        self.training_window = training_window
        
        # NN parameters
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        self.device = device
            
        # dropout rate
        self.p = p

        # neural network
        self.model = Model(input_size=bandit.n_features, 
                           hidden_size=self.hidden_size,
                           n_layers=self.n_layers,
                           p=self.p
                          ).to(self.device)        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        
        super().__init__(ucb_ts,
                         bandit, 
                         reg_factor=reg_factor,
                         confidence_scaling_factor=confidence_scaling_factor, ## for UCB
                         exploration_variance=exploration_variance, ## for TS
                         delta=delta,
                         throttle=throttle,
                         training_period=training_period,
                         device=self.device
                        )

    @property
    def approximator_dim(self):
        """Sum of the dimensions of all trainable layers in the network.
        """
        return sum(w.numel() for w in self.model.parameters() if w.requires_grad)
    
    @property
    def confidence_multiplier(self):
        """Constant equal to confidence_scaling_factor
        """
        return self.confidence_scaling_factor
    
    def update_output_gradient(self):
        """Get gradient of network prediction w.r.t network weights.
        """
        for a in self.bandit.arms:
            
            x = self.bandit.features[self.iteration, a].reshape(1,-1).float()                
            
            self.model.zero_grad()
            y = self.model(x)
            y.backward()
                        
            self.grad_approx[a] = torch.cat([
                w.grad.detach().flatten() / np.sqrt(self.hidden_size)
                for w in self.model.parameters() if w.requires_grad]
            ).to(self.device)
            
            
    def reset(self):
        """Reset the internal estimates.
        """
        self.reset_upper_confidence_bounds() ## for UCB
        self.reset_sample_rewards() ## for TS
        self.reset_regrets()
        self.reset_actions()
        self.reset_A_inv()
        self.reset_grad_approx()
        self.iteration = 0

    ## inital parameters
    def set_init_param(self, parameters):
        self.init_param = self.param_to_tensor(parameters)

    ## torch Parameter object to Tensor object
    def param_to_tensor(self, parameters):
        a = torch.empty(1).to(self.device)
        for p in parameters:
            a = torch.cat((a, p.data.flatten()))
        return a[1:].to(self.device)    
        
    def train(self):
        """Train neural approximator.        
        """
        ### train only when training_period occurs
        if self.iteration % self.training_period == 0:                        
            iterations_so_far = range(np.max([0, self.iteration-self.training_window]), self.iteration+1)
            actions_so_far = self.actions[np.max([0, self.iteration-self.training_window]):self.iteration+1].to('cpu').numpy() # this is a matrix            

            temp = torch.cat([self.bandit.features[t, actions_so_far[i]] for i, t in enumerate(iterations_so_far)])
            x_train = torch.reshape(temp, (1,-1,self.bandit.n_features)).squeeze().float().to(self.device)

            temp = torch.cat([self.bandit.rewards[t, actions_so_far[i]] for i, t in enumerate(iterations_so_far)])
            y_train = torch.reshape(temp, (1,-1)).squeeze().float().to(self.device)

            # train mode
            self.model.train()
            for _ in range(self.epochs):
                ## computing the regularization parameter
                tmp = (self.param_to_tensor(self.model.parameters()) - self.init_param).to(torch.device('cpu')).numpy()
                param_diff = np.linalg.norm(tmp)
                regularization = (self.reg_factor*self.hidden_size*param_diff**2)/2

                ## update weight
                y_pred = self.model.forward(x_train).squeeze()
                ### loss = nn.MSELoss()(y_train, y_pred)
                loss = nn.MSELoss(reduction='sum')(y_train, y_pred)/2 + regularization            
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        else:
            pass
                                        
    def predict(self):
        """Predict reward.
        """
        # eval mode
        self.model.eval()        
        self.mu_hat[self.iteration] = self.model.forward(self.bandit.features[self.iteration].float()).detach().squeeze()


# ### Hidden function, bandit, learning algorithm and regret settings

# In[ ]:


h1 = "h1"
h2 = "h2"
h3 = "h3"


# In[ ]:


def experiment(lin_neural, ucb_ts, h_str, n_features=20, hidden_size=100, n_samples=1, save = ''):
    """ kind explanation
    """
    ## Hidden function
    # tmp = np.random.uniform(low=-1.0, high=1.0, size=n_features)
    tmp = np.random.randn(n_features)
    a = torch.from_numpy(tmp / np.linalg.norm(tmp, ord=2)).to(device)

    if h_str == "h1":
        def h(x):
            return torch.dot(x, a).to(device)    
    elif h_str == "h2":
        def h(x):
            return (torch.dot(x, a)**2).to(device)    
    elif h_str == "h3":
        def h(x):
            PI = 3.14
            return torch.cos(PI*torch.dot(x, a)).to(device)
    
    ## Bandit
    bandit = Bandit(T,
                  n_arms,
                  n_features, 
                  h,
                  noise_std=noise_std,
                  n_assortment=n_assortment,
                  n_samples=n_samples,
                  round_reward_function=F,
                  device=device
                 )
    
    ## Learning algorithm and regret
    regrets = np.empty((n_sim, T))
    
    for i in range(n_sim):
        bandit.reset_rewards()
        
        if lin_neural == Neural:
            model = Neural(ucb_ts,
                           bandit,
                           hidden_size,
                           reg_factor=reg_factor,
                           delta=delta,
                           confidence_scaling_factor=confidence_scaling_factor,
                           exploration_variance=exploration_variance,
                           p=p,
                           training_window=training_window,
                           learning_rate=learning_rate,
                           epochs=epochs,
                           training_period=training_period,
                           device=device
                          )
            
            model.set_init_param(model.model.parameters()) # keep initial parameters for regularization
        
        ##TODO
        """
        if lin_neural == Lin:
            model = lin_neural(ucb_ts,
                               bandit,
                               reg_factor=reg_factor,
                               delta=delta,
                               confidence_scaling_factor=confidence_scaling_factor,
                               exploration_variance=exploration_variance
                              )
        """                
        
        model.run()
        regrets[i] = np.cumsum(model.regrets.to('cpu').numpy())
        np.cumsum(model.regrets.to('cpu').numpy())
    if save: # save regrets
        np.save('regrets/' + save, regrets)            


# # Experiment A: (h, d, m) - Algorithm - Regret

# ### Bandit settings

# In[ ]:


T = 1000
n_sim = 10 # number of simulations


n_arms = 20 # N
n_features_default = 20 # d
n_assortment = 4 # K
n_samples = 30 # M, number of samples per each round and arm, for TS

noise_std = 0.01 # noise of reward: xi = noise_std*N(0,1)


def F(x): # round_reward_function
    if x.dim == 1: # if x is a vector
        return torch.sum(x)
    else: # if x is a matrix
        return torch.sum(x, dim=-1)                


# ### Parameter settings

# In[ ]:


reg_factor = 0.5 # lambda
delta = 0.1 # delta
exploration_variance = 1 # nu, for TS and CombLinUCB
confidence_scaling_factor = 1 # gamma, for CN-UCB


# ### Neural network settings

# In[ ]:


hidden_size_default = 20 # m
epochs = 100 # repeat training for each period
training_period = 5 ### training period
training_window = 100
learning_rate = 0.01

p = 0.0 # no dropout

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# In[ ]:


device


# ### (h1)

# In[ ]:


T = 1000


# In[ ]:


experiment(Neural, "UCB", h1, n_features=80, hidden_size=40, save='reg_h1_CNUCB_80_40')


# In[ ]:


experiment(Neural, "TS", h1, n_features=80, hidden_size=40, save='reg_h1_CNTS_80_40')


# In[ ]:


experiment(Neural, "TS", h1, n_features=80, hidden_size=40, n_samples=30, save='reg_h1_CNTSOpt_80_40')


# ### (h2)

# In[ ]:


T = 1000


# In[ ]:


experiment(Neural, "UCB", h2, n_features=80, hidden_size=40, save='reg_h2_CNUCB_80_40')


# In[ ]:


experiment(Neural, "TS", h2, n_features=80, hidden_size=40, save='reg_h2_CNTS_80_40')


# In[ ]:


experiment(Neural, "TS", h2, n_features=80, hidden_size=40, n_samples=30, save='reg_h2_CNTSOpt_80_40')


# ### (h3)

# In[ ]:


T = 2000


# In[ ]:


experiment(Neural, "UCB", h3, n_features=80, hidden_size=40, save='reg_h3_CNUCB_80_40')


# In[ ]:


experiment(Neural, "TS", h3, n_features=80, hidden_size=40, save='reg_h3_CNTS_80_40')


# In[ ]:


experiment(Neural, "TS", h3, n_features=80, hidden_size=40, n_samples=30, save='reg_h3_CNTSOpt_80_40')


# ### (h1, 20, 20)

# In[ ]:


experiment(Neural, "UCB", h1, n_features=20, hidden_size=20, save='reg_h1_CNUCB_20_20')


# In[ ]:


experiment(Neural, "TS", h1, n_features=20, hidden_size=20, save='reg_h1_CNTS_20_20')


# In[ ]:


experiment(Neural, "TS", h1, n_features=20, hidden_size=20, n_samples=50, save='reg_h1_CNTSOpt_20_20')


# ### (h2, 20, 20)

# In[ ]:


experiment(Neural, "UCB", h2, n_features=20, hidden_size=20, save='reg_h2_CNUCB_20_20')


# In[ ]:


experiment(Neural, "TS", h2, n_features=20, hidden_size=20, save='reg_h2_CNTS_20_20')


# In[ ]:


experiment(Neural, "TS", h2, n_features=20, hidden_size=20, n_samples=50, save='reg_h2_CNTSOpt_20_20')


# ### (h3, 20, 20)

# In[ ]:


T = 2000


# In[ ]:


experiment(Neural, "UCB", h3, n_features=20, hidden_size=20, save='reg_h3_CNUCB_20_20')


# In[ ]:


experiment(Neural, "TS", h3, n_features=20, hidden_size=20, save='reg_h3_CNTS_20_20')


# In[ ]:


experiment(Neural, "TS", h3, n_features=20, hidden_size=20, n_samples=50, save='reg_h3_CNTSOpt_20_20')


# In[ ]:


T = 1000


# ### (h1, 100, 60)

# In[ ]:


experiment(Neural, "UCB", h1, n_features=100, hidden_size=60, save='reg_h1_CNUCB_100_60')


# In[ ]:


experiment(Neural, "TS", h1, n_features=100, hidden_size=60, save='reg_h1_CNTS_100_60')


# In[ ]:


experiment(Neural, "TS", h1, n_features=100, hidden_size=60, n_samples=50, save='reg_h1_CNTSOpt_100_60')


# ### (h2, 100, 60)

# In[ ]:


experiment(Neural, "UCB", h2, n_features=100, hidden_size=60, save='reg_h2_CNUCB_100_60')


# In[ ]:


experiment(Neural, "TS", h2, n_features=100, hidden_size=60, save='reg_h2_CNTS_100_60')


# In[ ]:


experiment(Neural, "TS", h2, n_features=100, hidden_size=60, n_samples=50, save='reg_h2_CNTSOpt_100_60')


# ### (h3, 100, 60)

# In[ ]:


experiment(Neural, "UCB", h3, n_features=100, hidden_size=60, save='reg_h3_CNUCB_100_60')


# In[ ]:


experiment(Neural, "TS", h3, n_features=100, hidden_size=60, save='reg_h3_CNTS_100_60')


# In[ ]:


experiment(Neural, "TS", h3, n_features=100, hidden_size=60, n_samples=50, save='reg_h3_CNTSOpt_100_60')

