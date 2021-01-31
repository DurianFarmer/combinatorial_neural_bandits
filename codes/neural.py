import numpy as np
import torch
import torch.nn as nn
from .ucb_ts import UCB_TS
from .utils import Model
    

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
            x = torch.FloatTensor(
                self.bandit.features[self.iteration, a].reshape(1,-1).float()
            ).to(self.device)
            
            self.model.zero_grad()
            y = self.model(x)
            y.backward()
            
            sqrt_m = float(np.sqrt(self.hidden_size))            
            self.grad_approx[a] = torch.cat([
                w.grad.detach().flatten() / sqrt_m 
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
            actions_so_far = self.actions[np.max([0, self.iteration-self.training_window]):self.iteration+1] # this is a matrix

            temp = np.array([self.bandit.features[t, actions_so_far[i]] for i, t in enumerate(iterations_so_far)])
            x_train = torch.FloatTensor(np.reshape(temp, (1,-1,self.bandit.n_features)).squeeze()).to(self.device)

            temp = np.array([self.bandit.rewards[t, actions_so_far[i]] for i, t in enumerate(iterations_so_far)])
            y_train = torch.FloatTensor(np.reshape(temp, (1,-1)).squeeze()).to(self.device)               

            # train mode
            self.model.train()
            for _ in range(self.epochs):
                ## computing the regularization parameter
                tmp = (self.param_to_tensor(self.model.parameters()) - self.init_param).to(torch.device('cpu')).numpy()
                param_diff = torch.from_numpy(np.linalg.norm(tmp))
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
        self.mu_hat[self.iteration] = self.model.forward(torch.FloatTensor(self.bandit.features[self.iteration])).detach().squeeze()
        