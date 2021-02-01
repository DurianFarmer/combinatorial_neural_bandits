import numpy as np
import torch
import os
import itertools

from tqdm import tqdm
import abc

import torch.nn as nn
if not os.path.exists('regrets'):
    os.mkdir('regrets')

# ### Classes and functions
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
            self.layers = [nn.Linear(size[i], size[i+1], bias=False) for i in range(self.n_layers)]
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
    


















print('Welcome to combinatorial neural bandit world~!!')
users_reaction = input('Input in order: () \n (cf. type "q" to exit!!)')
try:
    --
except e:
    users_reaction = input('Wrong input!! Input in order: () \n (cf. type "q" to exit!!)')

if users_reaction =='1':
    while user_list:
        a= user_list.pop()
        b= random.choice(first_pool)
        first_pool.remove(b) 
        #a[0],b 활용해서 seat update
        seat_pool = []
        for row in c.execute('''SELECT * from seat where cluster_id = ? and owner_id is null''', (b,) ):
            seat_pool.append(row[0])
        occupied_seat = []
        for row in c.execute('''SELECT * from seat where cluster_id = ? and owner_id is not null''',(b,) ):
            occupied_seat.append(row[0])
        sd_pool = []
        for xx in occupied_seat:
            for row in c.execute('''SELECT * from seat where cluster_id = ? and near = ?''',(b,xx)):
                sd_pool.append(row[0])
        mmm_test = set(seat_pool) - set(sd_pool)
        mmm_test = list(mmm_test)
        if len(occupied_seat) <=4:
            selected_seat = random.choice(mmm_test)
        else:
            selected_seat = random.choice(seat_pool)

        c.execute('''UPDATE seat SET owner_id =? WHERE sid=?''', (str(a[0]),selected_seat) )
        print(f'assign {a[1]} to seat {selected_seat}')

        c.execute('''UPDATE cluster SET number_owned =number_owned +1 WHERE cid =?''', (b,) )

        if not first_pool:
            first_pool = []
            for row in c.execute('SELECT distinct cid from cluster where number_of_seat-number_owned >0'):
                first_pool.append(*row)
    print('done')
    conn.commit()
    conn.close()
    
elif users_reaction=='2':
    first_pool = []
    for row in c.execute('SELECT distinct cid from cluster where number_of_seat-number_owned >0'):
        first_pool.append(*row)

    while True:
        print('students that do not have a seat:', pid_list)
        
        if pid_list:            
            print('please type pid')
            mm = input()
            if mm == 'q':
                print('goodbye')
                break            
            elif mm not in pid_list:
                print('select a pid in the list!!')
                continue
                        
            b= random.choice(first_pool)
            first_pool.remove(b)
            seat_pool = []
            for row in c.execute('''SELECT * from seat WHERE cluster_id = ? AND owner_id IS NULL''', (b,) ):
                seat_pool.append(row[0])
            occupied_seat =[]
            for row in c.execute('''SELECT * from seat WHERE cluster_id = ? AND owner_id is not NULL''',(b,)):
                occupied_seat.append(row[0])
            sd_pool =[]
            for xx in occupied_seat:
                for row in c.execute('''SELECT * from seat WHERE cluster_id = ? AND near = ?''',(b,xx)):
                    sd_pool.append(row[0])
            mmm_test = set(seat_pool) - set(sd_pool)
            mmm_test = list(mmm_test)
            if len(occupied_seat) <=4:
                selected_seat = random.choice(mmm_test)
            else:
                selected_seat = random.choice(seat_pool)

            c.execute('''UPDATE seat SET owner_id =? WHERE sid=?''', (str(mm), selected_seat))
            print(f'assign {mm} to seat {selected_seat}')
            c.execute('''UPDATE cluster SET number_owned =number_owned +1 WHERE cid =?''', (b,) )
            conn.commit()
            if not first_pool:
                first_pool = []
                for row in c.execute('SELECT distinct cid from cluster where number_of_seat-number_owned >0'):
                    first_pool.append(*row)

            pid_list.remove(mm)
            if not pid_list:
                print('Every student has a seat. Goodbye.')
                break    

            print('insert done, do you want to quit? (y/n)')
            mmm = input()
            if mmm == 'y':
                print('goodbye')
                break
        else:
            print('Every student has a seat. Goodbye.')
            break
    conn.close()

elif users_reaction == 'q':
    print('goodbye')
