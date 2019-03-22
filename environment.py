import numpy as np


class Environment():
    def __init__(self, initial_balance, data):
        self.reset_data(initial_balance, data);
        self.state_size = len(data.columns)
        self.action_size = 3
        
    def get_reward(self):
        reward = self.current_balance - self.initial_balance
        return reward

    def step(self, action):
        
        # actions = 0: stay, 1: buy, 2: sell
        reward = 0
        
        previous_step = self.current_step 
        self.current_step = self.current_step + 1
        
        if action == 0:
            # Do nothing
            self.pos_size[self.current_step] = self.pos_size[previous_step]
            self.current_balance[self.current_step] = self.current_balance[previous_step]
            
        elif action == 1: #Buy
            # Buy back short
            if self.pos_size[previous_step] < 0:
                self.pos_size[self.current_step] = 0
                self.current_balance[self.current_step] = self.current_balance[previous_step] + self.pos_size[previous_step]*self.data.iloc[self.current_step]['Close']  
                a = 1
            # Buy   
            elif self.pos_size[previous_step] == 0:
                self.pos_size[self.current_step] = self.current_balance[previous_step]/self.data.iloc[self.current_step]['Close']
                self.current_balance[self.current_step] = 0

            # Do nothing
            else:
                self.pos_size[self.current_step] = self.pos_size[previous_step]
                self.current_balance[self.current_step] = self.current_balance[previous_step]
            
        elif action == 2:
            # Sell if open position
            if self.pos_size[previous_step] > 0:
                self.current_balance[self.current_step] = self.pos_size[previous_step]*self.data.iloc[self.current_step]['Close']
                self.pos_size[self.current_step] = 0
            # Sell short
            elif self.pos_size[previous_step] == 0:
                self.pos_size[self.current_step] = -1 * self.current_balance[previous_step]/self.data.iloc[self.current_step]['Close']
                self.current_balance[self.current_step] = 2*self.current_balance[previous_step]
            # Do nothing    
            else:
                self.pos_size[self.current_step] = self.pos_size[previous_step]
                self.current_balance[self.current_step] = self.current_balance[previous_step]
                
        self.total_balance[self.current_step] = self.current_balance[self.current_step] + self.pos_size[self.current_step]*self.data.iloc[self.current_step]['Close']
                
        
        self.final_balance = self.total_balance[self.current_step]
        
        profit = self.total_balance[self.current_step]-self.total_balance[previous_step]
        
        reward = profit
        
        if self.current_step == self.num_steps-1:
            done = True
        else:
            done = False

        return self.data.iloc[self.current_step], reward, done
    
    def reset(self):
        self.balance = self.initial_balance
        self.pos_size = np.zeros(self.num_steps)
        self.current_step = 0
        self.current_balance = np.zeros(self.num_steps)
        self.current_balance[0] = self.initial_balance
        self.final_balance = 0
        self.total_balance = np.zeros(self.num_steps)
        self.total_balance[0] = self.initial_balance
        return self.data.iloc[self.current_step]
    
    def reset_data(self, initial_balance, data):
        self.initial_balance = initial_balance
        self.data = data;
        self.num_steps = len(self.data)
        state = self.reset()
        return state