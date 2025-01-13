from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from env_hiv_fast import FastHIVPatient

import numpy as np
import os
import random
import time
import torch
import torch.nn as nn

from copy import deepcopy
from evaluate import evaluate_HIV, evaluate_HIV_population

env = TimeLimit(
    env=FastHIVPatient(domain_randomization=True), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dim = env.observation_space.shape[0]
n_actions = env.action_space.n
hidden_dim = 512

DQN_model = nn.Sequential(
    nn.Linear(state_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, n_actions)
).to(device)

config = {'nb_actions': n_actions,
          'learning_rate': 0.001,
          'gamma': 0.95,
          'buffer_size': 100_000,
          'epsilon_min': 0.01,
          'epsilon_max': 1.,
          'epsilon_decay_period': 10_000,
          'epsilon_delay_decay': 100,
          'batch_size': 1024,
          'gradient_steps': 1,
          'update_target_strategy': 'replace', # or 'ema'
          'update_target_freq': 500,
          'update_target_tau': 0.005,
          'criterion': torch.nn.SmoothL1Loss(),
          'monitoring_nb_trials': 50}

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity) # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)

SAVE_PATH = "trained_agent"
os.makedirs(SAVE_PATH, exist_ok=True)

# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    """
    Defines an interface for agents in a simulation or decision-making environment.

    An Agent must implement methods to act based on observations, save its state to a file,
    and load its state from a file. This interface uses the Protocol class from the typing
    module to specify methods that concrete classes must implement.

    Protocols are a way to define formal Python interfaces. They allow for type checking
    and ensure that implementing classes provide specific methods with the expected signatures.
    """
    def __init__(
        self, 
        config=config, 
        model=DQN_model,
    ):
        self.env = env
        self.device = device
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.nb_actions = config['nb_actions']
        self.memory = ReplayBuffer(config['buffer_size'],self.device)
        self.epsilon_max = config['epsilon_max']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_stop = config['epsilon_decay_period']
        self.epsilon_delay = config['epsilon_delay_decay']
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.total_steps = 0
        self.model = model 
        self.criterion = config['criterion']
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.nb_gradient_steps = config['gradient_steps']
        self.target_model = deepcopy(self.model).to(self.device)
        self.update_target_strategy = config['update_target_strategy']
        self.update_target_freq = config['update_target_freq']
        self.update_target_tau = config['update_target_tau']
        
        self.best_model = deepcopy(self.model).to(self.device)
        self.best_score_agent = 0
        self.best_score_agent_dr = 0
        
    def greedy_action(self, network, state):
        with torch.no_grad():
            Q = network(torch.Tensor(state).unsqueeze(0).to(self.device))
            return torch.argmax(Q).item()
        
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, self.gamma, 1-D, QYmax)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
            
    def train(
        self,
    ):
        env = self.env
        max_episode = 500
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        while episode < max_episode:
            start = time.time()
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = np.random.randint(self.nb_actions)
            else:
                action = self.greedy_action(self.model, state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()
            # update target network if needed
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict + (1*tau)*target_state_dict
                self.target_model.load_state_dict(target_state_dict)
            # next transition
            step += 1
            if done or trunc:
                # score_agent = evaluate_HIV(agent=agent, nb_episode=1)
                
                # if score_agent > self.best_score_agent:
                #     if score_agent > 2e10:
                #         score_agent_dr = evaluate_HIV_population(agent=agent, nb_episode=1)
                #         if score_agent_dr > self.best_score_agent_dr:
                #             self.best_score_agent_dr = score_agent_dr
                #             self.best_score_agent = score_agent
                #             self.best_model = deepcopy(self.model).to(self.device)
                #             self.save(SAVE_PATH)
                #             print_score_agent_dr = '{:4.1f}'.format(score_agent_dr)
                #     else:
                #         self.best_score_agent = score_agent
                #         self.best_model = deepcopy(self.model).to(self.device)
                #         self.save(SAVE_PATH)
                #         print_score_agent_dr = 'N/A'
                score_agent_dr = evaluate_HIV_population(agent=agent, nb_episode=1)
                print_score_agent_dr = '{:4.1f}'.format(score_agent_dr)
                if score_agent_dr > self.best_score_agent_dr:
                    self.best_score_agent_dr = score_agent_dr
                    score_agent = evaluate_HIV(agent=agent, nb_episode=1)
                    print_score_agent = '{:4.1f}'.format(score_agent)
                    if score_agent > self.best_score_agent:
                        self.best_score_agent = score_agent
                        self.best_model = deepcopy(self.model).to(self.device)
                        self.save(SAVE_PATH)
                elif score_agent_dr >= 1e10:
                    score_agent = evaluate_HIV(agent=agent, nb_episode=1)
                    print_score_agent = '{:4.1f}'.format(score_agent)
                    if score_agent > self.best_score_agent:
                        self.best_score_agent = score_agent
                        self.best_model = deepcopy(self.model).to(self.device)
                        self.save(SAVE_PATH)
                else:
                    print_score_agent = 'N/A'
                
                end = time.time()
                exec_time = end - start
                episode += 1
                print("Episode ", '{:3d}'.format(episode), 
                      ", epsilon ", '{:6.2f}'.format(epsilon), 
                      ", batch size ", '{:5d}'.format(len(self.memory)), 
                      ", episode return ", '{:4.1f}'.format(episode_cum_reward),
                      ", score agent ", print_score_agent,
                      ", score agent DR ", print_score_agent_dr,
                      ", exec_time ", '{:4.1f}'.format(exec_time),
                      sep='')
                state, _ = env.reset()
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
            else:
                state = next_state
        return episode_return
        
    def act(self, observation, use_random=False):
        """
        Determines the next action based on the current observation from the environment.

        Implementing this method requires processing the observation and optionally incorporating
        randomness into the decision-making process (e.g., for exploration in reinforcement learning).

        Args:
            observation (np.ndarray): The current environmental observation that the agent must use
                                       to decide its next action. This array typically represents
                                       the current state of the environment.
            use_random (bool, optional): A flag to indicate whether the agent should make a random
                                         decision. This is often used for exploration. Defaults to False.

        Returns:
            int: The action to be taken by the agent.
        """
        if use_random:
            return self.env.action_space.sample()
        else:
            action = self.greedy_action(self.model, observation)
        return action

    def save(self, path):
        """
        Saves the agent's current state to a file specified by the path.

        This method should serialize the agent's state (e.g., model weights, configuration settings)
        and save it to a file, allowing the agent to be later restored to this state using the `load` method.

        Args:
            path (str): The file path where the agent's state should be saved.

        """
        torch.save(self.best_model.state_dict(), path + "/dqn.pt")
        print("Agent saved")

    def load(self):
        """
        Loads the agent's state from a file specified by the path (HARDCODED). This not a good practice,
        but it will simplify the grading process.

        This method should deserialize the saved state (e.g., model weights, configuration settings)
        from the file and restore the agent to this state. Implementations must ensure that the
        agent's state is compatible with the `act` method's expectations.

        Note:
            It's important to ensure that neural network models (if used) are loaded in a way that is
            compatible with the execution device (e.g., CPU, GPU). This may require specific handling
            depending on the libraries used for model implementation. WARNING: THE GITHUB CLASSROOM
        HANDLES ONLY CPU EXECUTION. IF YOU USE A NEURAL NETWORK MODEL, MAKE SURE TO LOAD IT IN A WAY THAT
        DOES NOT REQUIRE A GPU.
        """
        load_path = SAVE_PATH + "/dqn.pt"
        self.model.load_state_dict(torch.load(load_path, weights_only=True, map_location=torch.device('cpu')))
        self.target_model = deepcopy(self.model).to('cpu')
        print("Agent loaded")
      
if __name__ == "__main__":  
    agent = ProjectAgent()
    print('Training agent...')
    agent.train()
    print('Agent trained and saved')
