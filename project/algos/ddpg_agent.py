"""
DDPG Agent

This file implements the DDPG algorithm. Students need to complete the code in the update() and get_action() functions.
"""

from pathlib import Path
import copy, time
import utils.common_utils as cu
import torch
import numpy as np
import torch.nn.functional as F

from .agent_base import BaseAgent
from .ddpg_utils import Policy, Critic, ReplayBuffer

def to_numpy(tensor):
    return tensor.cpu().numpy().flatten()

class DDPGAgent(BaseAgent):
    def __init__(self, config=None):
        super(DDPGAgent, self).__init__(config)
        self.device = self.cfg.device  # ""cuda" if torch.cuda.is_available() else "cpu"
        self.name = 'ddpg'
        state_dim = self.observation_space_dim
        self.action_dim = self.action_space_dim
        self.max_action = self.cfg.max_action
        self.lr=self.cfg.lr
      
        self.buffer = ReplayBuffer((state_dim,), self.action_dim, max_size=int(float(self.cfg.buffer_size)))
        
        self.batch_size = self.cfg.batch_size
        self.gamma = self.cfg.gamma
        self.tau = self.cfg.tau
        
        # used to count number of transitions in a trajectory
        self.buffer_ptr = 0
        self.buffer_head = 0 
        self.random_transition = 5000 # collect 5k random data for better exploration
        self.max_episode_steps=self.cfg.max_episode_steps

        # Networks
        # Critic
        self.q = Critic(state_dim=state_dim, action_dim=self.action_dim).to(self.device)  
        self.q_target = copy.deepcopy(self.q).to(self.device)  
        self.q_optim = torch.optim.Adam(self.q.parameters(), lr=float(self.lr))
        # Actor
        self.pi = Policy(state_dim=state_dim,action_dim=self.action_dim, max_action=self.max_action).to(self.device)
        self.pi_target = copy.deepcopy(self.pi).to(self.device)
        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=float(self.lr))

        # Noise for learning:

        self.initial_noise = 0.2 * self.max_action   # start quite exploratory
        self.final_noise   = 0.05 * self.max_action  # settle down later
        self.noise_decay_steps = int(20000)           # decay horizon (tune as needed)
        self.total_steps = 0



    def update(self,):
        """ After collecting one trajectory, update the pi and q for #transition times: """
        info = {}

        # --- START FIX ---
        # Check if the buffer has enough samples to start learning
        # self.buffer_ptr is the total number of transitions added
        if self.buffer_ptr < self.random_transition:
            return {"critic_loss": float('nan'), "actor_loss": float('nan')}  # Return empty info, do not update
        
        # update the network once per transition
        update_iter = self.buffer_ptr - self.buffer_head
        
        # Track losses
        q_loss_list, pi_loss_list = [], []

        for _ in range(update_iter):
            info = self._update()
            q_loss_list.append(info.get('critic_loss', 0))
            pi_loss_list.append(info.get('actor_loss', 0))

        self.buffer_head = self.buffer_ptr
        
        # Return the average loss over the update iteration
        if not q_loss_list:
             return {"critic_loss": float('nan'), "actor_loss": float('nan')}

        return {
            "critic_loss": np.mean(q_loss_list),
            "actor_loss": np.mean(pi_loss_list)
        }
    
    def calculate_target(self, batch):
        with torch.no_grad():
            next_action = self.pi_target(batch.next_state)
            q_tar = self.q_target(batch.next_state, next_action)
            target_q = batch.reward + batch.not_done * self.gamma * q_tar
        return target_q
    
    def calculate_critic_loss(self, current_q, target_q):
        return F.mse_loss(current_q, target_q)
        
    def calculate_actor_loss(self, batch):
        return -self.q(batch.state, self.pi(batch.state)).mean()
    
    def _update(self):
        # get batch data
        batch = self.buffer.sample(self.batch_size, device=self.device) 
        #
        current_q = self.q(batch.state, batch.action)
        target_q = self.calculate_target(batch)

        critic_loss = self.calculate_critic_loss(current_q, target_q)

        # optimise the critic
        self.q_optim.zero_grad()
        critic_loss.backward()
        self.q_optim.step()

        actor_loss = self.calculate_actor_loss(batch=batch)

        # optimise the actor

        self.pi_optim.zero_grad()
        actor_loss.backward()
        self.pi_optim.step()

        cu.soft_update_params(self.q, self.q_target, self.tau)
        cu.soft_update_params(self.pi, self.pi_target, self.tau)

        return {"critic_loss": float(critic_loss.item()),
        "actor_loss": float(actor_loss.item())}

    
    @torch.no_grad()
    def get_action(self, observation, evaluation=False):

        if (self.total_steps < self.random_transition) and not evaluation:
            action = np.random.uniform(-self.max_action, self.max_action, self.action_dim)

            action_out = torch.from_numpy(action).float()

            self.total_steps += 1
            return action_out, {}
        
        if observation.ndim == 1:
            observation = observation[None]  # add batch dim

        x = torch.from_numpy(observation).float().to(self.device)

        # policy action
        action = self.pi(x).cpu()  # shape: [B, action_dim]

        if not evaluation:
            # linearly decay the std from initial_noise -> final_noise
            frac = min(1.0, (self.total_steps - self.random_transition) / float(self.noise_decay_steps))
            # ---
            
            noise_std = self.final_noise + (self.initial_noise - self.final_noise) * (1.0 - frac)
            noise = torch.randn_like(action) * noise_std  # Gaussian action noise
            action = action + noise

        action = action.clamp(-self.max_action, self.max_action)

        self.total_steps += 1
        # return [action_dim] if input was 1D, otherwise [B, action_dim]
        action_out = action[0] if observation.shape[0] == 1 else action
        return action_out, {}



    def record(self, state, action, next_state, reward, done):
        """ Save transitions to the buffer. """
        self.buffer_ptr += 1
        self.buffer.add(state, action, next_state, reward, done)
    
    def train_iteration(self):
        #start = time.perf_counter()
        # Run actual training        
        reward_sum, timesteps, done = 0, 0, False
        # Reset the environment and observe the initial state
        obs, _ = self.env.reset()
        while not done:
            
            # Sample action from policy
            action, _ = self.get_action(obs)

            # Perform the action on the environment, get new state and reward
            next_obs, reward, done, _, _ = self.env.step(to_numpy(action))

            # Store action's outcome (so that the agent can improve its policy)        
            
            done_bool = float(done) if timesteps < self.max_episode_steps else 0 
            self.record(obs, action, next_obs, reward, done_bool)
                
            # Store total episode reward
            reward_sum += reward
            timesteps += 1
            
            if timesteps >= self.max_episode_steps:
                done = True
            # update observation
            obs = next_obs.copy()

        # update the policy after one episode
        #s = time.perf_counter()
        info = self.update()
        #e = time.perf_counter()
        
        # Return stats of training
        info.update({
                    'episode_length': timesteps,
                    'ep_reward': reward_sum,
                    })
        
        end = time.perf_counter()
        return info
        
    def train(self):
        if self.cfg.save_logging:
            L = cu.Logger() # create a simple logger to record stats
        start = time.perf_counter()
        total_step=0
        run_episode_reward=[]
        log_count=0
        
        for ep in range(self.cfg.train_episodes + 1):
            # collect data and update the policy
            train_info = self.train_iteration()
            train_info.update({'episodes': ep})
            total_step+=train_info['episode_length']
            train_info.update({'total_step': total_step})
            run_episode_reward.append(train_info['ep_reward'])
            
            if total_step>self.cfg.log_interval*log_count:
                average_return=sum(run_episode_reward)/len(run_episode_reward)
                if not self.cfg.silent:
                    print(f"Episode {ep} Step {total_step} finished. Average episode return: {average_return}")
                if self.cfg.save_logging:
                    train_info.update({'average_return':average_return})
                    L.log(**train_info)
                run_episode_reward=[]
                log_count+=1

        if self.cfg.save_model:
            self.save_model()
            
        logging_path = str(self.logging_dir)+'/logs'   
        if self.cfg.save_logging:
            Path(logging_path).mkdir(parents=True, exist_ok=True)
            L.save(logging_path, self.seed)
        self.env.close()

        end = time.perf_counter()
        train_time = (end-start)/60
        print('------ Training Finished ------')
        print(f'Total traning time is {train_time}mins')
        
    def load_model(self):
        # define the save path, do not modify
        filepath=str(self.model_dir)+'/model_parameters_'+str(self.seed)+'.pt'
        
        d = torch.load(filepath)
        self.q.load_state_dict(d['q'])
        self.q_target.load_state_dict(d['q_target'])
        self.pi.load_state_dict(d['pi'])
        self.pi_target.load_state_dict(d['pi_target'])
    
    def save_model(self):   
        # define the save path, do not modify
        filepath=str(self.model_dir)+'/model_parameters_'+str(self.seed)+'.pt'
        
        torch.save({
            'q': self.q.state_dict(),
            'q_target': self.q_target.state_dict(),
            'pi': self.pi.state_dict(),
            'pi_target': self.pi_target.state_dict()
        }, filepath)
        print("Saved model to", filepath, "...")
        
        
