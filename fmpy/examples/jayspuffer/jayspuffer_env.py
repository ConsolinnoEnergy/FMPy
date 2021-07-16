from typing import Counter

from numpy.lib.npyio import load
from fmpy.gym_interface import FMI_env, FMI_env_stable
import os
import time
import numpy as np
import numpy as np
from gym import spaces
import os
from time import sleep, time
import pandas as pd

path = os.path.split(__file__)[0] + "/" if len(os.path.split(__file__)[0]) > 0  else ""
fmu_file = path + "KI_im_Puffer.fmu"

_fmu = FMI_env(fmu_file)
# _CONF['time_step'] = 60 * 60

# _CONF['model_output_names'] = ["add5.y"]

# _CONF['model_input_names'] = [
#     'u__demand_th__1__thermal_power_minus__1',
#     'u__boiler_el__1__electric_power_minus__1'
#     ]


loads = {'u__demand_th__1__thermal_power_minus__1': [10]*24}


class JaysPuffer(FMI_env_stable):
    relative_tolerance = 10e-6
    step_size = 1.
    tau = 60 * 60 # 1 hour
    input_names = ['u__boiler_el__1__electric_power_minus__1', 'u__demand_th__1__thermal_power_minus__1']
    output = ["add5.y"]
    action = np.array([0])
    T_min = 50.
    T_max = 70.
    _env_name = "Jayspuffer"
    price = [0.]*6 + [1.]*6 + [0.]*6 + [1]*6
    statistic_input = loads
    
    def __init__(self):
        super().__init__(fmu_file)
        self.state_hist = []
        self.action_hist = []
        self.viewer = None

    def transform_state(self, value: np.array) -> np.array:
        state = super().transform_state(value)
        return np.array([state[0],state[1], self.price[self._counter], self.action[0]],dtype = float)

    def policy(self):
        if self.state[0] < 50:
            self.action = np.array([1])
        if self.state[0] > 70:
            self.action = np.array([0])
        return self.action

    def reset(self):
        self.action = np.array([0])
        state = super().reset()
        self.state_hist.append(state[0])
        return state
        
    def step(self, action):
        if len(action.shape)== 0:
            action = np.array([action])
        self.action = action
        self.action_hist.append(action)
        if action != np.array([0]):
            action = np.array([20.])
        state, reward, done, info = super().step(action)
        self.state_hist.append(state[0])
        return state, reward, done, info 

    def reward(self, action, old_state):
        T = self.state[0]
        reward = 0.
        if T < (self.T_min - 10):
            reward -= 100
            self.done = True
        if T > (self.T_max + 10):
            reward -= 100
            self.done = True
        if (self.T_min - 10) <= T < self.T_min:
            reward -= 10
        if self.T_max < T <= (self.T_max + 10):
            reward -= 10
        if self.failed_simulation:
            reward -= 100
            self.done = True
        # where we want to be
        if self.T_min <= T <= self.T_max:
            reward += self.price[self._counter]

        return reward


    def _get_action_space(self):
        return spaces.Discrete(2)
    def _get_observation_space(self):
        return spaces.Box(np.array([0,0,0,0]), np.array([100,20,1,1]), dtype=np.float32)

    def render(self):
        pass
    @staticmethod
    def policy_hyst(environment, observation, threshold=2.): # 2. is the optimal threshold 
        T_min = environment.T_min
        T_max = environment.T_max
        T = observation[0]
        last_action = observation[-1]
        if T < T_min + threshold:
            return 1
        if T > T_max - threshold:
            return 0

        return int(last_action)

    @staticmethod
    def policy_price(environment, observation, threshold=2.): # 2. is the optimal threshold 
        T_min = environment.T_min
        T_max = environment.T_max
        T = observation[0]
        last_action = observation[-1]
        price = observation[-2]

        if T < T_min + threshold:
            return 1

        if T > T_max - threshold:
            return 0

        if price > 0:
            return 1
        else:
            return 0 

    @staticmethod
    def policy_price_v2(environment, observation, threshold1=2., threshold2=5.): 

        T_min = environment.T_min 

        T_max = environment.T_max 

        T = observation[0] 

        last_action = observation[-1] # letzte Aktion 

        price = observation[-2] 

        if T < T_min + threshold1: 

            return 1 

        if T > T_max - threshold1: 

            return 0 

        if last_action == 1 and (T >= (threshold2 + T_min)):  

            if price > 0: 

                return 1 

            else: 

                return 0 

        if last_action !=1 and (T <= T_max - threshold2): 

            if price > 0: 

                return 1 

            else: 

                return 0 

        

        return int(last_action) 

    @staticmethod
    def run_episode(environment, policy, render=False, sleep = 0):
        from time import sleep as slope

        observation = environment.reset()
        rewards = 0.
        done = False
        step_count = 0
        while not done:

            action = policy(environment, observation)
            # print("observation: ", observation," action: ", action)
            observation, reward, done, info = environment.step(action)
           
            rewards += reward
            step_count += 1
            if render:
                environment.render()
                slope(sleep)
        

        environment.close()
        return rewards, step_count

    @property
    def n_buckets(self):
        return self._n_buckets

    @n_buckets.setter
    def n_buckets(self, value):
        self._n_buckets = value
        self.discrete_os_win_size = (
           self.observation_space.high[0]
            - self.observation_space.low[0]) / self.n_buckets
        self.DISCRETE_OS_SIZE = [self.n_buckets,self.n_buckets,2,2]

    def get_discrete_state(self,state:np.array)->tuple:
            """given a continous state calculate/count in which window, in other words discrete state you are"""
            discrete_state_T = (float(state[0]) - self.observation_space.low[0]) / self.discrete_os_win_size
            discrete_state_L = (float(state[1]) - self.observation_space.low[1]) / self.discrete_os_win_size
            T = discrete_state_T.astype(int)
            if T == self.n_buckets:
                T -= 1
            L = discrete_state_L.astype(int)
            if L == self.n_buckets:
                L -= 1
          
            return (T,int(state[1]),int(state[2]),L)
    

    @staticmethod
    def policy_q(environment, observation, q_table = None, epsilon = 0., n_buckets  = 15):
        environment.n_buckets = n_buckets
  
        if q_table is None:
            
            q_table = np.zeros(shape=(environment.DISCRETE_OS_SIZE + [environment.action_space.n]))


        
        discrete_state = environment.get_discrete_state(observation)
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
    
        else:
            
            action = np.random.randint(0, 2)
            
        
      
        return action
        
    @staticmethod
    def train_q_table(
        environment,
        episodes,
        q_table = None,
        LEARNING_RATE = 0.01,
        DISCOUNT=0.99,
        epsilon_end = 1,
        n_buckets = 15,
        SAVE_EVERY = 500,
        render =False
        ):
      
        environment.n_buckets = n_buckets
        
        if q_table is None:
            q_table =  100 * 2 * (np.random.random_sample((environment.DISCRETE_OS_SIZE + [environment.action_space.n])) -1/2)
        epsilon = 1.

        epsilon_decay = epsilon / (episodes * epsilon_end)
        ep_rewards = []
        ep_steps = []
        aggr_ep_rewards = {"ep":[], "avg":[], "min":[], "max":[], "steps": []}
        ep_times = []
  
        for episode in range(1,episodes + 1 ):
            episode_reward = 0
            done = False
            observation = environment.reset()
            x = time()
            
            while not done:
         
                
                action = environment.policy_q(environment, observation, q_table, epsilon, n_buckets)
               
                new_observation, reward, done, _ = environment.step(np.array([action]))
                discrete_state = environment.get_discrete_state(observation)
                
                new_discrete_state = environment.get_discrete_state(new_observation)
                episode_reward += reward
                discrete_action = action
                if render and (episode % SAVE_EVERY == 0):
                    environment.render()
                
                if not done:
                    
                    
                    max_future_q = np.max(q_table[new_discrete_state])
    
                    current_q = q_table[discrete_state + (discrete_action,)]
                    new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
                    q_table[discrete_state+(discrete_action,)] = new_q
            
                observation = new_observation
            
            
            y = time()
            ep_times.append(y - x)
            epsilon = max(epsilon - epsilon_decay,0)
            
            if len(ep_rewards)>0 and episode_reward > max(ep_rewards):
                environment._q_table = q_table
                np.save(f"{environment._env_name}_highest.npy",q_table)
            
            ep_rewards.append(episode_reward)
            ep_steps.append(environment._counter)
     
            if not episode % SAVE_EVERY:
                
                
                average_reward = sum(ep_rewards[-SAVE_EVERY:])/len(ep_rewards[-SAVE_EVERY:])
                average_steps = sum(ep_steps[-SAVE_EVERY:])/len(ep_steps[-SAVE_EVERY:])
                average_time = sum(ep_times[-SAVE_EVERY:])/len(ep_times[-SAVE_EVERY:])
                aggr_ep_rewards["ep"].append(episode)
                aggr_ep_rewards["avg"].append(average_reward)
                aggr_ep_rewards["max"].append(max(ep_rewards[-SAVE_EVERY:]))
                aggr_ep_rewards["min"].append(min(ep_rewards[-SAVE_EVERY:]))
                print(f"Episode: {episode} avg: {average_reward} min: {min(ep_rewards[-SAVE_EVERY:])} max {max(ep_rewards[-SAVE_EVERY:])} avg_steps_per_episode {average_steps} avg_time_per_episode {average_time}")
                np.save(f"{os.path.split(__file__)[0]}/q_tables/{environment._env_name}_episode_{episode}.npy", q_table)
        return q_table
    
    def render(self, mode="human"):
        screen_width = 800
        screen_height = 800

        world_height = 100
        world_width = len(self.load)
        scale_y = screen_height / world_height
        scale_x = screen_width / world_width
        xs = np.array([0.0, screen_width])
        from gym.envs.classic_control import rendering

        if self.viewer is None:

            self.viewer = rendering.Viewer(screen_width, screen_height)
            ys = np.array([scale_y * self.T_max] * 2)
            xys = list(zip(xs, ys))

            temperature_max = rendering.make_polyline(xys)
            temperature_max.set_linewidth(2)
            temperature_max.set_color(1.0, 0.0, 0.0)
            self.viewer.add_geom(temperature_max)

            ys = np.array([scale_y * self.T_min] * 2)
            xys = list(zip(xs, ys))

            temperature_min = rendering.make_polyline(xys)
            temperature_min.set_color(0.0, 0.0, 1.0)
            temperature_min.set_linewidth(2)
            self.viewer.add_geom(temperature_min)

        ys = scale_y * np.array(self.state_hist)
        xs = scale_x * np.array(range(len(self.state_hist)))
        xys = list(zip(xs, ys))
        temperature_level = rendering.make_polyline(xys)
        temperature_level.set_linewidth(2)
        self.viewer.add_onetime(temperature_level)

        ys = 25 * scale_y * np.array(self.price[: len(self.state_hist)])
        xs = scale_x * np.array(range(len(self.state_hist)))
        xys = list(zip(xs, ys))
        price_level = rendering.make_polyline(xys)
        price_level.set_linewidth(2)
        price_level.set_color(0.0, 1.0, 0.0)
        self.viewer.add_onetime(price_level)

        ys = 10 * scale_y * np.array(self.load[: len(self.state_hist)])
        xs = scale_x * np.array(range(len(self.state_hist)))
        xys = list(zip(xs, ys))
        price_level = rendering.make_polyline(xys)
        price_level.set_linewidth(2)
        price_level.set_color(1.0, 1.0, 0.0) # yellow load
        self.viewer.add_onetime(price_level)

        ys = 10 * scale_y * np.array(self.action_hist)
        xs = scale_x * np.array(range(len(self.action_hist)))
        xys = list(zip(xs, ys))
        action_level = rendering.make_polyline(xys)
        action_level.set_linewidth(2)
        action_level.set_color(0.0, 1.0, 1.0)
        self.viewer.add_onetime(action_level)
        return self.viewer.render(return_rgb_array= mode == "rgb_array")

    def close(self):
        super().close
        self.viewer.close()




        
if __name__ == "__main__":
    path = os.path.split(__file__)[0]
    if len(path) == 0:
        filename = "KI_im_Puffer.fmu"
    else:
        filename = path + "/KI_im_Puffer.fmu"
    done = False
    fmi=JaysPuffer()
    vars = fmi.model_description.modelVariables

    print("initial state: ", fmi.reset())
    t=time()
    while fmi.done == False:
        action = fmi.policy()
        print("action: ", action)
        state, reward, done, info = fmi.step(action)
        print("state: ",state)
        print("done?: ", done)


    elapsed = time() - t
    print("time for training ",elapsed, " seconds" )
 








