from typing import Counter
from gym import spaces
from fmpy.simulation import Input
from numpy.lib.npyio import load
from fmpy.gym_interface import FMI_env_stable
import os
import time
import numpy as np
import numpy as np
from gym import spaces
import os
from time import sleep, time
import pandas as pd
path = os.path.split(__file__)[0] + "/" if len(os.path.split(__file__)[0]) > 0  else ""
fmu_file = path + "Simulationsumgebung.FMUs.FMU_MAGGIE_Energiesystem.fmu"
load_file =  path + "Loadprofiles_MAGGIE.csv"
n_steps =  6 * 24 # one day
_fmu = FMI_env_stable(fmu_file)
loads = pd.read_csv(load_file,sep=";", decimal=",").iloc[range(n_steps)]
loads = loads.set_index("time")

loads.index = pd.DatetimeIndex(loads.index * 1e9)
loads = loads.resample("10Min").mean()
loads.to_csv(path + "test.csv")
stat_max  = list(loads.max() + 0.1)
loads = loads.to_dict(orient='list')

class Maggie(FMI_env_stable):
    relative_tolerance = 10e-5
    heat_buffer = [0.]*6


    heat_chp = 0.
    heat_hp = 0.
    heat_boiler = 0.
    tau = 60 * 10 
    output = [x.name for x in _fmu.model_description.modelVariables if (x.causality == "output" and 'y__' in x.name)]
    _output_to_input = {
            s.replace("init__", "y__") :s 
            for s in [x.name for x in _fmu.model_description.modelVariables if ('init__' in 'y__' in x.name)]
        }
    input_names = [x.name for x in _fmu.model_description.modelVariables if (x.causality == "input" and 'u__' in x.name)]
    statistic_input = loads
    def output_to_input(self, x):
        return self._output_to_input[x]
    def __init__(self):
        super().__init__(fmu_file)
    def _get_action_space(self):
        return spaces.Box( low=np.array([0.]*len(self.action_input)) , high=np.array([1.]*3 + [0.4]*6) ,dtype=np.float32)
    def _get_observation_space(self):
        return spaces.Box( low=np.array([0.]*len(self.output + list(self.statistic_input.keys()))) , high=np.array([100.]*len(self.output) +  stat_max ) ,dtype=np.float32)
    def render(self):
        pass
    def close(self):
        super().close
        if not self.viewer is None: 
            self.viewer.close()

    def reset(self):
        return super().reset()
    def step(self, action):
        # if not self.action_space.contains(action):
        #     raise ValueError
        return super().step(action)
    def policy_simple(self):
        for i in range(0,6):
            if self.get_current(f'y__dezPuffer{i + 1}__degC_low') < 48.:
                # print(f'buffer {i} starts heating...')
                self.heat_buffer[i] = 0.3
            elif self.get_current(f'y__dezPuffer{i + 1}__degC_high') > 62.:
                self.heat_buffer[i] = 0.
                # print(f'buffer {i} stops heating...')
            
        if (self.get_current('y__Puffer__degC__low__BHKW') < 55.):
            # print("chp heating!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            self.heat_chp = 1.
        elif self.get_current('y__Puffer__degC__high__BHKW') > 77.:
            # print("chp not heating")
            self.heat_chp = 0.
        
        if (self.get_current('y__Puffer__degC__low__WP') < 35.):
            self.heat_hp = 1.
        elif self.get_current('y__Puffer__degC__high__WP') > 45.:
            self.heat_hp = 0.
    
        
        if self.get_current('y__Puffer__degC__low__BHKW') < 45.:
            # print("the boiler is on as t is: ", self.get_current('y__Puffer__degC__low__BHKW') )
            self.heat_boiler = 1.
        elif self.get_current('y__Puffer__degC__low__BHKW') > 50.:
            self.heat_boiler = 0.
            # print("the boiler is off as t is: ", self.get_current('y__Puffer__degC__low__BHKW') )

        return np.array([self.heat_chp,self.heat_boiler,self.heat_hp] + self.heat_buffer)
    
    def _reward_policy(self, action:np.array, last_state:np.void)->float:
        """the reward of maggie

        Parameters
        ----------
        action : np.array
            
        last_state : np.void
            the state before the current state

        Returns
        -------
        float
            the reward value
        """
        if self._failed_simulation:
            return -100.

        # where the system is totally fine
        safety_zone_buffer = [
            (self.get_current(f'y__dezPuffer{i + 1}__degC_low') >= 48.) and self.get_current(f'y__dezPuffer{i + 1}__degC_high') <= 62.
            for i in range(6)
        ] 
        way_off_buffer = [
            (self.get_current(f'y__dezPuffer{i + 1}__degC_low') < 38.) and self.get_current(f'y__dezPuffer{i + 1}__degC_high') > 72.
            for i in range(6)
        ]

        safety_zone_chp = (self.get_current('y__Puffer__degC__low__BHKW') >= 55.) and (self.get_current('y__Puffer__degC__low__BHKW') <= 77.)
        way_off_chp = (self.get_current('y__Puffer__degC__low__BHKW') < 45.) and (self.get_current('y__Puffer__degC__low__BHKW') > 90.)
        safety_zone_hp = (self.get_current('y__Puffer__degC__low__WP') >= 35.) and (self.get_current('y__Puffer__degC__low__WP') <= 45.)
        way_off_hp = (self.get_current('y__Puffer__degC__low__WP') < 25.) and (self.get_current('y__Puffer__degC__low__WP') > 55.)
        safety_zone = all(safety_zone_buffer) and safety_zone_chp and safety_zone_hp
        reward = 0.
        if safety_zone:
            production_gain = np.array([1,0,2]) # chp meh..., boiler bad ..., heat pump good
            reward += (production_gain * action).sum()
        else:

            for i in range(6):
                if not safety_zone_buffer[i]:
                    reward -= 2.
            if not safety_zone_hp:
                reward -= 1.
            if not safety_zone_chp:
                reward -= 1.
            
            if way_off_chp or way_off_hp or any(way_off_buffer):
                self.done = True
                print("you are way off")

        return reward


            
class MaggieDiscrete(Maggie):
    def _get_action_space(self):
        return spaces.Discrete(2**(3 + 6))
    def step(self, action):
        # convert the action from single integer encoding to binary encoding
        divider = np.array([2**i for i in range(3 + 6)])
        action = (action//divider)%2 
        # replace 0/1 by 0/0.3 when loading a buffer
        for i in range(3,9):
            action[i] = 0.35 * action[i]

        return super().step(action)

    def policy_simple(self):
        # get the action
        action  = super().policy_simple()
        for i in range(6,10):
            action[i] = 1 if action[i] != 0 else 0
        divider = np.array([2**i for i in range(3 + 6)])
        action = np.array([np.array(divider * action).sum()],dtype=int)
        return action
        
def maggie_test():
    maggie = Maggie()
    done = False
    maggie.reset()
    d= maggie.pdstate
    while not done:
        print("--------------------------------------")
        print("the step: ", maggie._counter)
        print("--------------------------------------")
        action = maggie.policy_simple()
        observation, reward, done, info = maggie.step(action)
        d = pd.concat([d,maggie.pdstate])
    
    d.to_csv("maggie_env_stable_test.csv")
if __name__ == "__main__":
    maggie_test()
    








    
    
    

















