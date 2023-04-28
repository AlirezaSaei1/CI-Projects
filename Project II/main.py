#import library

import gymnasium as gym
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map


#hyperparameters
gamma = 0.9
size = 16

#selecting action using policy iteration method
def select_action(env):
   state_number = env.observation_space.n
   action_number = env.action_space.n
   
   values_vector = np.zeros(state_number)

   policy = np.ones((state_number, action_number))/action_number

   while True:
      while True:
         delta = 0
         for s in range(state_number):
            v = values_vector[s]
            outer_sum = 0
            for a in range(action_number):
               inner_sum = 0
               for prob, next_state, reward, _ in env.P[s][a]:
                  inner_sum += prob * (reward + gamma * values_vector[next_state])
               outer_sum += policy[s][a] * inner_sum
            values_vector[s] = outer_sum
         delta = max(delta, abs(v - values_vector[s]))
         if delta < 1e-8:
            break
      is_policy_stable = True
      for s in range(state_number):
         old_action = np.argmax(policy[s])
         action_values = []
         for a in range(action_number):
            inner_sum = 0
            for prob, next_state, reward, _ in env.P[s][a]:
               inner_sum += prob * (reward + gamma * values_vector[next_state])
            action_values.append(inner_sum)
         best_action = np.argmax(action_values)
         if old_action != best_action:
            is_policy_stable = False
         policy[s] = np.eye(action_number)[best_action]
      if is_policy_stable:
         return np.argmax(policy, axis=1)


#create Enviroment

env = gym.make("FrozenLake-v1", desc=generate_random_map(size=size), render_mode="human", is_slippery=True)

observation, info = env.reset(seed=42)
max_iter_number = 1000

value = select_action(env)

for _ in range(max_iter_number):

   action = value[observation]

   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:

      observation, info = env.reset()

env.close()