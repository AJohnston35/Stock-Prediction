** NOTES **

# Markov Decision Process - agent interacts with an environment, tries to fulfill an objective, and learns how to correct actions in order to achieve that goal.

# Agent -action-> Environment -state&reward-> Agent

# Agent can take actions:

# In this case, BUY, SELL, and HOLD a stock

# The action affects the environment in some way, and hence forms a new state

# The state is the current aspects of the environment that are sent back to the agent

# In this case, current price of the stock, and predicted price of the stock (and percentage returns)

# The agent receives the reward from the environment

# In this case, the percentage returns made or lost in the trade

# For equations:

# S = set of states

# A = set of actions

# R = set of rewards

# t = time step

# S t = the represented state at a specific time step

# A t = the action taken at a specific time step

# (S t, A t) = a state action pair; the action that correlates with the specific state it was take under (they are both referenced at the same time step)

# Every state has a value, and every action in a specific state has a value

# The 'value' of an action in a specific state is simply how good taking this action is in order to maximize the reward

# Transition probabilities:

# Agent must be able to predict what the reward from an action will be, and what state we will end up in after taking this action

# Formula:

# p(s',r|s,a) = Pr{S t+1 = s', R t+1 = r|S t = s, A t = a}

# In psuedo equation, the probability of state prime (any chosen state) actually being the next state (at time step t+1), and any reward actually being the reward we get, is based off of the preceding state at St, and the preceding action at At

# The return is the sum of all future rewards

# G t = R t+1 + R t+2 + R t+3 + ... + R T , where T is the final time step

# A series of time steps is called an episode

# Q-learning prioritizes speed to maximize short term rewards

# We do this by taking a discount from rewards that we will receive in the future.

# This is defined as discount rate, or gamma (y)

# G t = R t+1 + yR t+2 + y^2R t+3 + y^3R t+4 + ...

# (Factored) G t = R t+1 + yG t+1

# The discounted expected reward at any time step is simply the reward you will get after this next action, plus the discounted expected reward at the next time step

# For Q-learning, the policy is simply to choose the action that has the highest discounted expected return

# Action-value function: the discounted expected return of being in a state, choosing an action, and following a policy from then onwards

# q pi (s,a) = E [G t|S t = s, A t = a]

# Capital E means expected

# This equation just means: the value of taking an action in a state, and following a policy onwards is equal to the discounted expected return, considering we take this specific action in this specific state.

# A policy is which actions an agent takes in different states. The policy for Q-learning is to choose the action that has the highest q-value from the possible actions that we can take in any given state.

# Optimal action-value function gives the largest expected return achievable from taking any action in the future states

# Bellman Equation:

# q*(s,a) = E[R t+1 + ymaxq*(s',a')]

# The reward we will get after taking an action, plus the action in the next state that has the highest maximum expected return

# We only look at the next state because the q-value of a state-action pair represents the highest expected discounted reward ever achievable after taking said action

# Epsilon-greedy strategy defines the equation for exploration vs exploitation

# Define a variable epsilon, E. This variable is set to 1 and will slowly decay over time. In the beginning, the agent's actions will be completely random.

# As epsilon decays, the agent will be more likely to exploit the knowledge it has gained

# Learning rate, or alpha helps calculate q-values so that outliers don't sway the q-value too much, but the q-value is adjusted according to the outcomem

# Higher learning rate means placing more emphasis on the new q value, whereas lower learning rate means placing more emphasis on the old q value

# Equation for updating the q-value

# q(new)(s,a) = (1 - alpha) _ q(s,a) + alpha(R(t+1) + ymaxq_(s',a'))

# DQN:
