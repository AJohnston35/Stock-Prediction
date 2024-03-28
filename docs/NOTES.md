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

# Q-Learning vs. Deep Q-Learning vs. Deep Q-Network

# Q-Learning: does not require knowledge of the transition and reward functions, trains the value function to learn which actions are more valuable in each state and select the optimal action accordingly, and evaluates and updates a policy that differs from the policy used to take action

# Algorithm:

# for each episode do

# initialize S

# foreach step of episode do

# choose A from S using policy derived form Q

# take action A, observe R, S'

# Q(S,A)<- Q(S,A)+alpha[R + y maxa Q(S',a) - Q(S,A)]

# S <- S'

# end

# until S is terminal

# end

# Deep Q-Learning and Deep Q-Network

# Drawback of Q-learning is that it becomes infeasible when dealing with large state spaces, as the size of the Q-table grows exponentially with the number of states and actions

# One approach is Deep Q-Learning:

# The neural network receives the state as an input and outputs the Q-values for all possible actions.

# Instead of mapping a (state,action) pair to a Q-value, the neural network maps input states to (action,Q-value) pairs

# Environment: interacts with an environment with a state, action space, and reward function.

# Replay Memory: uses a replay memory buffer to store past experiences. Each experience is a tuple(state, action, reward, next state) representing a single transition from one state to another. These are stored to sample from later randomly

# Deep Neural Network: uses a deep neural network to estimate the Q-values for each (state,action) pair. The NN takes the state as input and outputs the Q-value for each action. The network is trained to minimize the difference between the predicted and target Q-values.

# Epsilon-Greedy: defines balance between exploration and exploitation. Agent selects a random action with probability epsilon and selects the action with the highest Q-value with probability (1-epsilon)

# Target network: uses a separate target network to estimate target Q-values. The target network is a copy of the main neural network with fixed parameters. The target network is updated periodically to prevent the overestimation of Q-values.

# Training: trains the neural network using the Bellman equation to estimate the optimal Q-values. Loss function is mean squared error between predicted and target values. Target Q-value is calculated using the target network and Bellman equation. Weights are updated using backpropagation and stochastic gradient descent

# Testing: uses the learnged policy to make environmental decisions after training. Agent selects the action with the highest Q-value for a given state

# Putting it all together:

# Initialize two NNs with same weights and a list for storing all experiences. Initialize parameters (batch size, learning rate, etc.)

# The "training loop"

# 1. Agent starts the game and chooses action via exploration/exploitation

# a. In the beginning, actions will be mostly random but will be mostly intentional by the end.

# b. When actions become intentional, they are chosen in the following process: the state is input into the neural network, then the neural networks output a list of q-values, and the action which has the highest q-value is the one which is chosen.

# 2. After taking an action, the agent records the following information: the state is started in, action it took, reward it got, and the following state it was in.

# a. one 'set' of these numbers is known as an experience

# b. the experiences are stored in the Replay Memory, which is a list of all the experiences.

# 3. Every couple of time steps, the agent indexes a set of experiences for the networks to train on.

# a. the network will state the original state as input into the neural network, and it will spit out the q-values for each possible action. But, we only care about the q-values of the action that was actually taken in the experience.

# b. since we have the reward and following state in the experience from taking that action, we have slightly more information on what the Q-value should be. We calculate the optimal Q-value by adding the reward, and the largest Q-value outputted from the network at the following state.

# c. we train the network by comparing the original Q-value for the action the agent took in the experience to the optimal Q-value that we calculated. This is the 'loss' of the network.

# d. the target network is not trained yet, as its parameters remain fixed and update infrequently

# 4. Every dozen or so time steps (the target network's parameters are updated to match the parameters of the policy network)

# Now applying it to the project:

# Stock market environment (currently): subset of the some of the largest stocks trading on the NASDAQ

# Agent:

# Initialize with size of state and action spaces, memory buffer capacity, and inventory management.

# The agent employs a deep neural network model, using Keras, to approximate the Q-values of state-action pairs.

# The model incorporates LSTM layers, enabling the agent to capture and leverage temporal dependencies.

# An attention mechanism is applied to enhance the agent's ability to focus on relevant information.

# Agent adopts epsilon-greedy strategy for action selection.

# Based on the current state and predicted Q-values from the model, the agent either explores the environment or exploits the learned knowledge.

# Experience replay is comprised of tuples of state, action, reward, and done.

# During training, a random batch of experiences is sampled from the buffer, breaking the temporal correlation between consecutive samples.

# Throughout training, the agent iteratively interacts with the environment, updates the Q-network's weights using Q-learning, and refines its decision-making strategies.

# States:

# In the case of the project, the state vector is historical stock data, indicators, and news sentiment for a company.

# MULTI ACTION AND MULTI ASSET SETUP

# States will be tuples of stocks, and each stock contains a tuple of data pertaining that stock for that iteration.

# Data for each stock will be: historical data, indicators, interest rates, news sentiment, and percent returns from the previous day

# Action space:

# Agent will have 3 distinct actions: Buy, sell, and hold

# The agent may only perform one action for each ticker each day

# Agent may always HOLD

# Agent may always BUY, given it has sufficient funds

# Agent may only SELL, when has bought previously
