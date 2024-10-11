import gym
import numpy as np

# Create the Pendulum environment with render_mode for visualization
env = gym.make('Pendulum-v1', render_mode='human')

# Set parameters
num_episodes = 5000
learning_rate = 0.1
discount_factor = 0.99  # Discount factor (gamma)
epsilon = 0.1           # Epsilon for epsilon-greedy action selection
num_bins = 12           # Number of bins for discretization of state space
num_action_bins = 5     # Number of bins for discretization of action space

# Define the bins for each state dimension (Pendulum has 3 dimensions: cos(theta), sin(theta), and angular velocity)
bins = [
    np.linspace(-1.0, 1.0, num_bins),  # cos(theta)
    np.linspace(-1.0, 1.0, num_bins),  # sin(theta)
    np.linspace(-8.0, 8.0, num_bins)   # angular velocity
]

# Discretize the continuous action space into a set of discrete actions
actions = np.linspace(-2.0, 2.0, num_action_bins)

# Initialize the Q-table
q_table = np.zeros([num_bins] * len(bins) + [num_action_bins])

# Discretization function for states
def discretize_state(state):
    """Discretizes the continuous state into discrete bins."""
    state_indices = []
    for i in range(len(state)):
        # Find the index of the bin into which each component falls
        state_indices.append(np.digitize(state[i], bins[i]) - 1)
    return tuple(state_indices)

# Function for epsilon-greedy action selection
def epsilon_greedy_action(state_indices, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(range(num_action_bins))  # Explore: select a random action
    else:
        return np.argmax(q_table[state_indices])  # Exploit: select the action with max Q-value

# SARSA algorithm
for episode in range(num_episodes):
    state = env.reset()
    
    # Extract state if env.reset() returns a tuple
    if isinstance(state, tuple):
        state = state[0]

    # Discretize the initial state
    state_indices = discretize_state(state)

    action_index = epsilon_greedy_action(state_indices, epsilon)
    action = actions[action_index]  # Map the action index to a continuous action

    done = False
    while not done:
        # Take action and observe the next state and reward
        next_state, reward, terminated, truncated, _ = env.step([action])
        done = terminated or truncated

        # If env.step() returns a tuple, take the first element
        if isinstance(next_state, tuple):
            next_state = next_state[0]
        
        # Discretize the next state
        next_state_indices = discretize_state(next_state)

        # Choose next action using epsilon-greedy policy
        next_action_index = epsilon_greedy_action(next_state_indices, epsilon)
        next_action = actions[next_action_index]

        # Calculate the SARSA update
        q_table[state_indices + (action_index,)] += learning_rate * (
            reward + discount_factor * q_table[next_state_indices + (next_action_index,)] - q_table[state_indices + (action_index,)]
        )
        
        # Update state and action
        state_indices = next_state_indices
        action_index = next_action_index
        action = next_action

    # Optional: Decay epsilon to reduce exploration over time
    if epsilon > 0.01:
        epsilon -= 0.0001

print("Training completed!")

# Testing the learned policy with visualization
num_test_episodes = 10
for episode in range(num_test_episodes):
    state = env.reset()
    
    # If env.reset() returns a tuple, take the first element
    if isinstance(state, tuple):
        state = state[0]

    # Discretize the initial state
    state_indices = discretize_state(state)

    done = False
    total_reward = 0
    print(f"Test Episode {episode + 1}")
    
    while not done:
        action_index = np.argmax(q_table[state_indices])
        action = actions[action_index]  # Map the action index to a continuous action
        next_state, reward, terminated, truncated, _ = env.step([action])
        done = terminated or truncated
        
        # If env.step() returns a tuple, take the first element
        if isinstance(next_state, tuple):
            next_state = next_state[0]
        
        # Discretize the next state
        state_indices = discretize_state(next_state)
        total_reward += reward

        # Render the environment at each step during testing
        env.render()
    
    print(f"Total reward: {total_reward}")

env.close()
