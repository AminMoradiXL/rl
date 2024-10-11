import gym
import numpy as np

# Create the MountainCar environment with render_mode for visualization
env = gym.make('MountainCar-v0', render_mode='human')

# Set parameters
num_episodes = 5000
learning_rate = 0.1
discount_factor = 0.99  # Discount factor (gamma)
epsilon = 0.1           # Epsilon for epsilon-greedy action selection
num_bins = 20           # Number of bins for discretization

# Define the bins for each state dimension (MountainCar has 2 dimensions: position and velocity)
bins = [
    np.linspace(-1.2, 0.6, num_bins),  # Position of the car
    np.linspace(-0.07, 0.07, num_bins) # Velocity of the car
]

# Initialize the Q-table
q_table = np.zeros([num_bins] * len(bins) + [env.action_space.n])

# Discretization function
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
        return env.action_space.sample()  # Explore: select a random action
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

    action = epsilon_greedy_action(state_indices, epsilon)
    
    done = False
    while not done:
        # Take action and observe the next state and reward
        next_state, reward, terminated, truncated, _ = env.step(action)
       
