import gym
import numpy as np

# Create the Acrobot environment with render_mode for visualization
env = gym.make('Acrobot-v1', render_mode='human')

# Set parameters
num_episodes = 5000
learning_rate = 0.1
discount_factor = 0.99  # Discount factor (gamma)
epsilon = 0.1           # Epsilon for epsilon-greedy action selection
num_bins = 10           # Number of bins for discretization

# Define the bins for each state dimension (Acrobot has 6 dimensions)
bins = [
    np.linspace(-1.0, 1.0, num_bins),  # cos(theta1)
    np.linspace(-1.0, 1.0, num_bins),  # sin(theta1)
    np.linspace(-1.0, 1.0, num_bins),  # cos(theta2)
    np.linspace(-1.0, 1.0, num_bins),  # sin(theta2)
    np.linspace(-4.0, 4.0, num_bins),  # angular velocity of link 1
    np.linspace(-9.0, 9.0, num_bins)   # angular velocity of link 2
]

# Initialize the Q-table
q_table = np.zeros([num_bins] * len(bins) + [env.action_space.n])

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
        return env.action_space.sample()  # Explore: select a random action
    else:
        return np.argmax(q_table[state_indices])  # Exploit: select the action with max Q-value

# Q-Learning algorithm
for episode in range(num_episodes):
    state = env.reset()
    
    # Extract state if env.reset() returns a tuple
    if isinstance(state, tuple):
        state = state[0]

    # Discretize the initial state
    state_indices = discretize_state(state)

    done = False
    while not done:
        # Choose action using epsilon-greedy policy
        action = epsilon_greedy_action(state_indices, epsilon)

        # Take action and observe the next state and reward
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # If env.step() returns a tuple, take the first element
        if isinstance(next_state, tuple):
            next_state = next_state[0]
        
        # Discretize the next state
        next_state_indices = discretize_state(next_state)

        # Q-Learning update
        best_next_action = np.argmax(q_table[next_state_indices])
        td_target = reward + discount_factor * q_table[next_state_indices + (best_next_action,)]
        td_error = td_target - q_table[state_indices + (action,)]
        q_table[state_indices + (action,)] += learning_rate * td_error

        # Update state
        state_indices = next_state_indices

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
        action = np.argmax(q_table[state_indices])
        next_state, reward, terminated, truncated, _ = env.step(action)
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
