import gym
import numpy as np

# Create the Taxi environment with render_mode for visualization
env = gym.make('Taxi-v3', render_mode='human')

# Set parameters
num_episodes = 5000
learning_rate = 0.1
discount_factor = 0.99  # Discount factor (gamma)
epsilon = 0.1           # Epsilon for epsilon-greedy action selection

# Initialize the Q-table
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Function for epsilon-greedy action selection
def epsilon_greedy_action(state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Explore: select a random action
    else:
        return np.argmax(q_table[state])  # Exploit: select the action with max Q-value

# Q-Learning algorithm
for episode in range(num_episodes):
    state = env.reset()
    
    # Extract state if env.reset() returns a tuple
    if isinstance(state, tuple):
        state = state[0]

    done = False
    while not done:
        # Choose action using epsilon-greedy policy
        action = epsilon_greedy_action(state, epsilon)

        # Take action and observe the next state and reward
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Q-Learning update
        best_next_action = np.argmax(q_table[next_state])
        td_target = reward + discount_factor * q_table[next_state, best_next_action]
        td_error = td_target - q_table[state, action]
        q_table[state, action] += learning_rate * td_error

        # Update state
        state = next_state

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

    done = False
    total_reward = 0
    print(f"Test Episode {episode + 1}")
    
    while not done:
        action = np.argmax(q_table[state])
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Update state
        state = next_state
        total_reward += reward

        # Render the environment at each step during testing
        env.render()
    
    print(f"Total reward: {total_reward}")

env.close()
