import gym
import numpy as np
import matplotlib.pyplot as plt

# Define parameters
n_steps = 3  # Number of steps for multi-step Q-learning
gamma = 0.99  # Discount factor
alpha = 0.1  # Learning rate
epsilon = 0.1  # Epsilon for epsilon-greedy policy
num_episodes = 500
max_steps_per_episode = 500
render_frequency = 50  # Render every 50 episodes for visualization

# Create the environment with render_mode='human'
env = gym.make('Acrobot-v1', render_mode='human')
state_space_size = env.observation_space.shape[0]
action_space_size = env.action_space.n

# Define the number of bins for each state variable
bins = 10
q_table_shape = (bins,) * state_space_size + (action_space_size,)
q_table = np.random.uniform(low=-1, high=1, size=q_table_shape)

# Discretize the state space
def discretize_state(state):
    """Discretize the continuous state into discrete bins."""
    state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
    # Replace inf bounds with reasonable limits for Acrobot
    state_bounds = [(-1, 1) if bound[0] == -np.inf else bound for bound in state_bounds]
    state_bounds = [(bound[0], 1) if bound[1] == np.inf else bound for bound in state_bounds]
    
    ratios = [(state[i] - state_bounds[i][0]) / (state_bounds[i][1] - state_bounds[i][0]) for i in range(len(state))]
    new_state = [int(round((bins - 1) * ratio)) for ratio in ratios]
    new_state = [min(bins - 1, max(0, s)) for s in new_state]
    return tuple(new_state)

# Epsilon-greedy action selection
def epsilon_greedy_action(state, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(0, action_space_size)
    return np.argmax(q_table[state])

# Store rewards for visualization
reward_history = []

# Training loop
for episode in range(num_episodes):
    state_info = env.reset()
    if isinstance(state_info, tuple):
        state = state_info[0]
    elif isinstance(state_info, dict):
        state = state_info.get('observation', state_info)
    else:
        state = state_info
    state = discretize_state(state)
    done = False
    t = 0

    states, actions, rewards = [], [], []
    total_reward = 0

    while not done and t < max_steps_per_episode:
        # Render the environment for visualization at specified intervals
        if (episode + 1) % render_frequency == 0:
            env.render()

        # Select an action
        action = epsilon_greedy_action(state, epsilon)

        # Execute action
        result = env.step(action)
        if len(result) == 4:
            next_state_info, reward, done, _ = result
        else:
            next_state_info, reward, done, truncated, _ = result
            done = done or truncated

        # Ensure the state is properly extracted in case it's a dictionary
        if isinstance(next_state_info, tuple):
            next_state = next_state_info[0]
        elif isinstance(next_state_info, dict):
            next_state = next_state_info.get('observation', next_state_info)
        else:
            next_state = next_state_info
        next_state = discretize_state(next_state)

        # Store transition
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        total_reward += reward

        # If we have accumulated enough steps, update Q-values
        if len(states) >= n_steps or done:
            # Calculate n-step return
            n_step_return = sum([gamma ** i * rewards[i] for i in range(len(rewards))])
            if not done:
                n_step_return += gamma ** n_steps * np.max(q_table[next_state])

            # Update the Q-value
            q_table[states[0]][actions[0]] += alpha * (n_step_return - q_table[states[0]][actions[0]])

            # Remove the first state, action, and reward to maintain the n-step window
            states.pop(0)
            actions.pop(0)
            rewards.pop(0)

        state = next_state
        t += 1

    # Store total reward for this episode for visualization
    reward_history.append(total_reward)

    # Decay epsilon for exploration-exploitation trade-off
    epsilon = max(0.01, epsilon * 0.99)

    # Print the progress
    if (episode + 1) % 50 == 0:
        print(f'Episode: {episode + 1}, Total Reward: {total_reward}')

# Close the environment
env.close()

# Plot the reward history to visualize the learning progress
plt.plot(reward_history)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward per Episode over Time (Acrobot-v1)')
plt.show()
