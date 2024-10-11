import supersuit as ss
from pettingzoo.mpe import simple_tag_v3
from stable_baselines3 import PPO
from pettingzoo.utils.conversions import aec_to_parallel

# Step 1: Create the simple_tag environment with render_mode for testing
env = simple_tag_v3.env(max_cycles=25, render_mode='human')  # Set render_mode here for visualization

# Step 2: Convert the AEC environment to a parallel environment
parallel_env = aec_to_parallel(env)

# Step 3: Wrap the parallel environment with SuperSuit for compatibility with stable-baselines3
# Pad observations to make sure all agents have the same observation size
parallel_env = ss.pad_observations_v0(parallel_env)

# Convert to a vectorized environment
env = ss.pettingzoo_env_to_vec_env_v1(parallel_env)
env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')

# Step 4: Create the PPO model using the wrapped environment
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=256, batch_size=64, n_epochs=10)

# Step 5: Train the model
model.learn(total_timesteps=1000000)  # Adjust total_timesteps as needed

# Step 6: Save the trained model
model.save("mappo_simple_tag")

# Step 7: Testing the trained model
# Reload the model (optional)
model = PPO.load("mappo_simple_tag")

# Recreate the environment for testing with render_mode='human'
test_env = simple_tag_v3.env(max_cycles=25, render_mode='human')
parallel_test_env = aec_to_parallel(test_env)

# Wrap the test environment with SuperSuit
parallel_test_env = ss.pad_observations_v0(parallel_test_env)
test_env = ss.pettingzoo_env_to_vec_env_v1(parallel_test_env)
test_env = ss.concat_vec_envs_v1(test_env, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')

# Step 8: Run a few episodes to visualize the trained agents
obs = test_env.reset()
for _ in range(1000):  # Adjust the number of steps as needed for longer visualization
    action, _ = model.predict(obs)
    obs, rewards, dones, info = test_env.step(action)
    # No need to call test_env.render() as render_mode='human' will automatically display it

# Close the environment after testing
test_env.close()
