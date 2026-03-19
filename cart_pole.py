import gymnasium as gym
import time

# 1. Initialize the environment
env = gym.make('CartPole-v1', render_mode='human')

# 2. Reset the environment to its starting state
observation, info = env.reset()

# Run the simulation for 100 frames
for _ in range(100):
    # 3. Choose a random action (0 = push left, 1 = push right)
    action = env.action_space.sample() 
    
    # 4. Apply the action to the environment
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Unpack the new state from the observation array
    cart_pos, cart_vel, pole_angle, pole_ang_vel = observation
    print(f"Pole Angle: {pole_angle:.3f} radians")
    
    # If the pole falls too far, the episode terminates. We must reset it.
    if terminated or truncated:
        print("Pole fell! Resetting...")
        observation, info = env.reset()
        
    time.sleep(0.5) # Slows down the loop so we can watch it

# Close the window when done
env.close()