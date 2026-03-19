import gymnasium as gym
import time

env = gym.make('CartPole-v1', render_mode='human')
observation, info = env.reset()

# The tuned PD weights
Kp = 10.0   # Proportional: Push harder the further it leans
Kd = 1.0    # Derivative: Dampen the push based on how FAST it is falling

for _ in range(500):
    cart_pos, cart_vel, pole_angle, pole_ang_vel = observation
    print(f"Pole Angle: {pole_angle:.3f} radians")
    # The PD formula using Gym's incredibly smooth angular velocity
    pd_output = (Kp * pole_angle) + (Kd * pole_ang_vel)

    # Convert the math into a Gym action
    if pd_output > 0:
        action = 1  # Push Right
    else:
        action = 0  # Push Left

    # Take the step
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        # Let's be honest about WHY it failed this time
        if abs(pole_angle) >= 0.20:
            print("Pole fell! Resetting...")
        else:
            print("Cart drove off the screen! Resetting...")
            
        observation, info = env.reset()
        
    time.sleep(0.02)

env.close()