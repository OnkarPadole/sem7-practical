#python3 -m pip install gymnasium


import gymnasium as gym
import numpy as np

# Create Taxi environment
env = gym.make("Taxi-v3")

# Initialize Q-table
q_table = np.zeros((env.observation_space.n,
                    env.action_space.n))

# Parameters
alpha = 0.1
gamma = 0.9
epsilon = 0.1
episodes = 10000

# Training
for episode in range(episodes):

    state, info = env.reset()
    done = False

    while not done:

        # Choose action
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        # Perform action
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Update Q-table
        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[next_state])
            - q_table[state, action]
        )

        state = next_state

# Testing the trained agent
state, info = env.reset()
done = False

print("Taxi Environment:")

while not done:

    action = np.argmax(q_table[state])

    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    print(env.unwrapped.render())

print("Passenger successfully dropped!")
env.close()
