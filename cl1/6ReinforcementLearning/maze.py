
import numpy as np

# Maze: 0 = Path, 1 = Wall, 2 = Goal
maze = np.array([
    [0, 0, 0, 1],
    [1, 1, 0, 1],
    [0, 0, 0, 0],
    [0, 1, 1, 2]
])

Q = np.zeros((16, 4))

actions = [(-1,0), (1,0), (0,-1), (0,1)]

alpha = 0.1
gamma = 0.9
epsilon = 0.2

# Training
for episode in range(1000):
    state = (0, 0)

    for step in range(100):
        s = state[0]*4 + state[1]

        if np.random.rand() < epsilon:
            a = np.random.randint(4)
        else:
            a = np.argmax(Q[s])

        r, c = state
        nr, nc = r+actions[a][0], c+actions[a][1]

        if not (0 <= nr < 4 and 0 <= nc < 4) or maze[nr,nc] == 1:
            nr, nc = r, c
            reward = -5
        elif maze[nr,nc] == 2:
            reward = 100
        else:
            reward = -1

        ns = nr*4 + nc

        Q[s,a] += alpha * (
            reward + gamma*np.max(Q[ns]) - Q[s,a]
        )

        state = (nr,nc)

        if maze[nr,nc] == 2:
            break

# Display Q-table
print("Q-Table:")
print(np.round(Q, 2))

print("Training Completed!")
