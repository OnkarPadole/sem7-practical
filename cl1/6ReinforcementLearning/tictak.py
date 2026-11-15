
import numpy as np
import random

# a. Setting up the environment
class TicTacToe:

    def __init__(self):
        self.board = [0] * 9

    # b. Defining the game
    def reset(self):
        self.board = [0] * 9
        return tuple(self.board)

    def available_moves(self):
        return [i for i in range(9) if self.board[i] == 0]

    def check_winner(self):
        wins = [
            (0,1,2), (3,4,5), (6,7,8),
            (0,3,6), (1,4,7), (2,5,8),
            (0,4,8), (2,4,6)
        ]

        for a, b, c in wins:
            if self.board[a] == self.board[b] == self.board[c] != 0:
                return self.board[a]

        return 0

    def display(self):
        symbols = {0: "-", 1: "X", -1: "O"}

        for i in range(0, 9, 3):
            print(" ".join(symbols[x] for x in self.board[i:i+3]))
        print()

# c. Building the reinforcement learning model
class QLearningAgent:

    def __init__(self):
        self.q_table = {}
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.2

    def get_action(self, state, moves):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(9)

        if random.random() < self.epsilon:
            return random.choice(moves)

        return max(moves, key=lambda x: self.q_table[state][x])

    def update(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(9)

        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(9)

        old_value = self.q_table[state][action]
        next_value = np.max(self.q_table[next_state])

        self.q_table[state][action] = old_value + self.alpha * (
            reward + self.gamma * next_value - old_value
        )

# d. Training the model
game = TicTacToe()
agent = QLearningAgent()

episodes = 50000

for episode in range(episodes):

    state = game.reset()
    done = False
    moves = game.available_moves()

    while not done:

        # Agent plays as X
        action = agent.get_action(state, moves)
        game.board[action] = 1

        winner = game.check_winner()

        if winner == 1:
            agent.update(state, action, 1, tuple(game.board))
            break

        moves = game.available_moves()

        if not moves:
            agent.update(state, action, 0, tuple(game.board))
            break

        # Random opponent plays as O
        opponent_action = random.choice(moves)
        game.board[opponent_action] = -1

        winner = game.check_winner()

        if winner == -1:
            agent.update(state, action, -1, tuple(game.board))
            break

        next_state = tuple(game.board)
        next_moves = game.available_moves()

        if not next_moves:
            agent.update(state, action, 0, next_state)
            break

        agent.update(state, action, 0, next_state)

        state = next_state
        moves = next_moves

# e. Testing the trained model
agent.epsilon = 0

print("Tic-Tac-Toe Game")
print("You are O and AI is X")
print("Enter position from 0 to 8")

game.reset()

while True:

    # AI turn
    state = tuple(game.board)
    moves = game.available_moves()

    if not moves:
        print("Game Draw!")
        break

    action = agent.get_action(state, moves)
    game.board[action] = 1

    print("\nAI Move:")
    game.display()

    if game.check_winner() == 1:
        print("AI Wins!")
        break

    if not game.available_moves():
        print("Game Draw!")
        break

    # Human turn
    while True:
        try:
            pos = int(input("Enter your move (0-8): "))

            if pos in game.available_moves():
                game.board[pos] = -1
                break
            else:
                print("Invalid position! Try again.")

        except ValueError:
            print("Please enter a number from 0 to 8.")

    game.display()

    if game.check_winner() == -1:
        print("You Win!")
        break

    if not game.available_moves():
        print("Game Draw!")
        break
