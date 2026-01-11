import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

# Config
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
BLOCK_SIZE = 20 

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # Randomness
        self.gamma = 0.9 # Discount rate
        self.memory = deque(maxlen=MAX_MEMORY) 
        
        # 11 + 92
        self.model = Linear_QNet(92, 256, 3) 
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        
        # same as before
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state_11 = [
            # danger forward
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # danger Left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # move direction
            dir_l, dir_r, dir_u, dir_d,
            
            # food Location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y   # food down
        ]
        
        # *** Add nearby 9 * 9 cells ***
        rad = 4
        grid_size = rad * 2 + 1 
        grid = np.ones((grid_size, grid_size), dtype=int) 

        # scan nearby cells, radius = rad
        for i, y_offset in enumerate(range(-rad, rad + 1)):
            for j, x_offset in enumerate(range(-rad, rad + 1)):
                check_pt = Point(head.x + x_offset * BLOCK_SIZE, head.y + y_offset * BLOCK_SIZE)
                
                if check_pt.x < 0 or check_pt.x > game.w - BLOCK_SIZE or \
                   check_pt.y < 0 or check_pt.y > game.h - BLOCK_SIZE:
                    grid[i][j] = 1
                elif check_pt in game.snake:
                    grid[i][j] = 1
                elif check_pt == game.food:
                    grid[i][j] = 2
                else:
                    grid[i][j] = 0
        
        # rotate matrix
        k = 0
        if game.direction == Direction.UP:
            k = 0 
        elif game.direction == Direction.RIGHT:
            k = 1 
        elif game.direction == Direction.DOWN:
            k = 2 
        elif game.direction == Direction.LEFT:
            k = 3 

        grid_rotated = np.rot90(grid, k=k)
        state_grid = grid_rotated.flatten().tolist()

        # merge
        state_11 = [int(x) for x in state_11]
        final_state = state_11 + state_grid
        
        return np.array(final_state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    
    agent = Agent()
    game = SnakeGameAI()
    
    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old) # get move
        reward, done, score = game.play_step(final_move) # move and get the new state
        state_new = agent.get_state(game)
        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print(f'Game {agent.n_games} | Score {score} | Record {record}')

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

# run
if __name__ == '__main__':
    train()