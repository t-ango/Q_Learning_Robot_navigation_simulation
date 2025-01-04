'''
This Python program simulates a robot navigating a 6x6 grid using Greedy Search, 
Monte Carlo simulation, and Q-Learning algorithms. The Greedy Search prioritizes 
immediate rewards and minimizes distance to the goal, while the Monte Carlo 
simulation explores random paths to find the safest route. Q-Learning enables 
the robot to learn an optimal path through trial and error, updating its strategy 
based on past experiences
'''

import numpy as np
import random
import heapq

class Robot:

    def __init__(self):
        # Defining R- and Q-matrices:
        self.grid_size = 6

        #Reward matrix:
        #Initial values: safe = 1, mountains = -1, water = -2, finish = 10, start = 0
        #Initial values are increased towards the bottom of the grid to encourage the robot
        #to move towards the goal
        
        self.R_matrix = np.array([
            [-2,  -1,  -1,  0,  1, -2],  # Row A
            [-2, -2, 2, -2,  2, 2],  # Row B
            [-1,  3, 3, -2,  3, -1],  # Row C
            [-1,  3,  2, 2,  2, 1],  # Row D
            [0, 4,  -1, 3,  -2, 1],  # Row E
            [10,  -1,  -2,  -2,  -2,  -2]   # Row F
        ])

        #Q-matrix initizlized with 0:
        self.Q_matrix = np.zeros((self.grid_size * self.grid_size, 4))

        #Start possition:
        self.x = 3
        self.y = 0

        #Finish possition:
        self.finish_x = 0
        self.finish_y = 5

        # Learning parameters
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.1  # Exploration rate


    #ACTION SECTION

    def get_x(self):
        # Return the current column of the robot, should be in the range 0-5.
        return self.x

    def get_y(self):
        # Return the current row of the robot, should be in the range 0-5.
        return self.y
    
    def reset(self):
            #put the robot back to the starting point
            self.x, self.y = 3,0

    def has_reached_goal(self):
        # Return 'True' if the robot is in the goal state.
        return self.x == self.finish_x and self.y == self.finish_y
        
    def reset_random(self):
        #Place the robot in a new random state (x, y) within the grid.
        #Excludes the goal state to avoid starting directly at the goal.

        # Randomly select new (x, y) coordinates
        while True:
            self.x = random.randint(0, self.grid_size - 1)
            self.y = random.randint(0, self.grid_size - 1)
            
            # Avoid placing the robot at the goal state (F1 in this case)
            if not (self.x == self.finish_x and self.y == self.finish_y):
                break  

        return self.x, self.y 

    def manhattan_distance (self, node1: tuple[int, int], node2: tuple[int, int]) -> int:
        #Calculate the Manhattan distance between two points on the grid.
        x1, y1 = node1
        x2, y2 = node2
        return abs(x1 - x2) + abs(y1 - y2)
    
    def path_backtrack(self, end_node: tuple[int, int], came_from: dict) -> list:
        #Reconstruct the path from the start node to the end node.
        path = [end_node]
        current = end_node

        while came_from[current] is not None:
            current = came_from[current]
            path.append(current)

        path.reverse()  # Reverse to get the path from start to end
        return path

    def greedy_search(self, graph: dict, start_node: tuple[int, int], end_node: tuple[int, int]) -> tuple[float, list]:
        '''
        Perform a greedy search from start_node to end_node in the given graph.
        Returns total distance of the path and list representing the path from start to goal.
        '''
        came_from = {start_node: None}
        visited = set()
        queue = [(0, start_node)]  # Start with zero cost

        while queue:
            current_score, current = heapq.heappop(queue)
            
            if current == end_node:
                break  # Goal reached

            visited.add(current)

            # Explore neighbors (up, down, left, right)
            x, y = current
            neighbors = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]
            
            for neighbor in neighbors:
                if neighbor not in graph or neighbor in visited:
                    continue  # Skip if the neighbor is out of bounds or already visited
                
                # Calculate the reward for moving to the neighbor
                reward = self.R_matrix[neighbor[1]][neighbor[0]]  # R_matrix is accessed as [y][x]

                # Total score for the neighbor
                new_score = current_score - reward + self.manhattan_distance(neighbor, end_node)  # Minimizing distance, maximizing reward
                
                if neighbor not in came_from:
                    came_from[neighbor] = current
                    heapq.heappush(queue, (new_score, neighbor))

        # Reconstruct the path
        path = self.path_backtrack(end_node, came_from)
        
        # Calculate total path reward
        path_reward = sum(self.R_matrix[y][x] for (x, y) in path)
        
        return path_reward, path

    def greedy_path(self):
        '''
        Finds the greedy path from the robot's current position to the goal (F1).
        Uses Manhattan distance and matrix value as the heuristic to guide the greedy search.
        '''
        start = (self.x, self.y)
        goal = (self.finish_x, self.finish_y)
        
        # Create the graph representation of the grid with rewards (weights)
        graph = {}
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                node = (x, y)
                graph[node] = {}

                # Check for valid neighbors and assign the corresponding reward
                if y > 0:  # Up
                    graph[node][(x, y - 1)] = self.R_matrix[y - 1, x]
                if y < self.grid_size - 1:  # Down
                    graph[node][(x, y + 1)] = self.R_matrix[y + 1, x]
                if x > 0:  # Left
                    graph[node][(x - 1, y)] = self.R_matrix[y, x - 1]
                if x < self.grid_size - 1:  # Right
                    graph[node][(x + 1, y)] = self.R_matrix[y, x + 1]

        # Perform the greedy search to find the path
        path_reward, path = self.greedy_search(graph, start, goal)
        
        return path, path_reward
    
    #MONTE_CARLO BLOCK

    def monte_carlo_exploration_reward (self, n_simulations: int) -> tuple[float, list]:
        """
        Perform Monte Carlo simulation to find the safest path from start to goal (A4 -> F1).
        The method will calculate the total accumulated reward for each simulation based on the reward matrix.
        The path with the highest reward will be considered the safest route.
        
        Parameters:
        -----------
        n_simulations: int
            The number of Monte Carlo simulations to run.
        
        Returns:
        --------
        best_reward: float
            The highest accumulated reward during the simulations.
        best_path: list
            The safest path (highest reward).
        """
        best_reward = float('-inf')  # Initialize with a very low value
        best_path = []  # Track the path with the highest reward

        start = (3, 0)  # A4 start position
        goal = (0, 5)   # F1 goal position

        for simulation in range(n_simulations):
            # Reset the robot to the start position (A4)
            self.reset()

            current_path = [start]  # Record the path taken
            total_reward = 0  # Initialize the total reward for this simulation
            visited = set()  # Track visited positions
            visited.add(start)  # Add start as visited

            # Perform transitions until the robot reaches the goal (F1)
            while not self.has_reached_goal():
                current_x, current_y = self.get_x(), self.get_y()

                # Retrieve possible moves based on the current position
                possible_moves = [
                    (current_x, current_y - 1),  # Up
                    (current_x, current_y + 1),  # Down
                    (current_x - 1, current_y),  # Left
                    (current_x + 1, current_y)   # Right
                ]

                # Filter moves to stay within the grid and avoid visited cells
                valid_moves = [
                    (nx, ny) for (nx, ny) in possible_moves
                    if (0 <= nx < self.grid_size) and (0 <= ny < self.grid_size) and ((nx, ny) not in visited)
                ]

                if not valid_moves:  # If no valid moves left
                    print(f"Simulation {simulation + 1}: No valid moves left, breaking the simulation.")
                    break  # Exit the loop

                # Choose a random valid move
                next_x, next_y = random.choice(valid_moves)

                # Update the robot's position and total reward
                self.x, self.y = next_x, next_y
                total_reward += self.R_matrix[next_y][next_x]  # Add the reward for the new state
                current_path.append((next_x, next_y))
                visited.add((next_x, next_y))  # Mark this position as visited

            # Print progress of simulation
            print(f"Simulation {simulation + 1}/{n_simulations} - Reward: {total_reward}, Path: {current_path}")

            # Check if this simulation had the best reward
            if total_reward > best_reward:
                best_reward = total_reward
                best_path = current_path

        return best_reward, best_path
    
    
    #Q-LEARNING BLOCK

    def state_index(self, x, y):
        # Helper function to map (x, y) position to a state index
        return y * self.grid_size + x

    def get_next_state_eg(self):
        '''
        Use epsilon-greedy policy to choose the next action.
        With probability epsilon, choose a random action (explore).
        With probability 1-epsilon, choose the action with the highest Q-value (exploit).
        '''
        state = self.state_index(self.x, self.y)
        
        if random.uniform(0, 1) < self.epsilon:
            # Explore: choose a random action
            action = random.choice([0, 1, 2, 3])  # 0: up, 1: down, 2: left, 3: right
        else:
            # Exploit: choose the best action with the highest Q-value
            action = np.argmax(self.Q_matrix[state])
        
        return action
    
    def take_action(self, action):
        """
        Update the robot's position based on the action.
        0: up, 1: down, 2: left, 3: right.
        Ensure the robot does not move out of bounds.
        """
        if action == 0 and self.y > 0:  # Up
            self.y -= 1
        elif action == 1 and self.y < self.grid_size - 1:  # Down
            self.y += 1
        elif action == 2 and self.x > 0:  # Left
            self.x -= 1
        elif action == 3 and self.x < self.grid_size - 1:  # Right
            self.x += 1
    

    def one_step_q_learning(self, visited_states):
        """
        Perform one step of Q-learning with a mechanism to penalize revisiting states.
        """
        # Current state
        state = self.state_index(self.x, self.y)
        
        # Choose an action using epsilon-greedy
        action = self.get_next_state_eg()
        
        # Take the action and move to the next state
        self.take_action(action)
        
        # Next state
        next_state = self.state_index(self.x, self.y)

        #get reward 
        reward = self.R_matrix[self.y, self.x]  # Reward from R matrix

        # Check if the state has been visited and apply a penalty if so
        if (self.x, self.y) in visited_states:
            reward = -5  # Penalize revisiting
            
        # Add the current state to the visited states
        visited_states.add((self.x, self.y))
        
        # Q-learning update rule
        best_future_q = np.max(self.Q_matrix[next_state])
        self.Q_matrix[state, action] += self.alpha * (reward + self.gamma * best_future_q - self.Q_matrix[state, action])
    

    def q_learning(self, episodes=10):
        """
        Perform Q-learning over a number of episodes, with memory of visited states.
        """
        for episode in range(episodes):
            # Reset the robot to the start position (A4)
            self.reset()

            # Set to track visited states
            visited_states = set()
            
            while not self.has_reached_goal():
                # Perform one step of Q-learning, passing the visited states
                self.one_step_q_learning(visited_states)