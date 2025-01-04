'''This script tests the robot's navigation in a 6x6 grid using Q-Learning. 
It runs multiple simulations where the robot learns an optimal path through 
repeated episodes, then follows the learned path to reach the goal, 
tracking the total reward and path taken in each simulation.'''


import numpy as np
from robot import Robot

def main():
    # Create an instance of the Robot
    robot = Robot()
    
    # Number of simulations to run
    n_simulations = 10


    for simulation in range(n_simulations):
        print(f"Simulation {simulation + 1}/{n_simulations}")
        
        # Reset the robot to the starting position
        robot.reset()

        total_reward = 0  # Initialize total reward for this simulation
        current_path = [(robot.get_x(), robot.get_y())]  # Start path with initial position

        # Perform Q-learning for a number of episodes
        episodes = 100  # Adjust the number of episodes for more learning
        robot.q_learning(episodes)

        # After learning, reset the robot again to start following the learned path
        robot.reset()

        # Now track the path taken after learning
        while not robot.has_reached_goal():
            # Take action according to the learned Q-values (exploit)
            action = robot.get_next_state_eg()  # Choose the best action based on learned Q-values
            robot.take_action(action)  # Take the chosen action

            # Collect the reward
            reward = robot.R_matrix[robot.get_y(), robot.get_x()]
            total_reward += reward  # Update the total reward
            current_path.append((robot.get_x(), robot.get_y()))  # Record the current position

        # Print the total reward and path taken for this simulation
        print(f"Total Reward for Simulation {simulation + 1}: {total_reward}")
        print(f"Path Taken: {current_path}")
        print("-" * 40)

if __name__ == "__main__":
    main()