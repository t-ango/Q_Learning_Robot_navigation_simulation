'''This script tests the robot's navigation using both Monte Carlo simulation 
and Greedy Search. It first runs multiple Monte Carlo simulations to find 
the safest path with the highest reward, then performs a greedy search to identify 
the path that maximizes rewards while minimizing distance to the goal.'''


from robot import Robot

if __name__ == "__main__":

    # Initialize the robot
    robot = Robot()

    # Number of simulations to run
    n_simulations = 100


    # Perform the Monte Carlo simulation and find the safest path
    best_reward, best_path = robot.monte_carlo_exploration_reward(n_simulations)

    # Print the best path and its reward
    print(f"\nBest Reward: {best_reward}")
    print(f"Best Path: {best_path}")

    robot.reset()  # A4 is (3, 0)

    # Perform the greedy path search
    greedy_path, greedy_distance = robot.greedy_path()

    # Print the results of the greedy path
    print("\nGreedy Path Results:")
    print(f"Greedy Path: {greedy_path}")
    print(f"Total Reward (Distance): {greedy_distance}")