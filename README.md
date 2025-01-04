# Q_Learning_Robot_navigation_simulation

This repository contains a Python-based simulation of a robot navigating a 6x6 grid using Q-Learning, Monte Carlo Simulation, and Greedy Search algorithms. The robot learns to optimize its pathfinding and decision-making strategies based on varying goals like reward maximization and distance minimization. The project also includes test scripts and visualization tools to demonstrate the robot's behavior.

**Project Structure**

.
├── robot.py                  # Core program implementing the robot and its navigation algorithms
├── test_q_learning.py        # Tests robot navigation using Q-Learning
├── test_monte_carlo.py      # Tests navigation using Monte Carlo and Greedy Search algorithms
├── visualization_q_learning.py          # Provides a visualization of the robot's movement and paths
├── README.md                 # Documentation for the repository
├── grid.jpg                  # Terrain example

**Features**

Navigation Algorithms
Greedy Search:
Prioritizes immediate rewards and minimizes the Manhattan distance to the goal.
Suitable for fast navigation but may not always yield the optimal path.
Monte Carlo Simulation:
Explores random paths and selects the safest route based on cumulative rewards.
Evaluates multiple paths to identify the one with the highest reward.
Q-Learning:
Reinforcement learning algorithm where the robot learns an optimal path through trial and error.
Updates its strategy dynamically based on past experiences and adjusts to maximize long-term rewards.

**How to Run**

Prerequisites
Python 3.7+
NumPy library (pip install numpy)

Running the Programs
Main Program:
Implements all navigation algorithms.
To execute:
python robot.py
Test Scripts:
Q-Learning Test:
python test_q_learning.py
Tracks the robot's learning process and evaluates its ability to navigate efficiently.
Monte Carlo and Greedy Search Test:
python test_monte_carlo.py
Compares the performance of Monte Carlo simulations with Greedy Search.
Visualization Program:
Displays the robot's movements and paths visually.
To execute:
python visualization_q_learning.py

Example Outputs

Q-Learning Test:
Total Reward for Simulation: XX
Path Taken: [(3, 0), (4, 0), ..., (0, 5)]
Monte Carlo and Greedy Search Test:
Monte Carlo Simulation:
Best Reward: XX
Safest Path: [(3, 0), (3, 1), ..., (0, 5)]
Greedy Search:
Greedy Path: [(3, 0), (4, 0), ..., (0, 5)]
Total Reward (Distance): XX

Visualization

The visualization program provides a graphical representation of the robot’s journey through the grid. It highlights:

Monte Carlo Safe Path
Greedy Path
Q-Learning Optimal Path

Customization

Adjust the reward matrix in robot.py to simulate different terrains.
Modify learning parameters (e.g., alpha, gamma, epsilon) for Q-Learning to observe its impact on the robot's behavior.
Change the number of episodes or simulations in the test scripts to experiment with learning efficiency.

Future Improvements

Add support for larger grids or dynamic grid sizes.
Enhance visualization with step-by-step animations.
Integrate other reinforcement learning algorithms like SARSA or DQN.

License

This project is open-source and available under the MIT License.

