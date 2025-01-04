'''This script visualizes the robot's navigation using Q-Learning in a grid 
environment with Pygame. The robot learns the optimal path to the goal, avoiding 
revisiting the same spots, while the graphical interface displays the robot's 
movements in real-time, updating as the learning progresses.
OBS: code adjusted: visited_states added for tracking the visited nodes'''


import sys
import time
import pygame
from pygame.locals import *
from robot import Robot

# Add colors as needed.
GREEN_COLOR = pygame.Color(0, 255, 0)
BLACK_COLOR = pygame.Color(0, 0, 0)
WHITE_COLOR = pygame.Color(255, 255, 255)

if __name__ == "__main__":
    pygame.init()
    fps_clock = pygame.time.Clock()

    play_surface = pygame.display.set_mode((500, 500))
    pygame.display.set_caption('Karaktersatt Oppgave 1 DTE2602')
    simulator_speed = 3 # Adjust this value to change the speed of the visualiztion. Bigger number = more faster...

    bg_image = pygame.image.load("grid.jpg") # Loads the simplified grid image.
    #bg_image = pygame.image.load("map.jpg") # Uncomment this to load the terrain map image.

    robot = Robot() # Create a new robot.
    robot.reset() #changed to reset to start from A4

    visited_states = set() #added to restrict visiting the same spots

    # Pygame boilerplate code.
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                running = False
                break
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.event.post(pygame.event.Event(QUIT))
                    running = False
                    break

        play_surface.fill(WHITE_COLOR) # Fill the screen with white.
        play_surface.blit(bg_image, (0, 0)) # Render the background image.

        # Render the robot over the image.
        pygame.draw.rect(play_surface, BLACK_COLOR, Rect(robot.get_x() * 70 + 69, robot.get_y() * 70 + 69, 22, 22)) # A black outline.
        pygame.draw.rect(play_surface, GREEN_COLOR, Rect(robot.get_x() * 70 + 70, robot.get_y() * 70 + 70, 20, 20)) # The robot is rendered in green, you may change this if you want.

        # Calls related to Q-learning.
        if robot.has_reached_goal():
            robot.reset() #changed to reset to start from A4
            visited_states = set() #added for restricting the visited spots
        else:
            robot.one_step_q_learning(visited_states)

        # Refresh the screen.
        pygame.display.flip()
        fps_clock.tick(simulator_speed)
