import pygame
import random
import numpy as np 

pygame.init()
screen_width, screen_height = 400, 400
screen = pygame.display.set_mode((screen_width, screen_height))
clock = pygame.time.Clock()
agent_radius = 5

def make_height_map():
    map = np.random.random((screen_width, screen_height))
    return map

class Agent:
    def __init__(self, pos):
        self.x = pos[0]
        self.y = pos[1]
        self.vel = 5
        self.fitness = None
        self.p_best = pos
        self.p_best_fit = None

    def get_fitness(self):
        
        return self.fitness

    def update_position(self, p_best, p_best_fit):
        self.x += self.vel
        self.y += self.vel
        if self.x > screen_width:
            self.x = 0
        if self.y > screen_height:
            self.y = 0

class Swarm:
    def __init__(self):
        self.agents = [Agent((random.randrange(0, screen_width), random.randrange(0, screen_height))) for i in range(10)]
        self.g_best = None


def update_velocity(agents, base_vel):
    new_velocity = 1

    return new_velocity

def main():
    swarm = Swarm()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        for agent in swarm.agents:
            pygame.draw.circle(screen, (255, 255, 255), (agent.x, agent.y), agent_radius)
            agent.vel = update_velocity(swarm, agent.vel)
            agent.update_position(1, 2)
        
        #swarm.find

        pygame.display.flip()
        clock.tick(60)
        screen.fill((0,0,0))

print(make_height_map())
main()