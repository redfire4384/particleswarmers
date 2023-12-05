import pygame
import random
import numpy as np

pygame.init()
screen_width, screen_height = 400, 400
tile_size = 10 
noise_map_size = (screen_width // tile_size, screen_height // tile_size)
screen = pygame.display.set_mode((screen_width, screen_height))
clock = pygame.time.Clock()
agent_radius = 5

def make_height_map(smoothing_size=2):

    raw_map = np.random.random(noise_map_size)
    padded_map = np.pad(raw_map, smoothing_size//2, mode='edge')
    smoothed_map = np.zeros_like(raw_map)

    # Convolution
    for x in range(raw_map.shape[0]):
        for y in range(raw_map.shape[1]):
            region = padded_map[x:x+smoothing_size, y:y+smoothing_size]
            smoothed_map[x, y] = np.mean(region)

    # Normalize
    min_val = np.min(smoothed_map)
    max_val = np.max(smoothed_map)
    normalized_map = (smoothed_map - min_val) / (max_val - min_val)

    return normalized_map

def value_to_color(value):
    return (int(value * 255), 0, int((1 - value) * 255))

def draw_noise_map(noise_map):
    for x in range(noise_map_size[0]):
        for y in range(noise_map_size[1]):
            color = value_to_color(noise_map[x, y])
            pygame.draw.rect(screen, color, (x * tile_size, y * tile_size, tile_size, tile_size))

class Agent:
    def __init__(self, pos, noise_map):
        self.x = pos[0]
        self.y = pos[1]
        self.vel = 5
        self.fitness = None
        self.p_best = pos
        self.p_best_fit = None
        self.noise_map = noise_map

    def update_fitness(self):
        tile_x = int(self.x / tile_size)
        tile_y = int(self.y / tile_size)
        self.fitness = self.noise_map[tile_x, tile_y]

    def update_position(self, p_best, p_best_fit):
        pass

class Swarm:
    def __init__(self, agentnum, noise_map):
        self.agents = [Agent((random.randrange(0, screen_width), random.randrange(0, screen_height)), noise_map) for i in range(agentnum)]
        self.g_best_pos = None
        self.g_best_fit = None

    def update_global_best(self):
        for agent in self.agents:
            if self.g_best_fit is None or agent.fitness > self.g_best_fit:
                self.g_best_fit = agent.fitness
                self.g_best_pos = (agent.x, agent.y)

def update_velocity(agents, base_vel):
    new_velocity = 1
    return new_velocity

def main():
    noise_map = make_height_map()
    swarm = Swarm(10, noise_map)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        draw_noise_map(noise_map)

        swarm.update_global_best()

        for agent in swarm.agents:
            pygame.draw.circle(screen, (255, 255, 255), (agent.x, agent.y), agent_radius)
            agent.vel = update_velocity(swarm, agent.vel)
            agent.update_position(1, 2)
            agent.update_fitness()

        pygame.display.flip()
        clock.tick(60)
        screen.fill((0,0,0))

main()