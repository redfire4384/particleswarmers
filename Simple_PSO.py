import pygame
import random
import numpy as np

pygame.init()

c1, c2 = 2, 2
w = 0.0001

bird = pygame.image.load("bird.png")

screen_width, screen_height = 800, 800
tile_size = 20
noise_map_size = (screen_width // tile_size, screen_height // tile_size)
screen = pygame.display.set_mode((screen_width, screen_height))
clock = pygame.time.Clock()
agent_radius = tile_size // 2


def make_height_map(smoothing_size=5):

    raw_map = np.random.random(noise_map_size)
    padded_map = np.pad(raw_map, smoothing_size // 2, mode='edge')
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
        self.position = pos
        self.vel = [1, 1]
        self.fitness = None
        self.p_best = pos
        self.p_best_fit = None
        self.noise_map = noise_map

    def update_fitness(self):
        tile_x = int(self.position[0] // tile_size)
        tile_y = int(self.position[1] // tile_size)
        self.fitness = self.noise_map[tile_x, tile_y]

    def update_position(self):
        self.position = np.add(self.position, self.vel)
        x, y = self.position[0], self.position[1]
        self.position[0] = min(screen_width - 10, max(0, x))
        self.position[1] = min(screen_height - 10, max(0, y))
    
    def update_p_best(self):
        if self.p_best_fit is None or self.fitness < self.p_best_fit:
            self.p_best_fit = self.fitness
            self.p_best = self.position

class Swarm:
    def __init__(self, agentnum, noise_map):
        self.agents = [Agent((random.randrange(0, screen_width), random.randrange(0, screen_height)), noise_map) for i in range(agentnum)]
        self.g_best_pos = None
        self.g_best_fit = None

    def update_global_best(self):
        for agent in self.agents:
            if self.g_best_fit is None or agent.fitness < self.g_best_fit:
                self.g_best_fit = agent.fitness
                self.g_best_pos = agent.position

def update_velocity(agent, g_best_pos):
    r1 = random.uniform(0, 1)
    r2 = random.uniform(0, 1)

    new_velocity_x = w * agent.vel[0] + c1 * r1 * (agent.p_best[0] - agent.position[0]) + c2 * r2 * (g_best_pos[0] - agent.position[0])
    new_velocity_y = w * agent.vel[1] + c1 * r1 * (agent.p_best[1] - agent.position[1]) + c2 * r2 * (g_best_pos[1] - agent.position[1])

    new_velocity = [new_velocity_x, new_velocity_y]
    return new_velocity

def main():
    noise_map = make_height_map()
    swarm = Swarm(100, noise_map)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        draw_noise_map(noise_map)
        swarm.update_global_best()

        for agent in swarm.agents:
            # pygame.draw.circle(screen, (255, 255, 255), tuple(agent.position), agent_radius)
            screen.blit(bird, agent.position)
            agent.vel = update_velocity(agent, swarm.g_best_pos)
            agent.update_p_best()
            agent.update_position()
            agent.update_fitness()

        pygame.display.flip()
        clock.tick(60)
        screen.fill((0,0,0))

main()