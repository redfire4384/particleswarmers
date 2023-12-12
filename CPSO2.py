# Implementation of the Chaotic Particle Swarm Optimization algorithm

import pygame
import random
import numpy as np
from quicksort import quicksort

pygame.init()

# Constants
c1, c2 = 2, 2
w = 0.0001
k = 4
u = 1
y = 0.5
screen_width, screen_height = 800, 800
tile_size = 10
swarm_size = 100
max_iters = 5000
bird = pygame.image.load("bird.png")

# Derived Constants
map_size = (screen_width // tile_size, screen_height // tile_size)
screen = pygame.display.set_mode((screen_width, screen_height))
agent_radius = tile_size // 2
clock = pygame.time.Clock()

def make_tent_map(chaosnum):
    # Generate chaos vector of size chaosnum
    def gen_z():
        running = True
        while running:
            x = random.uniform(0,1)
            if x not in [0,.25,.5,.75,1]:
                running = False
        return x
    
    z = [gen_z()]
    for i in range(chaosnum-1):
        z.append(u*(1-2*abs(z[i]-0.5)))
    return z

def chaotic_disturbance(z, min, maxx, maxy):
    # add chaotic disturbance to given array z with search space bounds min maxx and maxy

    psi = optimum_cv(z, min, maxx, maxy)
    zprime = [[((1-y)*psi[i][0] + y * z[i].position[0]), ((1-y)*psi[i][1] + y * z[i].position[1])]for i in range(len(z)-1)]
    return zprime

def optimum_cv(x, min, maxx, maxy):
    # calculate optimal chaos vector for given array x within search space bounded by min maxx and maxy
     
    psi = [[(i.p_best[0] - min)/(maxx - min), (i.p_best[1] - min)/(maxy - min)] for i in x]
    return psi

def make_height_map(smoothing_size=5):
    # Creates random search space

    raw_map = np.random.random(map_size)
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
    # Converts values to a corresponding color value

    # #011632 to #85D5E6
    return (1 + int(value * 132), 22 + int(value * 191), 50 + int((value) * 180))

def draw_noise_map(noise_map):
    # Draws noise map to pygame window

    for x in range(map_size[0]):
        for y in range(map_size[1]):
            color = value_to_color(noise_map[x, y])
            pygame.draw.rect(screen, color, (x * tile_size, y * tile_size, tile_size, tile_size))


def update_velocity(agent, g_best_pos):
    # Updates velocity based on an agents current positon, their previous best position, the swarm's global best position, and a given inertia weight and acceleration constants

    r1 = random.uniform(0, 1)
    r2 = random.uniform(0, 1) 


    new_velocity_x = w * agent.vel[0] + c1 * r1 * (agent.p_best[0] - agent.position[0]) + c2 * r2 * (g_best_pos[0] - agent.position[0])
    new_velocity_y = w * agent.vel[1] + c1 * r1 * (agent.p_best[1] - agent.position[1]) + c2 * r2 * (g_best_pos[1] - agent.position[1])

    new_velocity = [new_velocity_x, new_velocity_y]
    return new_velocity

class Agent:
    # Particle object
    def __init__(self, pos, noise_map):
        self.position = pos
        self.vel = [1, 1]
        self.fitness = None
        self.p_best = pos
        self.p_best_fit = None
        self.noise_map = noise_map

    def update_fitness(self):
        # Updates fitness of agent based on current position

        tile_x = int(self.position[0] // tile_size)
        tile_y = int(self.position[1] // tile_size)
        self.fitness = self.noise_map[tile_x, tile_y]

    def update_position(self):
        # Updates position of agent based on velocity, bounded within the window

        self.position = np.add(self.position, self.vel)
        x, y = self.position[0], self.position[1]
        self.position[0] = min(screen_width - 10, max(0, x))
        self.position[1] = min(screen_height - 10, max(0, y))
    
    def update_p_best(self):
        # Updates personal best fitness and position

        if self.p_best_fit is None or self.fitness < self.p_best_fit:
            self.p_best_fit = self.fitness
            self.p_best = self.position

class Swarm:
    # Swarm object
    def __init__(self, agentnum, noise_map):
        xvars = make_tent_map(agentnum)
        yvars = make_tent_map(agentnum)
        self.z = [xvars, yvars]
        self.agents = self.gen_agents(0, screen_width, screen_height, noise_map, agentnum,self.z) 
        self.g_best_pos = None
        self.g_best_fit = None

    def distance_to_g_best(self, xy):
        # calculates distance to g best
        return np.sum(np.square(np.array(self.g_best_pos) - np.array(xy)))

    def update_global_best(self):
        # Updates global best position and fitness 
        for agent in self.agents:
            if self.g_best_fit is None or agent.fitness < self.g_best_fit:
                self.g_best_fit = agent.fitness
                self.g_best_pos = agent.position
    
    def gen_agents(self, min, maxx, maxy, noisemap, agentnum,z):
        # generates agents
        agents = [Agent([min + z[0][i]*(maxx-min), min + z[1][i]*(maxy-min)], noisemap) for i in range(agentnum)]
        return agents
    
    def rank_agents(self):
        # orders agents in ascending order by pbest 

        dist = [[self.distance_to_g_best(agent.p_best), agent] for agent in self.agents]
        ranked = quicksort(dist, 0, len(dist) - 1)
        return ranked
    

def main():
    # executes code

    noise_map = make_height_map()
    swarm = Swarm(swarm_size, noise_map)

    running = True
    condition = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
        # Runs pygame
        if condition:
            for i in range(max_iters):
                
                # Closes pygame even if window closed before max_iters reached
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break
                if not running:
                    break



                # Runs PSO loop
                draw_noise_map(noise_map)
                swarm.update_global_best()
                swarm.rank_agents()


                for agent in swarm.agents:
                    old_pos = agent.p_best
                    screen.blit(bird, agent.position)
                    agent.vel = update_velocity(agent, swarm.g_best_pos)
                    agent.update_position()
                    agent.update_p_best()
                    agent.update_fitness()

                    # Adds chaotic disturbance to bottom 30% of agents after being ranked by p_best
                    if old_pos[0] == agent.p_best[0] and old_pos[1] == agent.p_best[1] and i <= max_iters//2:
                        bottom_thirty = swarm.agents[1 - int(len(swarm.agents)*0.3):]
                        disturbance = chaotic_disturbance(bottom_thirty, 0, screen_width, screen_height)
                        for i in range(len(disturbance)-1):
                            bottom_thirty[i].position = [0 + disturbance[i][0]*(screen_width-0), 0 + disturbance[i][1]*(screen_height-0)]

                        


                pygame.display.flip()
                clock.tick(60)
                screen.fill((0,0,0))
            condition = False

# Startup Code

main()