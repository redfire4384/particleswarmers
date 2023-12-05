import pygame

pygame.init()
screen_width, screen_height = 800, 600
screen = pygame.display.set_mode((screen_width, screen_height))
clock = pygame.time.Clock()
agent_radius = 5

class Agent:
    def __init__(self):
        self.x = screen_width // 2
        self.y = screen_height // 2

    def update_position(self):
        self.x += 1
        self.y += 1


def main():
    agents = [Agent() for i in range(1)]
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        for agent in agents:
            pygame.draw.circle(screen, (255, 255, 255), (agent.x, agent.y), agent_radius)
            agent.update_position()

        pygame.display.flip()
        clock.tick(60)
        screen.fill((0,0,0))

main()