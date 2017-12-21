from pygame.locals import *
import pygame
import time


def main():
    picture = pygame.image.load('test.png')
    pygame.display.set_mode((800,600), pygame.FULLSCREEN)
    main_surface = pygame.display.get_surface()
    a = 1
    while a == 1:
        main_surface.blit(picture, (0, 0))
        pygame.display.update()
        time.sleep(5)
        a = 0


if __name__ == "__main__":
    main()