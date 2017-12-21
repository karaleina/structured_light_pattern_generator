import numpy as np
import cv2
import math
import pygame
import time

class PatternGenerator(object):

    def __init__(self):
        pass
        # self.__phi = None
        # self.__T = None
        # self.__w = None
        # self.__h = None
        # self.__image = None

    def calculate_vertical_pattern(self, phi, T, w, h):

        # Creating frame with given size
        image = np.empty((h, w))

        # Calculating intensity
        omega = 2 * math.pi / T
        y_single_row = [math.sin(omega*x + phi) for x in range(w)]
        for index, row in enumerate(image):
            image[index] = y_single_row

        # Scaling sinus to 0-1
        image = (image + 1) / 2

        return image

    def calculate_horizontal_pattern(self, phi, T, w, h):
        return self.calculate_vertical_pattern(phi, T, h, w).T

    def pattern_sin(self, w, h, phases=[], periods=[], directions=[]):
        images = []
        names = []
        for direction in directions:
            for period in periods:
                for phi in phases:
                    if direction == 90:
                        images.append(self.calculate_vertical_pattern(phi, T=period, w=w, h=h))
                        names.append("phi=" + str(phi) + " T=" + str(period) + " direct=" + str(direction))
                    elif direction == 0:
                        images.append(self.calculate_horizontal_pattern(phi, T=period, w=w, h=h))
                        names.append("phi=" + str(phi) + " T=" + str(period) + " direct=" + str(direction))

        return images, names


def gray(im):
    im = 255 * (im / im.max())
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = im
    return ret


def get_picture(index, images):
    Z = images[index].T
    Z = 255 * Z / Z.max()
    Z = gray(Z)

    return pygame.surfarray.make_surface(Z)


def main(images, names, w, h):

    for index, (image, name) in enumerate(zip(images, names)):
        cv2.imshow("Pattern_" + str(name), image)
        cv2.waitKey(200)
    cv2.destroyAllWindows()

    pygame.display.set_mode((w, h), pygame.FULLSCREEN)
    main_surface = pygame.display.get_surface()
    a = 0
    while a < len(images):
        Z = get_picture(index=a, images=images)
        main_surface.blit(Z, (0, 0))
        pygame.display.update()
        time.sleep(600)
        a += 1


if __name__ == "__main__":
    #pg = PatternGenerator()

    # pattern = pg.calculate_horizontal_pattern(phi=20, T=20, w=500, h=400)
    # print(pattern)
    # cv2.imshow("Pattern_horizontal", pattern)
    # pattern2 = pg.calculate_vertical_pattern(phi=20, T=20, w=500, h=400)
    # print(pattern2)
    # cv2.imshow("Pattern_vertical", pattern2)

    #phases = list(np.linspace(0, 90, num=5)) # radiany
    # images, names = pg.pattern_sin(w=800, h=600, phases=phases, periods=[160], directions=[0])
    # for index, (image, name) in enumerate(zip(images, names)):
    #     cv2.imshow("Pattern_" + str(name), image)

    w = 1600
    h = 1200
    pg = PatternGenerator()
    phases = list(np.linspace(0, 90, num=5))
    images, names = pg.pattern_sin(w=w, h=h, phases=[+math.pi], periods=[16], directions=[0])
    for name in names: print(name)
    main(images, names, w, h)