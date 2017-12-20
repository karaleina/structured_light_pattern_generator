import numpy as np
import cv2
import math

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

        return image

    def calculate_horizontal_pattern(self, phi, T, w, h):
        return self.calculate_vertical_pattern(phi, T, h, w).T

    def pattern_sin(self, w, h, phases=[], periods=[], directions=[]):
        images = []
        for direction in directions:
            for period in periods:
                for phi in phases:
                    if direction == 0:
                        images.append(self.calculate_vertical_pattern(phi, T=period, w=w, h=h))
                    elif direction == 90:
                        images.append(self.calculate_horizontal_pattern(phi, T=period, w=w, h=h))
        return images



if __name__ == "__main__":
    pg = PatternGenerator()
    pattern = pg.calculate_horizontal_pattern(phi=20, T=20, w=500, h=400)
    print(pattern)
    cv2.imshow("Pattern_horizontal", pattern)
    pattern2 = pg.calculate_vertical_pattern(phi=20, T=20, w=500, h=400)
    print(pattern2)
    cv2.imshow("Pattern_vertical", pattern2)

    images = pg.pattern_sin(w=500, h=400, phases=[0, 10, 30], periods=[8, 90, 50], directions=[0, 90])
    for index, image in enumerate(images):
        cv2.imshow("Pattern" + str(index), pattern)

    cv2.waitKey()

    # TODO add information about pattern