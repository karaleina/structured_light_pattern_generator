import cv2
import math
import numpy as np
from general_usefull_package.images import convert_image_if_needed
from skimage.restoration import unwrap_phase
from matplotlib import pyplot as plt



def phase_calculating(image):


    old_image = image.T
    phase_image = np.empty_like(image)

    # Index row refers to original image
    # Index col also refers to original image
    for index_col, col in enumerate(old_image):
        if 2 <= index_col <= (len(old_image[:,0]) - 2):
            for index_row, element in enumerate(col):
                if 2 <= index_row <= (len(old_image[:, 0]) - 2):
                    # TODO
                    c = np.power((col(index_row-1) - col(index_row+1)), 2)
                    d = np.power((col(index_row-2) - col(index_row+2)), 2)
                    a = np.power(4*c - d, 1/2)
                    b = 2*col(index_row) - col(index_row-2) - col(index_row+2)
                    phase_image[index_row, index_col] = np.arctan(a/b)

    return phase_image


def moving_values_to_the_positive_range(array):
    minimum = min(array.flatten())
    if minimum < 0:
        return np.add(array, np.abs(minimum))
    else:
        return array


def substate_fixed_component(image_unwrapped):
    temp = image_unwrapped.T
    for col_index, col in enumerate(temp):
        for index, element in enumerate(col):
            temp[col_index][index] = element - 2 * 2 * math.pi * 1 / 16 * index
    return temp.T


def main(image):

    # Apply phase calculating from 5 frames
    # TODO
    phase_all = phase_calculating(image)

    # Unwrapping
    image_unwrapped = unwrap_phase(phase_all)

    # Moving values
    image_unwrapped_positive_range = moving_values_to_the_positive_range(image_unwrapped)

    # Substracting fixed component
    bez_stalej = substate_fixed_component(image_unwrapped_positive_range)

    return bez_stalej


if __name__ == "__main__":

    phi_plus_pi = cv2.imread("images_tps/16_+pi.bmp")
    bez_stalej = main(phi_plus_pi)
    plt.figure(3)
    plt.imshow(bez_stalej, cmap="gray")
    plt.show()