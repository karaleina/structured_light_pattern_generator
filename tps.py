import cv2
import math
import numpy as np
from general_usefull_package.images import convert_image_if_needed
from skimage.restoration import unwrap_phase
from matplotlib import pyplot as plt


def phase_calculating(pixel):
    if 2*pixel[2] - pixel[0] - pixel[4] == 0:
        return math.pi/2
    else:
        return 2*np.arctan(2*(pixel[1] - pixel[3])/(2*pixel[2] - pixel[0] - pixel[4]))


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


def main(images):

    rows_no = images[0].shape[0]
    cols_no = images[0].shape[1]
    all_data = np.empty([rows_no, cols_no, len(images)])

    for index,image in enumerate(images):
        gray_image = convert_image_if_needed(image, convert_to="gray")
        all_data[:,:,index] = gray_image

    phase_all = all_data[:,:,0]

    # Apply phase calculating from 5 frames
    for index_row, row in enumerate(all_data):
        for index_col, element in enumerate(row):
            phase = phase_calculating(element)
            phase_all[index_row, index_col] = phase

    # Unwrapping
    image_unwrapped = unwrap_phase(phase_all)

    # Moving values
    image_unwrapped_positive_range = moving_values_to_the_positive_range(image_unwrapped)

    # Substracting fixed component
    bez_stalej = substate_fixed_component(image_unwrapped_positive_range)

    return bez_stalej


if __name__ == "__main__":

    phi_plus_pi = cv2.imread("images_tps/16_+pi.bmp")
    phi_plus_pi_2 = cv2.imread("images_tps/16_+pi_na_dwa.bmp")
    phi_0 = cv2.imread("images_tps/16_0.bmp")
    phi_minus_pi_2 = cv2.imread("images_tps/16_-pi_na_dwa.bmp")
    phi_minus_pi = cv2.imread("images_tps/16_-pi.bmp")

    alfas = [-math.pi, -math.pi / 2, 0, math.pi / 2, math.pi]
    images = [phi_minus_pi, phi_minus_pi_2, phi_0, phi_plus_pi_2, phi_plus_pi]
    bez_stalej = main(images)
    plt.figure(3)
    plt.imshow(bez_stalej, cmap="gray")
    plt.show()