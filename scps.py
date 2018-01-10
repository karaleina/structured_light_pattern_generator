import cv2
import math
import numpy as np
from general_usefull_package.images import convert_image_if_needed
from skimage.restoration import unwrap_phase
from matplotlib import pyplot as plt
import os


def phase_calculating(image):

    old_image = image.T
    phase_image = np.empty_like(image, dtype="float")

    # Index row refers to original image
    # Index col also refers to original image
    for index_col, col in enumerate(old_image):

        if index_col == 4:
            plt.figure(17)
            plt.plot(col[:100])


        if 10 <= index_col <= (len(old_image[:,0]) - 11):

            if index_col == 3:
                print(col[0:5])

            for index_row, element in enumerate(col):
                if 10 <= index_row <= (len(old_image[0, :]) - 11):
                    c = math.pow((int(col[index_row-5]) - int(col[index_row+5])), 2)
                    d = math.pow((int(col[index_row-10]) - int(col[index_row+10])), 2)
                    a = np.power(4*c - d, 1/2)
                    b = 2*int(col[index_row]) - int(col[index_row-10]) - int(col[index_row+10])

                    # if math.isnan(a) or b==0:
                    #     phase_image[index_row, index_col] = 0
                    # else:
                    phase_image[index_row, index_col] = np.arctan(a/b) * 2

        if index_col == 4:
            plt.figure(18)
            plt.plot(phase_image[:,4][:100])
            plt.show()

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
            temp[col_index][index] = element - 2 * 2 * math.pi * 1 / 8 * index
    return temp.T


def main(image):

    # TODO
    phase_all = phase_calculating(image)

    # Unwrapping
    image_unwrapped = unwrap_phase(phase_all)

    # Moving values
    image_unwrapped_positive_range = moving_values_to_the_positive_range(image_unwrapped)

    # Substracting fixed component
    bez_stalej = substate_fixed_component(image_unwrapped_positive_range)

    return phase_all, image_unwrapped, bez_stalej


def read_frame_from_video(avi):

    cap = cv2.VideoCapture(avi)
    while cap.isOpened():
        ret, frame = cap.read()
        gray = convert_image_if_needed(frame, convert_to="gray")

        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return gray


if __name__ == "__main__":

    # avi_path = os.path.abspath("C:\\Users\\ImioUser\\Desktop\\K&A\\ACTIVE3D\\ODDECH_paski_21_12_17\\8.avi")
    # image = np.array(read_frame_from_video(avi=avi_path))
    #
    img = cv2.imread("test_20.bmp")
    #image = cv2.imread("test.bmp")

    # image = cv2.imread("images_tps/16_+pi.bmp")
    # image = cv2.imread("images/test.png")


    image = convert_image_if_needed(img)
    phase_all, image_unwrapped, bez_stalej = main(image)

    plt.figure(1)
    plt.imshow(image)
    plt.figure(2)
    plt.imshow(phase_all, cmap="gray")
    plt.figure(3)
    plt.imshow(image_unwrapped, cmap="gray")
    plt.figure(4)
    plt.imshow(bez_stalej, cmap="gray")
    plt.show()


    print("NIC")