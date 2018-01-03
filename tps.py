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

phi_plus_pi = cv2.imread("images_tps/16_+pi.bmp")
phi_plus_pi_2 = cv2.imread("images_tps/16_+pi_na_dwa.bmp")
phi_0 = cv2.imread("images_tps/16_0.bmp")
phi_minus_pi_2 = cv2.imread("images_tps/16_-pi_na_dwa.bmp")
phi_minus_pi = cv2.imread("images_tps/16_-pi.bmp")

images = [phi_minus_pi, phi_minus_pi_2, phi_0, phi_plus_pi_2, phi_plus_pi]
alfas = [-math.pi, -math.pi/2, 0, math.pi/2, math.pi]
# for image in images:
#     print(image.shape)
#     print(image)
#     cv2.imshow("Fig", image)
# cv2.waitKey()

rows_no = images[0].shape[0]
cols_no = images[0].shape[1]
all_data = np.empty([rows_no, cols_no, len(images)])
for index,image in enumerate(images):
    gray_image = convert_image_if_needed(image, convert_to="gray")
    all_data[:,:,index] = gray_image
print(all_data.shape)
phase_all = all_data[:,:,0]
#phase = phase_calculating(all_data[128, 159, :])

# TODO Faster remapping
for index_row, row in enumerate(all_data):
    for index_col, element in enumerate(row):
        phase = phase_calculating(element)
        phase_all[index_row, index_col] = phase

phase_all_copy = phase_all.copy()
image_unwrapped = unwrap_phase(phase_all)

plt.figure(1)
plt.imshow(phase_all_copy, cmap="gray")
plt.figure(2)
plt.imshow(image_unwrapped, cmap="gray")
#
# plt.figure(0)
# plt.plot(phase_all)
# plt.colorbar()

# plt.figure(1)
# plt.plot(image_unwrapped)
# plt.colorbar()

print(image_unwrapped.shape)


# Przjeście po wszystkich kolumnach
temp = image_unwrapped.T
maximum = max(temp.flatten())
minimum = min(temp.flatten())
print(minimum, maximum)
temp = np.add(temp, np.abs(minimum))
image_unwrapped = np.add(image_unwrapped, np.abs(minimum))

for col_index, col in enumerate(temp):
    for index, element in enumerate(col):
        temp[col_index][index] = element - 2*2*math.pi*1/16*index

bez_stalej = temp.T
plt.figure(3)
plt.imshow(bez_stalej, cmap="gray")


diff = np.subtract(bez_stalej, image_unwrapped)
number_of_non_zero_elems = np.count_nonzero(diff)
print(number_of_non_zero_elems)

plt.show()


# equation  I_xy = a_xy + b_xy * math.cos(phi_xy + alfa)
# for each pixel
#for alfa in alfas:



