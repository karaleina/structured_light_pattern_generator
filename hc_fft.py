import numpy as np
import cv2
import math
import os
from matplotlib import pyplot as plt
from general_usefull_package.images import convert_image_if_needed
from general_usefull_package.plotting import count_ifft, count_fft, count_fft_imag, filter_fft
import scipy.io
from scipy.signal import resample
from skimage.restoration import unwrap_phase
from scps import moving_values_to_the_positive_range

def substate_fixed_component(signal):
    new_signal = signal.copy()
    for index, element in enumerate(new_signal):
       new_signal[index] = element -  2 * math.pi * 1 / 8 * index
    return new_signal


class HCAnalyser(object):

    def __init__(self, vo):
        self.__list_of_images = []
        self.__vectors_dataset = None
        self.__phase_image = None
        self.__vo = vo


    def analyse(self, avi):

        self.__read_all_frames(avi)

        # TODO and create vectors dataset

        self.__create_vectors_dataset()

        # TODO calculate FFT for every vector

        self.__calculate_phase_image()

        # TODO Filter FFT

        # TODO IFFT

        # TODO PHASE CALCULATION

        # TODO UNWRAPPING

        # TODO SUBSTRACT FIXED COMPONENT

        pass


    def __add_frame_to_dataset(self, frame):
        self.__list_of_images.append(frame)

    def __read_all_frames(self, avi):
        cap = cv2.VideoCapture(avi)
        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            frame_index += 1
            if frame is None:
                break
            gray = convert_image_if_needed(frame, convert_to="gray")
            self.__add_frame_to_dataset(gray)
            cv2.imshow('frame', gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def __create_vectors_dataset(self):

        self.__vectors_dataset = np.array(self.__list_of_images)
        print(self.__vectors_dataset, self.__vectors_dataset.shape)

        data = self.__vectors_dataset
        scipy.io.savemat('data.mat', mdict={'data': data})

    def __calculate_phase_image(self):

        dataset = self.__vectors_dataset

        pixel_signal = dataset[:,350, 880]
        self.__calculate_phase_from_pixel_signal(pixel_signal)

        pixel_signal = dataset[:, 600, 900]
        self.__calculate_phase_from_pixel_signal(pixel_signal)


        # self.__phase_image = np.zeros((len(dataset[0,:,0]), len(dataset[0,0,:])))
        # for row_index in range(len(dataset[0,:,0])):
        #     for col_index in range(len(dataset[0,0,:])):
        #         pixel_signal = dataset[:,row_index, col_index]
        #         phase = self.__calculate_phase_from_pixel_signal(pixel_signal)
        #         self.__phase_image[row_index, col_index] = phase


    def __calculate_phase_from_pixel_signal(self, pixel_signal):

        # FFT
        [xf, yf] = count_fft_imag(pixel_signal, number_of_samples_to_add=4*len(pixel_signal), T=self.__vo)

        # FILTRACJA
        new_y = filter_fft(xf, yf, f_min=1/2*self.__vo, f_max=3/2*self.__vo)

        # ODWROTNE IFFT
        signal = count_ifft(new_y)

        # RESAMPLING
        resampled_signal = resample(signal, num=len(pixel_signal))

        # ARCTANG
        signal_phase = np.array([2*np.arctan(np.imag(el)/np.real(el)) for el in resampled_signal])

        # UNWRAPPING
        unwrapped_phase = unwrap_phase(signal_phase)

        # POSITIVE_VALUES
        positive_range_of_unwrapped_phase = moving_values_to_the_positive_range(unwrapped_phase)

        # SUBSTRACTING CONTANT
        substracted_constant_from_the_phase = substate_fixed_component(positive_range_of_unwrapped_phase)

        # PLOTTING
        plt.figure(1)
        plt.plot(pixel_signal)
        plt.title("Signal_from_signal_pixel")

        plt.figure(2)
        plt.plot(xf, yf)
        plt.title("FFT")

        plt.figure(3)
        plt.plot(xf, new_y)
        plt.title("Filtred_fft")

        plt.figure(4)
        plt.plot(signal)
        plt.title("Reconstructed_signal")

        plt.figure(5)
        plt.plot(resampled_signal)
        plt.title("Resampled_signal")

        plt.figure(6)
        plt.plot(signal_phase)
        plt.title("Phase")

        plt.figure(7)
        plt.plot(positive_range_of_unwrapped_phase)
        plt.title("Unwrapped")

        plt.figure(8)
        plt.plot(substracted_constant_from_the_phase)
        plt.title("substracted_constant_from_the_phase")

        plt.show()


if __name__ == "__main__":
    hc = HCAnalyser(vo=1/8)
    avi_path = os.path.abspath("C:\\Users\\ImioUser\\Desktop\\K&A\\ACTIVE3D\\ODDECH_paski_21_12_17\\8.avi")
    hc.analyse(avi=avi_path)




