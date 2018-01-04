import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from general_usefull_package.images import convert_image_if_needed
from general_usefull_package.plotting import count_fft
import scipy.io


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
        plt.figure(1)
        plt.plot(pixel_signal)


        [xf, yf] = count_fft(pixel_signal, T=self.__vo)
        plt.figure(2)
        plt.plot(xf, yf)

        # BARLET

        print("Vo", self.__vo)
        f_min = 1/2*self.__vo
        f_max = 3/2*self.__vo

        print(f_min, f_max)

        new_y = yf
        for index, (x, y) in enumerate(zip(xf, yf)):
            if x<f_min or x>f_max:
                new_y[index] = 0
            else:
                new_y[index] = y

        plt.figure(3)
        plt.plot(xf, new_y)
        plt.show()

        #barlet = np.bartlett(self.__vo)

        # print(xf, barlet)

        phase = None
        return phase





if __name__ == "__main__":
    hc = HCAnalyser(vo=1/8)
    avi_path = os.path.abspath("C:\\Users\\ImioUser\\Desktop\\K&A\\ACTIVE3D\\ODDECH_paski_21_12_17\\8.avi")
    hc.analyse(avi=avi_path)




