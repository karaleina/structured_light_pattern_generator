from scipy.fftpack import fft, ifft
from scipy.signal import argrelextrema
from scipy.signal import freqz
# from filters import filters
from matplotlib import pyplot as plt
import numpy as np
import cv2
# from math_objects import math_objects


def count_fft(signal, T):

        new_y = signal.ravel()
        # FFT
        N = len(new_y)

        yf = fft(new_y)
        xf = np.linspace(0.0, 1.0 / (2 * T), N // 2)

        yf = yf / len(yf)
        yf_abs = np.abs(yf[0:len(yf) // 2])

        return [xf, yf_abs]

def count_fft_imag(signal, number_of_samples_to_add=0, T=0.02):
        new_y = signal.ravel()
        # FFT
        N = len(new_y)

        yf = fft(new_y, n=number_of_samples_to_add)
        xf = np.linspace(0.0, 1.0 / (2 * T), number_of_samples_to_add // 2)

        yf = yf / len(yf)
        yf = yf[0:len(yf) // 2]

        return [xf, yf]

def count_ifft(fft, xf=None, number_of_samples=None):
        return ifft(fft)

def filter_fft(xf, yf, f_min=None, f_max=None):
        new_y = yf
        for index, (x, y) in enumerate(zip(xf, yf)):
            if x<f_min or x>f_max:
                new_y[index] = 0
            else:
                new_y[index] = y

        return new_y

#
# def find_threshold_in_histogram(histogram, bin_edges, plot=True):
#
#     p_max = math_objects.Point(x=np.argmax(histogram), y=np.max(histogram))
#     #p_min = Point(x=np.argmin(histogram), y=np.min(histogram))
#
#     p_min = math_objects.Point(x=len(histogram)-1, y=histogram[-1])
#     # TODO Znalezc dobre minimum
#     line = math_objects.Line(p_min, p_max)
#     [a, b] = line.get_coeffs()
#
#     left_edge = min(p_max.get_x(),p_min.get_x())
#     right_edge = max(p_max.get_x(),p_min.get_x())
#
#     distances = np.empty_like(histogram)
#     for index, element in enumerate(histogram):
#         if not left_edge < index < right_edge:
#             distances[index] = -1
#         else:
#             new_point = math_objects.Point(x=index, y=element)
#             distances[index] = new_point.get_distance_to_the_line(line=line)
#
#     threshold_index = np.argmax(distances)
#     x_axis = range(len(histogram))
#     y_axis = np.multiply(x_axis, a) + b
#
#     if plot==True:
#         # Plotting
#         plt.figure(17).clear()
#         plt.bar(range(len(histogram)), histogram)
#         plt.plot(x_axis, y_axis)
#
#     return threshold_index

#
# def plotting_and_fft_calculating( breathing_signal, fs, lowcut=0.08, highcut = 2, figure_no=1):
#     if breathing_signal == None:
#         print("Brak oddechu")
#
#         plt.figure(figure_no).clear()
#         plt.title("Signal is none :( ")
#
#     else:
#         plt.ion()
#         plt.figure(figure_no).clear()
#         plt.subplot(3, 2, 1)
#         plt.plot(breathing_signal)
#
#         plt.subplot(3, 2, 2)
#         plt.title("FFT")
#         xf, yf_abs = count_fft(breathing_signal, T=1/fs)
#         plt.plot(xf, yf_abs)
#         plt.xlabel("[Hz]")
#
#         # Filtration
#         plt.subplot(3, 2, 3)
#         for order in [4]:
#             b, a = filters.butter_bandpass(lowcut, highcut, fs, order=order)
#             w, h = freqz(b, a, worN=len(yf_abs))
#             plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
#             abs_filter = abs(h)
#
#         # Filter window
#         plt.subplot(3, 2, 4).cla()
#         plt.xlabel('[Hz]')
#         plt.grid(True)
#         plt.legend(loc='best')
#         plt.title("Butterworth filter")
#
#         filtered_fft = abs_filter * yf_abs
#         plt.plot(xf, filtered_fft)
#         plt.title("Odflitrowane fft")
#
#         # Cropped filtration
#         plt.subplot(3, 2, 5).cla()
#         cropped_xf = xf[xf <= 0.8 * highcut]
#         cropped_filtered_fft = filtered_fft[0:len(cropped_xf)]
#         cropped_xf = cropped_xf[0:len(cropped_xf // 2)]
#         plt.plot(cropped_xf, cropped_filtered_fft)
#
#         # Peaks
#
#         plt.subplot(3, 2, 6).cla()
#         plt.plot(cropped_xf, cropped_filtered_fft, "*-")
#         plt.xlabel("[Hz]")
#
#         maxima = argrelextrema(cropped_filtered_fft, np.greater)
#         maxima = maxima[0]
#
#         true_maximas_freq = []
#         if len(maxima) > 0:
#             max_fft = None
#             index_of_max_fft = -1
#             for index in maxima:
#                 current_max_fft = cropped_filtered_fft[index]
#                 if max_fft == None or current_max_fft >= max_fft:
#                     max_fft = current_max_fft
#                     index_of_max_fft = index
#
#             true_maximas_freq.append(index_of_max_fft)
#
#             breath_freq = cropped_xf[index_of_max_fft]
#             breath_signal = breath_freq * 60
#             plt.title(str(round(float(breath_signal), 2)) + " oddechów/min")
#
#             max_fft_signal = cropped_filtered_fft[index_of_max_fft]
#             for max_index in maxima:
#                 plt.plot(cropped_xf[max_index], cropped_filtered_fft[max_index], "bo")
#                 if cropped_filtered_fft[max_index] > 1 / 3 * max_fft_signal and max_index != index_of_max_fft:
#                     true_maximas_freq.append(cropped_xf[max_index])
#         else:
#             print("Nie znaleziono maximów!")
#
#
#         plt.pause(0.005)
#
# #         return true_maximas_freq
#
#
# def simple_plotting(breathing_signal, fs, lowcut=0.08, highcut=2, figure_no=1):
#     if breathing_signal is None:
#         print("Brak oddechu")
#
#         plt.figure(figure_no).clear()
#         plt.grid()
#         plt.title("Signal is none :( ")
#
#     else:
#         breathing_signal = np.array(breathing_signal)
#         breathing_signal = breathing_signal.ravel()
#
#
#         plt.figure(figure_no).clear()
#         plt.subplot(2, 1, 1)
#
#         x = np.linspace(0, len(breathing_signal)-1,num=len(breathing_signal))
#         plt.plot(x, breathing_signal)
#         plt.grid()
#         plt.title("Breath signal")
#         plt.xlabel("[n]")
#         plt.tight_layout()
#
#         additional_zeros = np.zeros(7*len(breathing_signal))
#         breathing_signal = np.concatenate([breathing_signal, additional_zeros])
#
#         xf, yf_abs = count_fft(breathing_signal, 1/fs)
#
#         # Filtration
#         for order in [4]:
#             b, a = filters.butter_bandpass(lowcut, highcut, fs, order=order)
#             w, h = freqz(b, a, worN=len(yf_abs))
#             abs_filter = abs(h)
#
#         filtered_fft = abs_filter * yf_abs
#
#         # Cropped filtration
#
#         cropped_xf = xf[xf <= 0.8 * highcut]
#         cropped_filtered_fft = filtered_fft[0:len(cropped_xf)]
#         cropped_xf = cropped_xf[0:len(cropped_xf // 2)]
#
#         # Peaks
#         plt.subplot(2, 1, 2).cla()
#         plt.plot(cropped_xf, cropped_filtered_fft, "*-")
#         plt.xlabel("Frequency [Hz]")
#
#         maxima = argrelextrema(cropped_filtered_fft, np.greater)
#         maxima = maxima[0]
#
#         true_maximas_freq = []
#         if len(maxima) > 0:
#             max_fft = None
#             index_of_max_fft = -1
#             for index in maxima:
#                 current_max_fft = cropped_filtered_fft[index]
#                 if max_fft == None or current_max_fft >= max_fft:
#                     max_fft = current_max_fft
#                     index_of_max_fft = index
#
#             true_maximas_freq.append(index_of_max_fft)
#
#             breath_freq = cropped_xf[index_of_max_fft]
#             breath_signal = breath_freq * 60
#             plt.title("FFT (breath rate: " + str(round(float(breath_signal), 2)) + "1/s)")
#
#             max_fft_signal = cropped_filtered_fft[index_of_max_fft]
#             for max_index in maxima:
#                 plt.plot(cropped_xf[max_index], cropped_filtered_fft[max_index], "bo")
#
#                 if cropped_filtered_fft[max_index] > 1 / 3 * max_fft_signal and max_index != index_of_max_fft:
#                     true_maximas_freq.append(cropped_xf[max_index])
#         else:
#             print("Nie znaleziono maximów!")
#         plt.grid()
#
#         plt.tight_layout()
#         return true_maximas_freq


# def threshold_histogram_plotting(signal, bins):
#     hist, bin_edges = np.histogram(signal, bins=bins)
#     threshold = find_threshold_in_histogram(hist, bin_edges)
#     good_piramids_indexes = [index for index, i in enumerate(signal) if 0.1 * bins < i < threshold]
#
#     plt.figure(100000)
#     plt.hist(signal)
#
#     # TODO Automatic threshold
#
#     # Showing frames with coloured speckles
#     filepaths = filepaths.get_all_file_paths(os.getcwd())
#     for index in range(len(filepaths)):
#         if index < len(filepaths) - 1:
#             frame = cv2.imread(filepaths[index])
#             analyser.draw_points_with_high_correlation(
#                 index, frame, good_piramids_indexes)
#     cv2.waitKey()