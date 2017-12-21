import tkinter
from PIL import ImageTk, Image
import os
import cv2
import base64

# class FullScreenApp(object):
#     def __init__(self, master, **kwargs):
#         self.master = master
#         pad = 3
#         self._geom = '200x200+0+0'
#         master.geometry("{0}x{1}+0+0".format(
#             master.winfo_screenwidth() - pad, master.winfo_screenheight() - pad))
#         master.bind('<Escape>', self.toggle_geom)
#
#     def toggle_geom(self, event):
#         geom = self.master.winfo_geometry()
#         print(geom, self._geom)
#         self.master.geometry(self._geom)
#         self._geom = geom
#
#
#
#
# root = tkinter.Tk()
# app = FullScreenApp(root)
# root.mainloop()

if __name__ == "__main__":

    root = tkinter.Tk()
    raw_image = cv2.imread("test.png")
    # next_image = base64.encodebytes(raw_image)

    image = tkinter.PhotoImage(raw_image)
    label = tkinter.Label(image)
    label.pack()
    # window_label = tkinter.Label(root, text="Hello, world!")
    # root.overrideredirect(True)
    # root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))
    # root.focus_set()  # <-- move focus to this widget
    # root.bind("<Escape>", lambda e: root.quit())
    # window_label.pack()

    root.mainloop()