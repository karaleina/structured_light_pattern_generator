import cv2


def convert_image_if_needed(image, convert_to="gray"):
    if convert_to == "gray":
        if len(image.shape) < 3:
            return image
        elif len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            print("Nie wiadomo, co to za obraz! :(")

    elif convert_to == "color":
        if len(image.shape) < 3:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3:
            return image
        else:
            print("Nie wiadomo, co to za obraz! :(")
    else:
        print("Zdefiniuj poprawnie paramter 'convert_to', "
              "ktory moze przyjmowac wartosci 'gray' lub 'color'")