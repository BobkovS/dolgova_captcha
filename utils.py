from io import BytesIO

import cv2
import imutils
import numpy as np
from PIL import Image


def fill_image(image):
    """
    Заполняем пропуски на изображении
    """
    h, w = image.shape[:2]
    new_image = image.copy()
    for i in range(h):
        for j in range(w):
            if image[i, j] != 255:
                for k in range(-1, 2, 1):
                    for m in range(-1, 2, 1):
                        new_image[i + k, j + m] = 0
    return new_image


def erosion(image):
    """
    Морфологическая эррозия
    """
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    return image


def gif2bytes(gif_image):
    with BytesIO() as f:
        gif_image.save(f, format='PNG')
        f.seek(0)
        png_image = Image.open(f).convert('RGB')
        return png_image


def gif2png(gif_image):
    """
    Преобразование изображения из GIF формата в PNG
    """

    gif_image = cv2.cvtColor(gif_image, cv2.COLOR_BGR2RGB)
    gif_image = Image.fromarray(gif_image)
    png_image = gif2bytes(gif_image)
    open_cv_image = cv2.cvtColor(np.array(png_image), cv2.COLOR_RGB2GRAY)
    image = cv2.resize(open_cv_image, (200, 100), 0)
    return image


def resize_to_fit(image, width, height):
    """
    Приведение изображения к одному размеру для использования в модели
    """
    (h, w) = image.shape[:2]

    if w > h:
        image = imutils.resize(image, width=width)

    else:
        image = imutils.resize(image, height=height)

    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)

    image = cv2.copyMakeBorder(image, padH, padH, padW, padW,
                               cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))

    return image
