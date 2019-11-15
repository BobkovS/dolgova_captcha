import os
import pickle
import random
from io import BytesIO

import cv2
import numpy as np
import requests
import tensorflow as tf
from PIL import Image
from keras.models import load_model

from utils import gif2png, fill_image, erosion, resize_to_fit

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:2.0.1) Gecko/2010010' \
                  '1 Firefox/4.0.1',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-us,en;q=0.5',
    'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.7'}

NALOG_URL = 'https://service.nalog.ru/'

MODEL_LABELS_FILEPATH = os.path.join('data', 'model_labels.dat')
MODEL_FILEPATH = os.path.join('data', 'captcha_model.hdf5')

with open(MODEL_LABELS_FILEPATH, "rb") as f:
    lb = pickle.load(f)

model = load_model(MODEL_FILEPATH)

graph = tf.get_default_graph()


def get_captcha_image():
    """
    Получение изображения капчи с сайта nalog.ru
    """
    r = random.randint(0, 2147483647)
    req = requests.get('{0}static/captcha.bin?{1}'.format(NALOG_URL, r))
    a = req.text
    req = requests.get('{0}static/captcha.bin?r={1}&a={2}&version=1'.format(NALOG_URL, r, a))
    pil_captcha_image = Image.open(BytesIO(req.content)).convert('RGB')
    open_cv_image = cv2.cvtColor(np.array(pil_captcha_image), cv2.COLOR_RGB2BGR)
    return a, open_cv_image


def recognize_captcha_image(captcha_image):
    """
    Сегментация и распознавание полученного изображения капчи
    """
    image = gif2png(captcha_image)
    filled_image = fill_image(image)
    thresh = cv2.threshold(filled_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = erosion(thresh)
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = np.asarray(contours[1])
    letter_image_regions = []

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if w / h > 1:
            half_width = int(w / 2)
            if half_width / h > 1:
                fourth_width = int(half_width / 2)
                if fourth_width / h > 1:
                    pass
                else:
                    letter_image_regions.append((x, y, fourth_width, h))
                    letter_image_regions.append((x + fourth_width, y, fourth_width, h))
                    letter_image_regions.append((x + 2 * fourth_width, y, fourth_width, h))
                    letter_image_regions.append((x + 3 * fourth_width, y, fourth_width, h))
            else:
                letter_image_regions.append((x, y, half_width, h))
                letter_image_regions.append((x + half_width, y, half_width, h))
        else:
            letter_image_regions.append((x, y, w, h))

    if len(letter_image_regions) != 6:
        return -1

    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

    predictions = []

    for letter_bounding_box in letter_image_regions:
        x, y, w, h = letter_bounding_box

        letter_image = image[y:y + h, x:x + w]
        letter_image = resize_to_fit(letter_image, 20, 20)
        letter_image = np.expand_dims(letter_image, axis=2)
        letter_image = np.expand_dims(letter_image, axis=0)
        with graph.as_default():
            prediction = model.predict(letter_image)
        letter = lb.inverse_transform(prediction)[0]
        predictions.append(letter)

    captcha_text = "".join(predictions)
    return captcha_text


def show_image(img):
    cv2.imshow('image', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


while True:
    a, captcha_image = get_captcha_image()
    recognized_captcha_text = recognize_captcha_image(captcha_image)
    if recognized_captcha_text != -1:
        show_image(captcha_image)
        print(recognized_captcha_text)