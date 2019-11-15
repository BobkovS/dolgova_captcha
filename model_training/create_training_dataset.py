import cv2
import glob
import numpy as np
import os
from distutils.dir_util import copy_tree
from urllib.request import Request, urlopen
from PIL import Image
import requests
from io import BytesIO
import shutil
import argparse

CAPTCHA_IMAGE_FOLDER = "captcha_images"
OUTPUT_FOLDER = "model_training/extracted_letter_images"

headers = {
    'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:2.0.1) Gecko/2010010' \
    '1 Firefox/4.0.1',
    'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language':'en-us,en;q=0.5',
    'Accept-Charset':'ISO-8859-1,utf-8;q=0.7,*;q=0.7'}


def show_image(image):
    """
    Вывод изображения на экран
    """
    cv2.imshow('image', image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def gif2png(path_to_gif):
    """
    Преобразование изображения из GIF формата в PNG
    """
    img = Image.open(path_to_gif)
    img.save('temp.png', 'png', optimize=True, quality=70)
    image = cv2.imread('temp.png', 0)
    image = cv2.resize(image, (200, 100), 0)
    return image


def fill_image(image):
    """
        Заполняем пропуски на изображении
    """
    h, w = image.shape[:2]
    new_image = image.copy()
    for i in range(h):
        for j in range(w):
            if image[i, j] != 255:
                new_image[i, j] = 0
                new_image[i+1, j] = 0
                new_image[i-1, j] = 0
                new_image[i, j+1] = 0
                new_image[i, j-1] = 0
                new_image[i+1, j+1] = 0
                new_image[i+1, j-1] = 0
                new_image[i-1, j+1] = 0
                new_image[i-1, j-1] = 0
    return new_image


def erosion(image):
    """
        Морфологическая эррозия
    """
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    return image


def get_captcha_url():
    """
    Получение случайного изображения капчи с сайта nalog.ru
    """
    req = Request('https://service.nalog.ru/inn.do', None, headers)
    f = urlopen(req)
    page = f.read()
    start_index = str(page).find("captcha.bin")
    end_index = str(page).find("version=1") + 9
    url = 'https://service.nalog.ru/static/' + str(page)[start_index:end_index]
    return url


def load_captcha_images_from_url(count_of_pictures, url):
    """
    Загрузка изображений капчи (count_of_pictures раз). Капча имеет одинаковые ответы, но сформирована по разному
    """
    for i in range(count_of_pictures):
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        img.save('model_training/captcha_images/' + str(i) + '.png')


def get_captcha_correct_text_from_user():
    """
    Получение от пользователя верной расшифровки капчи
    """
    captcha_image_files = glob.glob(os.path.join('model_training/captcha_images', "*"))
    counts = {}
    image = cv2.imread(captcha_image_files[1])
    show_image(image)
    print('Input captcha correct text')
    captcha_correct_text = input()
    return captcha_correct_text, captcha_image_files, counts


def image_segmentation(captcha_image_files, counts, captcha_correct_text):
    """
    Сегментация изображения на отдельные части, запись в соответствующие папки
    """
    for (i, captcha_image_file) in enumerate(captcha_image_files):

        captcha_image_file = gif2png(captcha_image_file)

        image = fill_image(captcha_image_file)

        thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
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
            continue

        letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

        for letter_bounding_box, letter_text in zip(letter_image_regions, captcha_correct_text):

            x, y, w, h = letter_bounding_box

            letter_image = captcha_image_file[y - 2:y + h + 2, x - 2:x + w + 2]
            save_path = os.path.join(OUTPUT_FOLDER, letter_text)

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            count = counts.get(letter_text, 1)
            p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
            cv2.imwrite(p, letter_image)
            counts[letter_text] = count + 1


def adding_current_dataset_to_the_total_dataset():
    """
    Добавление выборки текущего цикла программы к общему датасету
    """
    os.makedirs('model_training/sets/' + captcha_correct_text)
    if not os.path.exists('model_training/digits'):
        os.makedirs('model_training/digits')
    for filename in os.listdir('model_training/extracted_letter_images'):
        os.rename('model_training/extracted_letter_images/' + filename, 'model_training/sets/' + captcha_correct_text + '/' + filename)
    copy_tree("model_training/sets/" + captcha_correct_text, "model_training/sets_temp/" + captcha_correct_text)

    for setname in os.listdir('model_training/sets'):
        for digitname in os.listdir('model_training/sets/' + setname):
            for file in os.listdir('model_training/sets/' + setname + '/' + digitname):
                os.rename('model_training/sets/' + setname + '/' + digitname + '/' + file, 'model_training/digits/' + digitname + '/' + str(
                    len([name for name in os.listdir('model_training/digits/' + digitname)])) + '.png')
    shutil.rmtree("model_training/sets/" + captcha_correct_text)


def parse_args():
    parser = argparse.ArgumentParser(prog='create_training_dataset.py')
    parser.add_argument("-u", "--unique",
                        required=True,
                        type=str,
                        help="Count of unique captcha images")
    parser.add_argument("-c", "--repeated",
                        required=True,
                        type=str,
                        help="Count of repeated captcha images")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    usable_args = parse_args()
    for i in range(int(usable_args.unique)):
        url = get_captcha_url()
        load_captcha_images_from_url(int(usable_args.repeated), url)
        captcha_correct_text, captcha_image_files, counts = get_captcha_correct_text_from_user()
        image_segmentation(captcha_image_files, counts, captcha_correct_text)
        adding_current_dataset_to_the_total_dataset()
