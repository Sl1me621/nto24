import numpy as np
import cv2


def one_hot_encode(label):

    """ Функция осуществляет перекодировку текстового "названия" сигнала
     в список элементов, соответствующий выходному сигналу

     Входные параметры: текстовая метка
     Выходные параметры: метка ввиде списка

     Пример:
        one_hot_encode("red") должно возвращать:        [1, 0, 0, 0, 0]
        one_hot_encode("yellow") должно возвращать:     [0, 1, 0, 0, 0]
        one_hot_encode("green") должно возвращать:      [0, 0, 1, 0, 0]
        one_hot_encode("yellow_red") должно возвращать: [0, 0, 0, 1, 0]
        one_hot_encode("off") должно возвращать:        [0, 0, 0, 0, 1]

     """
    one_hot_encoded = []

    if label == "red":
        one_hot_encoded = [1, 0, 0, 0, 0]
    elif label == "yellow":
        one_hot_encoded = [0, 1, 0, 0, 0]
    elif label == "green":
        one_hot_encoded = [0, 0, 1, 0, 0]
    elif label == "yellow_red":
        one_hot_encoded = [0, 0, 0, 1, 0]
    elif label == "off":
        one_hot_encoded = [0, 0, 0, 0, 1]

    return one_hot_encoded


def standardize_input(image):
    """Приведение изображений к стандартному виду. 
    Входные данные: изображение (bgr)
    Выходные данные: стандартизированное изображений.
    """
    standard_im = image  # по умолчанию, функция не меняет изображения

    ## TODO: Если вы хотите преобразовать изображение в формат, одинаковый для всех изображений, сделайте это здесь.
    return standard_im


def predict_label(rgb_image):
    """
     функция определения сигнала светофора по входному изображению

     Входные данные: изображение (bgr)
     Выходные данные: метка в формате списка (смотри one_hot_encode)

    """

    predicted_label = "yellow"
    encoded_label = one_hot_encode(predicted_label)  # по умолчанию, говорит что на всех изображения жёлтый сигнал

    ## TODO: ваша функция распознавания сигнала светофора должна быть здесь.
    return encoded_label
