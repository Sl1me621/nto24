import helpers
import cv2
import random
import numpy as np
## TODO: Допишите импорт библиотек, которые собираетесь использовать


def get_standatr_signs():
    """Функция, позволяющая получить эталонные изображения знаков для сравнения
       с изображениями для классификации.
       Стандартные изображения знаков хранятся во внутреннем каталоге.

       Если вы не собираетесь использовать эту функцию, пусть возвращает пустой список []
       """

    ## TODO: Отредактируйте функцию по своему усмотрению.
    ## Эталонные изображения знаков, вы можете загрузить вместе с решением
    ## Укажите правильные названия файлов со знаками и пути к ним
    ## Если вы не собираетесь использовать эту функцию, пусть возвращает пустой список []

    standart_signs_list = []

    image = cv2.imread("standards/no entry.jpg")
    image = cv2.inRange(image, (89, 91, 149), (255, 255, 255))
    image = cv2.resize(image, (64, 64))
    standart_signs_list.append(image)

    image = cv2.imread("standards/pedestrian crossing.jpg")
    image = cv2.inRange(image, (89, 91, 149), (255, 255, 255))
    image = cv2.resize(image, (64, 64))
    standart_signs_list.append(image)

    image = cv2.imread("standards/road works.jpg")
    image = cv2.inRange(image, (89, 91, 149), (255, 255, 255))
    image = cv2.resize(image, (64, 64))
    standart_signs_list.append(image)

    image = cv2.imread("standards/movement prohibition.png")
    image = cv2.inRange(image, (0, 0, 0), (255, 0, 255))
    image = cv2.resize(image, (64, 64))
    standart_signs_list.append(image)

    image = cv2.imread("standards/parking.jpg")
    image = cv2.inRange(image, (89, 91, 149), (255, 255, 255))
    image = cv2.resize(image, (64, 64))
    standart_signs_list.append(image)

    image = cv2.imread("standards/stop.jpg")
    image = cv2.inRange(image, (89, 91, 149), (255, 255, 255))
    image = cv2.resize(image, (64, 64))
    standart_signs_list.append(image)

    image = cv2.imread("standards/give way.jpg")
    image = cv2.inRange(image, (89, 91, 149), (255, 255, 255))
    image = cv2.resize(image, (64, 64))
    standart_signs_list.append(image)

    image = cv2.imread("standards/artificial roughness.jpg")
    image = cv2.inRange(image, (89, 91, 149), (255, 255, 255))
    image = cv2.resize(image, (64, 64))
    standart_signs_list.append(image)

    return standart_signs_list


def load_final_model():
    """ Функция осуществляет загрузку модели нейронной сети из файла.
        Выходные параметры: загруженная модель

        Если вы не собираетесь использовать эту функцию, пусть возвращает пустой список []
    """

    ## TODO: Отредактируйте функцию по своему усмотрению.
    ## Модель нейронной сети, загрузите вместе с решением.
    ## Если вы не собираетесь использовать эту функцию, пусть возвращает пустой список []

    # model = tensorflow.load_model(MODEL_FILE_NAME)
    model = []
    return model


def one_hot_encode(label):

    """ Функция осуществляет перекодировку текстового "названия" сигнала
     в список элементов, соответствующий выходному сигналу

     Входные параметры: текстовая метка
     Выходные параметры: метка ввиде списка

     Пример:
        one_hot_encode("no entry") должно возвращать:              [1, 0, 0, 0, 0, 0, 0, 0]
        one_hot_encode("parking") должно возвращать:               [0, 0, 0, 0, 1, 0, 0, 0]
        one_hot_encode("artificial roughness") должно возвращать:  [0, 0, 0, 0, 0, 0, 0, 1]
    """

    one_hot_label_dictionary = {"no entry": [1, 0, 0, 0, 0, 0, 0, 0],
                                "pedestrian crossing": [0, 1, 0, 0, 0, 0, 0, 0],
                                "road works": [0, 0, 1, 0, 0, 0, 0, 0],
                                "movement prohibition": [0, 0, 0, 1, 0, 0, 0, 0],
                                "parking": [0, 0, 0, 0, 1, 0, 0, 0],
                                "stop": [0, 0, 0, 0, 0, 1, 0, 0],
                                "give way": [0, 0, 0, 0, 0, 0, 1, 0],
                                "artificial roughness": [0, 0, 0, 0, 0, 0, 0, 1]}

    ## TODO: Отредактируйте функцию по своему усмотрению.
    one_hot_encoded = one_hot_label_dictionary[label]
    return one_hot_encoded


def standardize_input(image):
    """Приведение изображений к стандартному виду.
    Входные данные: изображение (bgr); прочитаны cv2.imread()
    Выходные данные: стандартизированное изображений.
    """
    standard_im = image  # по умолчанию, функция не меняет изображения
    ## TODO: Если вы хотите преобразовать изображение в формат,
    ## одинаковый для всех изображений, сделайте это здесь.

    return standard_im


def predict_label(image, model, standart_signs):
    """
         Функция, определяющая какой знак на изображении
         Входные данные: изображение (bgr), модель нейронной сети, эталонные изображения знаков
         Выходные данные: метка в формате списка (смотри one_hot_encode)
    """

    ## TODO: Отредактируйте эту функцию по своему усмотрению.
    # Вы можете пользоваться и нейросетевыми подходами,
    # контурным анализом с бинаризацией и сравнениями с шаблонами
    # НЕОБЯЗАТЕЛЬНО использовать всем функции представленные в этом файле
    # Алгоритм проверки будет вызывать функции get_standatr_signs, load_final_model и predict_label,
    # остальные функции должны вызываться из вешеперечисленных.

    standard_im = standardize_input(image)
    predicted_label = "stop"  # какое-то сравнение с шаблоном и прочие преобразования, на выходе дающие label
    encoded_label = one_hot_encode(predicted_label)

    # если используете нейросеть
    # predicted_label = model.predict(standard_im) # если вы используете нейронную сеть
    # encoded_label = one_hot_encode(predicted_label) # приведение формата ответа нейросети к требуемуму

    return encoded_label