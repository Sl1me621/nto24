import helpers
import cv2
import random
import numpy as np

def get_standatr_signs():
    """Функция, позволяющая получить стандартные изображения знаков для сравнения
    с оригинальными изображениями. Стандартные изображения знаков хранятся во внутреннем каталоге."""
    # стандартные изображения дорожных знаков
    a_unevenness = cv2.imread("data/standards/a_unevenness.jpg")
    a_unevenness = cv2.inRange(a_unevenness, (89, 91, 149), (255, 255, 255))
    a_unevenness = cv2.resize(a_unevenness, (64, 64))

    no_drive = cv2.imread("data/standards/no_drive.png")
    no_drive = cv2.inRange(no_drive, (89, 91, 149), (255, 255, 255))
    no_drive = cv2.resize(no_drive, (64, 64))

    no_entry = cv2.imread("data/standards/no_entry.jpg")
    no_entry = cv2.inRange(no_entry, (89, 91, 149), (255, 255, 255))
    no_entry = cv2.resize(no_entry, (64, 64))

    parking = cv2.imread("data/standards/parking.jpg")
    parking = cv2.inRange(parking, (0, 0, 0), (255, 0, 255))
    parking = cv2.resize(parking, (64, 64))

    pedistrain = cv2.imread("data/standards/pedistrain.jpg")
    pedistrain = cv2.inRange(pedistrain, (89, 91, 149), (255, 255, 255))
    pedistrain = cv2.resize(pedistrain, (64, 64))


    road_works = cv2.imread("data/standards/road_works.jpg")
    road_works = cv2.inRange(road_works, (89, 91, 149), (255, 255, 255))
    road_works = cv2.resize(road_works, (64, 64))

    stop = cv2.imread("data/standards/stop.jpg")
    stop = cv2.inRange(stop, (89, 91, 149), (255, 255, 255))
    stop = cv2.resize(stop, (64, 64))

    way_out = cv2.imread("data/standards/way_out.jpg")
    way_out = cv2.inRange(way_out, (89, 91, 149), (255, 255, 255))
    way_out = cv2.resize(way_out, (64, 64))

    standart_signs = {
        "a_unevenness": a_unevenness,
        "no_drive": no_drive,
        "no_entry": no_entry,
        "parking": parking,
        "pedistrain": pedistrain,
        "road_works": road_works,
        "stop": stop,
        "way_out": way_out
    }
    return standart_signs

def one_hot_encode(label):

    one_hot_encoded = []
    if label == "none":
        one_hot_encoded = [0, 0, 0, 0, 0, 0, 0, 0]
    elif label == "pedistrain":
        one_hot_encoded = [1, 0, 0, 0, 0, 0, 0, 0]
    elif label == "no_drive":
        one_hot_encoded = [0, 1, 0, 0, 0, 0, 0, 0]
    elif label == "stop":
        one_hot_encoded = [0, 0, 1, 0, 0, 0, 0, 0]
    elif label == "way_out":
        one_hot_encoded = [0, 0, 0, 1, 0, 0, 0, 0]
    elif label == "no_entry":
        one_hot_encoded = [0, 0, 0, 0, 1, 0, 0, 0]
    elif label == "road_works":
        one_hot_encoded = [0, 0, 0, 0, 0, 1, 0, 0]
    elif label == "parking":
        one_hot_encoded = [0, 0, 0, 0, 0, 0, 1, 0]
    elif label == "a_unevenness":
        one_hot_encoded = [0, 0, 0, 0, 0, 0, 0, 1]

    return one_hot_encoded

def predict_label(image):

    standart_signs = get_standatr_signs()
    a_unevenness = standart_signs["a_unevenness"]
    no_drive = standart_signs["no_drive"]
    no_entry = standart_signs["no_entry"]
    parking = standart_signs["parking"]
    pedistrain = standart_signs["pedistrain"]
    road_works = standart_signs["road_works"]
    stop = standart_signs["stop"]
    way_out = standart_signs["way_out"]

    predicted_label = [0, 0, 0, 0, 0, 0, 0, 1]

    image = cv2.resize(image, (64, 64))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)

    a_unevenness_val = 0
    no_drive_val = 0
    no_entry_val = 0
    none_val = 0
    parking_val = 0
    pedistrain_val = 0
    road_works_val = 0
    stop_val = 0
    way_out_val = 0

    # cv2.imshow("rgb_image", gray_image)
    # cv2.waitKey(0)

    for i in range(64):
        for j in range(64):
            if image[i][j] == a_unevenness[i][j]:
                a_unevenness_val += 1
            elif image[i][j] == no_drive[i][j]:
                no_drive_val += 1
            elif image[i][j] == no_entry[i][j]:
                no_entry_val += 1
            elif image[i][j] == parking[i][j]:
                parking_val += 1
            elif image[i][j] == pedistrain[i][j]:
                pedistrain_val += 1
            elif image[i][j] == road_works[i][j]:
                road_works_val += 1
            elif image[i][j] == stop[i][j]:
                stop_val += 1
            elif image[i][j] == way_out[i][j]:
                way_out_val += 1
            else:
                none_val += 1

    if a_unevenness_val > 3000:
        predicted_label = one_hot_encode("a_unevenness")
    elif no_drive_val > 3000:
        predicted_label = one_hot_encode("no_drive")
    elif no_entry_val > 3000:
        predicted_label = one_hot_encode("no_entry")
    elif parking_val > 3000:
        predicted_label = one_hot_encode("parking")
    elif pedistrain_val > 3000:
        predicted_label = one_hot_encode("pedistrain")
    elif road_works_val > 3000:
        predicted_label = one_hot_encode("road_works")
    elif stop_val > 3000:
        predicted_label = one_hot_encode("stop")
    elif way_out_val > 3000:
        predicted_label = one_hot_encode("way_out")
    else:
        predicted_label = one_hot_encode("none")

    return predicted_label