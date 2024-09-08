# -*- coding: utf-8 -*-
import eval
import cv2
import pandas as pd

"""Файл служит для определения точности вашего алгоритма
   Не редактируёте его!!!
   Для получения оценки точности, запустите файл на исполнение
"""

def extract_obect_list(row):
    object_list = []
    label = row[2]
    x = int(row[3])
    y = int(row[4])
    x2 = int(row[5])
    y2 = int(row[6])
    object_list.append([label, (x, y, x2, y2)])
    return object_list


def inspect(user_obj_list, true_obj_list):
    hit = 0
    true_label = true_obj_list[0][0]
    for object in user_obj_list:
        if object[0] == true_label:
            if IoU(object[1], true_obj_list[0][1]) > 0.6:
                hit = 1
                break
    miss = len(user_obj_list) - hit
    return miss, hit

def IoU(user_box,true_box):
    """IoU = Area of overlap / Area of union
       Output: 0.0 .. 1.0
       Не важно в каком порядке передаются рамки, IoU не изменится.
    """

    xA = max(user_box[0], true_box[0])
    yA = max(user_box[1], true_box[1])
    xB = min(user_box[2], true_box[2])
    yB = min(user_box[3], true_box[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (user_box[2] - user_box[0] + 1) * (user_box[3] - user_box[1] + 1)
    boxBArea = (true_box[2] - true_box[0] + 1) * (true_box[3] - true_box[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def main():
    model = eval.load_final_model()
    csv_file = "annotations.csv"
    data = pd.read_csv(csv_file, sep=',')
    data = data.sample(frac=1)
    all_miss_detection = 0
    all_good_detection = 0
    for row in data.itertuples():
        image = cv2.imread(row[1])
        user_obj_list = eval.predict_label_and_box(image, model)
        if user_obj_list == []:
            continue
        true_obj_list = extract_obect_list(row)
        # перебираем пользовательские метки, пока не найдём верную.
        # верной засчитается первая в списке с верной меткой и достаточным IOU.
        miss_detection, good_detection = inspect(user_obj_list, true_obj_list)
        all_miss_detection += miss_detection
        all_good_detection += good_detection

    total_object = len(data.index)
    print("Из " + str(total_object) + " объектов, верно детектированы и распознаны " + str(all_good_detection))
    print("При этом, совершено " + str(all_miss_detection) + " ошибочных детектирований")
    if (all_good_detection - all_miss_detection) <= 0:
        score = 0
        print("Алгоритм чаще ошибался, чем выдавал верные ответы")
    else:
        score = (all_good_detection - all_miss_detection) / total_object
    print("Точность: " + str(score))


if __name__ == '__main__':
    main()
