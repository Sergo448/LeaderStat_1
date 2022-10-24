# LeaderStat_1/init/First_OpenAndFindContours.py
import cv2
import numpy as np


def open_and_find_contousr(path):
    """
    :param path:
    :return:

    Разбиение текста на буквы
    Первым шагом разобьем текст на отдельные буквы.
    Для этого пригодится OpenCV, точнее его функция findContours.
    Откроем изображение (cv2.imread), переведем его в ч/б (cv2.cvtColor + cv2.threshold),
    слегка увеличим (cv2.erode) и найдем контуры.
    """

    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)

    # Get contours
    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    output = img.copy()

    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        # print("R", idx, x, y, w, h, cv2.contourArea(contour), hierarchy[0][idx])
        # hierarchy[i][0]: the index of the next contour of the same level
        # hierarchy[i][1]: the index of the previous contour of the same level
        # hierarchy[i][2]: the index of the first child
        # hierarchy[i][3]: the index of the parent
        if hierarchy[0][idx][3] == 0:
            cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)

    cv2.imshow("Input", img)
    cv2.imshow("Enlarged", img_erode)
    cv2.imshow("Output", output)
    cv2.waitKey(0)


# first try
open_and_find_contousr(path='/init/first_hello_world.png')
