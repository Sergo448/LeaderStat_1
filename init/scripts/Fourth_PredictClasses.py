# LeaderStat_1/init/Fourth_PredictClasses.py
import keras
import numpy as np
from Second_LettersExtarctor import letters_extract

"""
Распознавание

Для распознавания мы загружаем модель и вызываем функцию predict_classes.
"""

emnist_labels = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
                 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 99, 100, 101, 102, 103,
                 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122]


def emnist_predict_img(model, img):
    img_arr = np.expand_dims(img, axis=0)
    img_arr = 1 - img_arr / 255.0
    img_arr[0] = np.rot90(img_arr[0], 3)
    img_arr[0] = np.fliplr(img_arr[0])
    img_arr = img_arr.reshape((1, 28, 28, 1))

    predict = model.predict([img_arr])
    result = np.argmax(predict, axis=1)
    return chr(emnist_labels[result[0]])


"""
Как оказалось, изображения в датасете изначально были повернуты,
так что нам приходится повернуть картинку перед распознаванием.

Окончательная функция, которая на входе получает файл с изображением,
а на выходе дает строку, занимает всего 10 строк кода:
"""


def img_to_str(model: any, image_file: str):
    letters = letters_extract(image_file)
    s_out = ""
    for i in range(len(letters)):
        dn = letters[i + 1][0] - letters[i][0] - letters[i][1] if i < len(letters) - 1 else 0
        s_out += emnist_predict_img(model, letters[i][2])
        if dn > letters[i][1] / 4:
            s_out += ' '
    return s_out

"""
Здесь мы используем сохраненную ранее ширину символа, чтобы добавлять пробелы, если промежуток между буквами более 1/4 символа.

Пример использования:
"""

model = keras.models.load_model('emnist_letters.h5')
s_out = img_to_str(model, "hello_world.png")
print(s_out)