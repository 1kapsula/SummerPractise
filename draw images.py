import os
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np

# Путь к JSON файлу с разметкой
json_path = '/Users/dudberoll/PycharmProjects/GaprixCV/metData.json'

# Путь к папке с изображениями
image_folder_path = '/Users/dudberoll/PycharmProjects/GaprixCV/good_data'


# Функция для отображения изображения с разметкой
def show_image_with_annotations(image_path, annotations):
    # Загружаем изображение
    image = cv2.imread(image_path)
    # Преобразуем BGR изображение в RGB для корректного отображения в matplotlib
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Проходимся по каждому региону в разметке
    for region in annotations:
        points_x = region['shape_attributes']['all_points_x']
        points_y = region['shape_attributes']['all_points_y']
        # Формируем список точек для полигона
        points = [(x, y) for x, y in zip(points_x, points_y)]
        # Преобразуем в формат numpy
        points = np.array(points, np.int32)
        points = points.reshape((-1, 1, 2))
        # Рисуем полигон
        cv2.polylines(image, [points], isClosed=True, color=(255, 0, 0), thickness=2)

    # Отображаем изображение
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()


# Читаем JSON файл с разметкой
with open(json_path, 'r') as f:
    data = json.load(f)

# Проходимся по каждому файлу в разметке
for key, value in data.items():
    filename = value['filename']
    regions = value['regions']

    image_path = os.path.join(image_folder_path, filename)
    if os.path.exists(image_path):
        show_image_with_annotations(image_path, regions)
    else:
        print(f"Файл {image_path} не найден.")
