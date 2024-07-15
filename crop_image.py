import cv2
import numpy as np
# img_path = '/Users/dudberoll/PycharmProjects/GaprixCV/data/depth_005_Color.png'
# img = cv2.imread(img_path)
# cropped_image = img[33:421, 100:540]
point = (640, 480)
# croppe_image = cv2.resize(cropped_image, point)
# cv2.imwrite('depth_005_Color_cropped.png', croppe_image)


mask_image = cv2.imread("/Users/dudberoll/PycharmProjects/GaprixCV/Masked_Image.png", cv2.COLOR_RGB2GRAY)
cropped_image = mask_image[33:421, 100:540]
crop_Mask = cv2.resize(cropped_image, point)
# crop_Mask_8bit = cv2.convertScaleAbs(mask_image, alpha=(65535.0 / 255.0))
# crop_Mask_8bit = crop_Mask_8bit[33:421, 100:540].astype(np.uint8)
# gray_image = cv2.cvtColor(crop_Mask_8bit, cv2.COLOR_BGR2GRAY)
cv2.imwrite('depth_005_Color_Mask_cropped.png', crop_Mask)

ggg = cv2.imread("depth_005_Color_Mask_cropped.png", cv2.IMREAD_GRAYSCALE)
_, binary_mask = cv2.threshold(ggg, 128, 255, cv2.THRESH_BINARY)

# Сохранение бинаризованной маски
cv2.imwrite('depth_005_Color_Mask_cropped.png', binary_mask)

ggg = cv2.imread("depth_005_Color_Mask_cropped.png")
print(ggg.dtype)
# import os
# import cv2
# import glob
#
# # Путь к папке с изображениями
# folder_path = '/Users/dudberoll/PycharmProjects/GaprixCV/good_data'
#
# # Путь к папке, куда сохранять обрезанные изображения
# output_folder = '/Users/dudberoll/PycharmProjects/GaprixCV/good_data_cropped'
#
# # Создаем выходную папку, если её еще нет
# os.makedirs(output_folder, exist_ok=True)
#
# # Шаблон имени файлов, которые нужно обработать (например, все PNG-файлы)
# file_pattern = '*.png'
#
# # Получаем список файлов, соответствующих шаблону
# files = glob.glob(os.path.join(folder_path, file_pattern))
#
# # Размер обрезки
# top_left = (70, 33)
# bottom_right = (560, 440)
#
# # Размер для изменения размера
# resize_point = (640, 480)
#
# # Итерируемся по найденным файлам
# for file_path in files:
#     # Загружаем изображение
#     img = cv2.imread(file_path)
#
#     # Обрезаем изображение
#     cropped_image = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
#
#     # Изменяем размер
#     resized_image = cv2.resize(cropped_image, resize_point)
#
#     # Получаем имя файла без пути и расширения
#     file_name = os.path.splitext(os.path.basename(file_path))[0]
#
#     # Путь для сохранения обрезанного изображения
#     output_path = os.path.join(output_folder, f'{file_name}_cropped.png')
#
#     # Сохраняем изображение
#     cv2.imwrite(output_path, resized_image)


