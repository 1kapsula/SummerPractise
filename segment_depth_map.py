import numpy as np
import matplotlib.pyplot as plt
import skimage.io

# Параметры карты глубины
width = 640  # ширина изображения
height = 480  # высота изображения
dtype = np.uint16  # тип данных

# Чтение карты глубины из raw файла
raw_file_path = "/Users/dudberoll/PycharmProjects/GaprixCV/data/depth_005_Depth.raw"
depth_map = np.fromfile(raw_file_path, dtype=dtype).reshape((height, width))

# Определяем диапазон значений для сегментации
min_value = 1100
max_value = 1285

# Создаем маску для значений в заданном диапазоне
mask = (depth_map >= min_value) & (depth_map <= max_value)

# Применяем маску к карте глубины
depth_map_filtered = np.where(mask, depth_map, 0)

# Отображение результата
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(depth_map, cmap='gray')
plt.title('Оригинальная карта глубины')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(depth_map_filtered, cmap='gray')
plt.title('Фильтрованная карта глубины')
plt.axis('off')

plt.show()

# Сохранение фильтрованной карты глубины
filtered_file_path = "/Users/dudberoll/PycharmProjects/GaprixCV/depth_map_filtered.png"
skimage.io.imsave(filtered_file_path, depth_map_filtered.astype(np.uint16))
