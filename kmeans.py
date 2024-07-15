import cv2
import numpy as np
import mouse_click as mc



# Загрузка изображения
# image = cv2.imread('/Users/dudberoll/PycharmProjects/GaprixCV/data/depth_005_Depth.png')
image = mc.load_depth_map('/Users/dudberoll/PycharmProjects/GaprixCV/data/depth_005_Depth.raw', 640, 480)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Переключаем цветовое пространство на RGB

# Подготовка данных для кластеризации
pixel_values = image.reshape((-1, 3))  # Преобразование в одномерный массив пикселей

# Кластеризация методом k-means
k = 15  # Число кластеров (можно выбрать другое значение)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
_, labels, centers = cv2.kmeans(pixel_values.astype(np.float32), k, None, criteria, 15, cv2.KMEANS_RANDOM_CENTERS)

# Преобразование меток в 8-битное изображение
centers = np.uint8(centers)

segmented_image = centers[labels.flatten()]

# Восстановление изображения в исходные размеры
segmented_image = segmented_image.reshape(image.shape)
cv2.imwrite('segmented_image_005_kmeans.png', segmented_image)
# Вывод и отображение результатов
cv2.imshow('Segmented Image', segmented_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
