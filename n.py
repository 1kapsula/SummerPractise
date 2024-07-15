import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.color import rgb2gray
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from scipy.spatial import ConvexHull

# Загрузка изображения
image_path = "/Users/dudberoll/PycharmProjects/GaprixCV/depth_005_gray_morphology.png"
gray_image = skimage.io.imread(image_path)
image = skimage.io.imread(image_path)

# Нахождение углов с использованием метода Харриса
harris_response = corner_harris(gray_image)
corners = corner_peaks(harris_response, min_distance=5, threshold_rel=0.02)

# Уточнение координат углов с помощью subpix
corners_subpix = corner_subpix(gray_image, corners, window_size=13)

# Подготовка координат для закраски
if corners_subpix is not None:
    x = corners_subpix[:, 1]
    y = corners_subpix[:, 0]
else:
    x = corners[:, 1]
    y = corners[:, 0]

# Удаление NaN значений из координат
valid_indices = ~np.isnan(x) & ~np.isnan(y)
x = x[valid_indices]
y = y[valid_indices]

# Сортировка точек, если необходимо (например, по углу относительно центра)
center = np.mean(np.vstack((x, y)).T, axis=0)
angles = np.arctan2(y - center[0], x - center[1])
sort_order = np.argsort(angles)
x_sorted = x[sort_order]
y_sorted = y[sort_order]

# Вычисление площади контура (contour area)
points = np.vstack((x_sorted, y_sorted)).T
hull = ConvexHull(points)
contour_area = hull.volume  # Для 2D ConvexHull, volume дает площадь

print(f"Contour Area: {contour_area}")

# Отображение изображения с закрашенными углами
plt.figure(figsize=(10, 8))
plt.imshow(image, cmap='gray')
plt.plot(x_sorted, y_sorted, 'r+', markersize=10)
plt.fill(x_sorted, y_sorted, 'b', alpha=0.3)  # Закраска многоугольника
plt.title('Изображение с закрашенными углами')
plt.axis('off')
plt.show()
