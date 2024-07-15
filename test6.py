import cv2
import numpy as np

# Загрузка изображений
rgb_image_path = "/Users/dudberoll/PycharmProjects/GaprixCV/data/depth_005_Color.png"
depth_image_path = "/Users/dudberoll/PycharmProjects/GaprixCV/depth_map_filtered.png"
rgb_image = cv2.imread(rgb_image_path)
# rgb_image = cv2.cvtColor(rgb_imag, cv2.COLOR_BGR2HSV)
depth_image = cv2.imread(depth_image_path)
rgb_image = cv2.morphologyEx(rgb_image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

# Изменение размера RGB изображения до размеров карты глубины
depth_height, depth_width = depth_image.shape[:2]
rgb_image_resized = cv2.resize(rgb_image, (depth_width, depth_height))

# Создание копии RGB изображения
rgb_image_copy = rgb_image_resized.copy()

# Преобразование карты глубины в одноканальное изображение
depth_map = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)

# Применение фильтра Кэнни для выделения границ
edges_rgb = cv2.Canny(cv2.cvtColor(rgb_image_resized, cv2.COLOR_BGR2GRAY), 50, 150)
edges_depth = cv2.Canny(depth_map, 50, 150)

# Нахождение контуров
contours_rgb, _ = cv2.findContours(edges_rgb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_depth, _ = cv2.findContours(edges_depth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# Использование SUSAN для нахождения углов
gray_rgb = cv2.cvtColor(rgb_image_resized, cv2.COLOR_BGR2GRAY)
corners_susan = cv2.goodFeaturesToTrack(gray_rgb, maxCorners=100, qualityLevel=0.1, minDistance=10)

# Отображение углов SUSAN
if corners_susan is not None:
    for corner in corners_susan:
        x, y = corner.ravel()
        cv2.circle(rgb_image_copy, (int(x), int(y)), 5, (0, 255, 0), -1)  # Зеленые круги для углов

# Применение детектора Харриса для нахождения углов
#gray_rgb = cv2.cvtColor(rgb_image_resized, cv2.COLOR_BGR2GRAY)
#gray_rgb = np.float32(gray_rgb)  # Преобразование в float32
#corners_harris = cv2.cornerHarris(gray_rgb, blockSize=2, ksize=3, k=0.04)

# Увеличение углов для визуализации
#corners_harris = cv2.dilate(corners_harris, None)

# Сопоставление контуров
matched_contours = []
for contour_rgb in contours_rgb:
    for contour_depth in contours_depth:
        similarity = cv2.matchShapes(contour_rgb, contour_depth, cv2.CONTOURS_MATCH_I1, 0.0)
        if similarity < 0.3:  # Порог для определения совпадения
            matched_contours.append(contour_rgb)
            break

# Отображение совпавших контуров на копии RGB изображения
#for contour in matched_contours:
#    cv2.drawContours(rgb_image_copy, [contour], -1, (255, 0, 0), 2)  # Красный для совпавших контуров

# Установка порога для углов
#threshold = 0.01 * corners_harris.max()
#rgb_image_copy[corners_harris > threshold] = [0, 255, 0]  # Отображение углов зеленым

# Отображение изображений
cv2.imshow('RGB Image with Matched Contours', rgb_image_copy)
cv2.imshow('Edges on Depth Map', edges_depth)
cv2.imshow('Edges on RGB Image', edges_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
