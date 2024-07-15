import cv2
import numpy as np


def load_depth_map(raw_file_path, width, height):
    depth_map = np.fromfile(raw_file_path, dtype=np.uint16)
    depth_map = depth_map.reshape((height, width))
    return depth_map


def normalize_depth_value(value, max_value=65535, new_max_value=256):
    return int((value / max_value) * new_max_value)


def normalize_depth_map(depth_map, max_value=65535, new_max_value=256):
    return (depth_map / max_value) * new_max_value


def create_mask(depth_map, threshold):
    mask = depth_map > threshold
    return mask.astype(np.uint8) * 255  # Маска должна быть в диапазоне 0-255 для корректного применения


def apply_mask(image, mask):
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        depth_map, threshold = param
        depth_value = depth_map[y, x]
        # normalized_value = normalize_depth_value(depth_value)
        print(f"Depth value at ({x}, {y}): {depth_value}")


def main(raw_file_path, img_path, width, height, threshold):
    depth_map = load_depth_map(raw_file_path, width, height)
    original_image = cv2.imread(img_path)
    depth_map_normalized = normalize_depth_map(depth_map)

    mask = create_mask(depth_map, threshold)

    depth_map_display = cv2.normalize(depth_map_normalized, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    masked_image = apply_mask(depth_map_display, mask)
    original_image= cv2.bitwise_and(original_image, original_image, mask=mask)
    cv2.imshow('Depth Map', depth_map_display)
    cv2.imshow('Masked Image', masked_image)
    cv2.imwrite('Masked_Image.png', masked_image)
    cv2.setMouseCallback('Depth Map', mouse_callback, [depth_map, threshold])
    cv2.imwrite('original.png', original_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    raw_file_path = '/Users/dudberoll/PycharmProjects/GaprixCV/data/depth_005_Depth.raw'
    img_path = '/Users/dudberoll/PycharmProjects/GaprixCV/data/depth_005_Color.png'
    width = 640
    height = 480
    threshold = 1287 # Укажите порог (например, половина диапазона 16-битной глубины)
    main(raw_file_path,img_path, width, height, threshold)

    cropped_image = cv2.imread(img_path)[76:434, 454:36]
    cv2.imwrite('depth_005_Color_cropped.png', cropped_image)
