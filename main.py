import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
def draw_hist(src):
    hist_h = 256
    hist_w = 256
    bin_w = int(np.round(hist_w / 256))

    histSize = 256
    histRange = (0, 256)

    hist = cv2.calcHist([src], [0], None, [histSize], histRange)

    cv2.normalize(hist, hist, 0, 230, cv2.NORM_MINMAX)

    hist_img = np.full((hist_h, hist_w), 230, dtype=np.uint8)

    for i in range(1, histSize):
        cv2.rectangle(hist_img,
                      (bin_w * (i - 1), hist_h),
                      (bin_w * i, hist_h - int(np.round(hist[i - 1]))),0,-1)

    return hist_img
def RGBDtoPoint(depth_image, pnt):
    fx = 593.094604
    fy = 593.094604
    cx = 312.916870
    cy = 238.487900

    if depth_image is None:
        print("No depth data!!!")
        exit(1)

    depth_image = depth_image.astype(np.float32)  # Convert the image data to float type

    Z = depth_image[pnt[1], pnt[0]]
    if Z != 0:
        p = np.zeros(3)
        p[2] = Z
        p[0] = (pnt[0] - cx) * Z / fx
        p[1] = (pnt[1] - cy) * Z / fy
    else:
        print("NOT VALID POINT")
        p = None

    return p
def depth_to_3d_points(depth_image):
    height, width = depth_image.shape
    points = []

    for y in range(height):
        for x in range(width):
            point_3d = RGBDtoPoint(depth_image, (x, y))
            if point_3d is not None:
                points.append(point_3d)

    return np.array(points)
def load_raw_depth_map(filename, width, height):
    with open(filename, 'rb') as f:
        raw_data = f.read()

    # Предполагается, что данные хранятся в виде 16-битных значений без знака (uint16)
    depth_map = np.frombuffer(raw_data, dtype=np.uint16).reshape((height, width))

    return depth_map


def mask(src):
    # Create an empty mask with the same dimensions as the source image, single channel (grayscale)
    mask = np.zeros((src.shape[0], src.shape[1]), dtype=np.uint8)

    for x in range(src.shape[0]):
        for y in range(src.shape[1]):
            pixel = src[x, y]

            if pixel[1] < 224:
                value = (-1) * pixel[2] + pixel[0]
            else:
                value = pixel[2] + pixel[0]

            # Ensure the value is within the range (0-255)
            mask[x, y] = np.clip(value, 0, 255)

    return mask
def main():
    # mg = cv2.imread("/Users/dudberoll/PycharmProjects/GaprixCV/data/depth_014_Color.png")
    # img = cv2.morphologyEx(mg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)))
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    # kernel_size = (3, 3)  # Размер ядра (ширина, высота), должны быть нечетными числами
    # sigma = 0.5  # Стандартное отклонение по X и Y
    #
    #
    #
    # gray_img = cv2.GaussianBlur(gray_img, kernel_size, sigma)
    #
    # # Find Canny edges
    # edged = cv2.Canny(gray_img, 30, 200)
    # # Finding Contours
    # # Use a copy of the image e.g. edged.copy()  since findContours alters the image
    # contours, hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow("gaussian blur", gray_img)
    # cv2.imshow('Canny Edges After Contouring', edged)
    # cv2.waitKey(0)
    # print("Number of Contours found = " + str(len(contours)))
    #
    # # Draw all contours
    # # -1 signifies drawing all contours
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    # cv2.imshow('Contours', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # depth_image = cv2.imread("/Users/dudberoll/PycharmProjects/GaprixCV/data/depth_004_Depth.png", cv2.IMREAD_GRAYSCALE)
    # # depth_image = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
    # hist = draw_hist(depth_image)
    # his_max = hist.max()
    # print(his_max)
    # cv2.imshow("hist", hist)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # # Convert depth image to 3D points
    # points = depth_to_3d_points(depth_image)
    # # Plotting the 3D points
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap='viridis', marker='o')
    #
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    #
    # plt.show()


    #
    # # Путь к вашему RAW файлу
    # raw_file = '/Users/dudberoll/PycharmProjects/GaprixCV/data/depth_0002_Depth.raw'
    #
    # # Задайте ширину и высоту карты глубины
    # width = 640
    # height = 480
    #
    # # Загрузите карту глубины
    # depth_map = load_raw_depth_map(raw_file, width, height)
    #
    # # Отобразите карту глубины
    # plt.imshow(depth_map, cmap='gray')
    # plt.colorbar()
    # plt.title('Depth Map')
    # plt.show()

    # src = cv2.imread('/Users/dudberoll/PycharmProjects/GaprixCV/data/depth_004_Depth.png')
    # mask_image = mask(src)
    # cv2.imshow("mask", mask_image)
    # cv2.waitKey(0)

    mg1 = cv2.imread("/Users/dudberoll/PycharmProjects/GaprixCV/data/depth_005_Color.png")
    mg = skimage.io.imread("/Users/dudberoll/PycharmProjects/GaprixCV/data/depth_005_Color.png")
    cv2.imshow(mg1)
    skimage.io.imshow(mg)
if __name__ == "__main__":
    main()