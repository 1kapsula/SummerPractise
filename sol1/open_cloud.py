import numpy as np
import cv2 as cv
import open3d as o3d

# Параметры камеры
fx = 615.0  # Фокусное расстояние по оси x
fy = 615.0  # Фокусное расстояние по оси y
cx = 320.0  # Координата главной точки по оси x
cy = 240.0  # Координата главной точки по оси y

def depth_to_point_cloud(depth, fx, fy, cx, cy):
    h, w = depth.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    z = depth / 1000.0

    x = (x - cx) * z / fx
    y = (y - cy) * z / fy

    points = np.stack((x, y, z), axis=-1)
    return points.reshape(-1, 3)

def find_edges_3d(image, depth):
    # Создаем облако точек
    points = depth_to_point_cloud(depth, fx, fy, cx, cy)
    colors = cv.cvtColor(image, cv.COLOR_BGR2RGB).reshape(-1, 3) / 255.0

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)

    # Применяем кластеризацию для выделения объектов
    labels = np.array(cloud.cluster_dbscan(eps=0.05, min_points=10))

    max_label = labels.max()
    objects = []
    for i in range(max_label + 1):
        mask = labels == i
        obj_points = points[mask]
        if len(obj_points) < 4:  # Проверяем, что точек достаточно для вычисления оболочки
            continue
        obj_cloud = o3d.geometry.PointCloud()
        obj_cloud.points = o3d.utility.Vector3dVector(obj_points)
        objects.append(obj_cloud)

    edges = []
    for obj in objects:
        if len(obj.points) < 4:
            continue
        try:
            hull, _ = obj.compute_convex_hull()
            hull_edges = np.asarray(hull.triangles)
            hull_points = np.asarray(hull.vertices)

            for edge in hull_edges:
                pt1 = hull_points[edge[0]]
                pt2 = hull_points[edge[1]]
                pt3 = hull_points[edge[2]]
                edges.append((pt1, pt2))
                edges.append((pt2, pt3))
                edges.append((pt3, pt1))
        except RuntimeError as e:
            print(f"Error computing convex hull: {e}")

    return edges

def project_to_image(points, fx, fy, cx, cy):
    points = np.array(points)
    u = (fx * points[:, 0] / points[:, 2] + cx).astype(np.int)
    v = (fy * points[:, 1] / points[:, 2] + cy).astype(np.int)
    return u, v

def main():
    image = cv.imread('good_data/depth_004_Color.png')
    depth = np.fromfile('raw/depth_004_Depth.raw', np.uint16).reshape((480, 640))

    edges = find_edges_3d(image, depth)

    for edge in edges:
        pt1, pt2 = edge
        u1, v1 = project_to_image([pt1], fx, fy, cx, cy)
        u2, v2 = project_to_image([pt2], fx, fy, cx, cy)
        cv.line(image, (u1[0], v1[0]), (u2[0], v2[0]), (0, 0, 255), 2)

    cv.imwrite('result.png', image)
    cv.imshow('Edges', image)
    cv.waitKey()

if __name__ == '__main__':
    main()
    cv.destroyAllWindows()
