
## Trash
https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html

https://www.geeksforgeeks.org/find-and-draw-contours-using-opencv-python/ - выделение контуров

- [ ] карта глубины, расстояние от камеры до пола
- [ ] выделение границ
- [ ] построение модели с помощью карты глубины и границ (отдаленность пикселя и расположение границы соответсвенно)

Как построить 3d модель. Можно пробегаться по уровням карты глубины, например с диапазоном в 2-3 значения, и находить точки с максимальным расстоянием друг от друга. 

Посмотреть лекции полевого !

Надо удалить линию с фона коробки. Можно присваивать черным пикселям значение в их окрестности.  

https://habr.com/ru/articles/353890/ -  Простой фильтр для автоматического удаления фона с изображений


 надо написать функцию, которая будет похожа чем-то на CNN. 

https://www.youtube.com/watch?v=5ypQIUbpA7c

https://www.youtube.com/watch?v=-OSVKbSsqT0

https://github.com/isl-org/Open3D-ML

https://www.youtube.com/watch?v=EpjXp3DbJuQ

https://answers.opencv.org/question/222433/get-3d-point-from-clicked-depth-map-pixel/

https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/opencv_pointcloud_viewer.py

https://arxiv.org/abs/1612.01601 - superpixels

https://github.com/davidstutz/superpixel-benchmark/blob/master/docs/ALGORITHMS.md - algos cv
https://www.epfl.ch/labs/ivrl/research/slic-superpixels/#:~:text=We%20introduce%20a%20novel%20algorithm,generate%20compact%2C%20nearly%20uniform%20superpixels. - slic superpixels
TO DO:
- [ ] написать функцию которая будет сравнивать где находится граница углов на картинке с суперпиеселями. 
- [ ] посмотреть на карту глубины и там выделить границы. потом посмотреть на расстояние между точками которые выделены на карте глубины и на исходной картинке. оставить пары которые ближе друг к другу. 

https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_boundary_merge.html - # Hierarchical Merging of Region Boundary RAGs
https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_segmentations.html - # Comparison of segmentation and superpixel algorithms

https://github.com/Yangzhangcst/RGBD-semantic-segmentation/blob/master/README.md#Metrics - # RGBD semantic segmentation

https://ceur-ws.org/Vol-2665/paper24.pdf - Image Segmentation Based on RGBD Data

https://www.researchgate.net/figure/Correspondence-between-RGB-image-and-depth-map-The-3D-point-p-is-represented-by-its_fig2_327678902#:~:text=In%20this%20equation%2C%20f%20x%20%2C%20f%20y,by%20calibration%20procedures%20%5B45%5D. - фокусное растояние от камеры 

https://blender.stackexchange.com/questions/204886/calculating-3d-world-co-ordinates-using-depth-map-and-camera-intrinsics



