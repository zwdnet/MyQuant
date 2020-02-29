# coding:utf-8
"""k-means图像分割实例"""


import numpy as np
import PIL.Image as image
from sklearn.cluster import KMeans


# 加载图片
def loadData(filePath):
    f = open(filePath, "rb")
    data = []
    img = image.open(f)
    m,n = img.size
    for i in range(m):
        for j in range(n):
            x, y, z = img.getpixel((i, j))
            data.append([x/256.0, y/256.0, z/256.0])
    f.close()
    return np.mat(data), m, n


if __name__ == "__main__":
    imgData, row, col = loadData("test.jpg")
    km = KMeans(n_clusters = 3)
    # 聚类获取每个像素点颜色所属的类别
    label = km.fit_predict(imgData)
    # print(label)
    label = label.reshape([row, col])
    # print(label)
    # 输出结果到图片
    pic_new = image.new("L", (row, col))
    for i in range(row):
        for j in range(col):
            pic_new.putpixel((i,j), int(256/(label[i][j]+1)))
    pic_new.save("result.jpg", "JPEG")
    
