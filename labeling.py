import cv2
import numpy as np
from queue import Queue

def label_connected_components(binary_image):
    height, width = binary_image.shape
    label = 0
    labels = np.zeros((height, width), dtype=int)
    edges = [[] for _ in range(1000)]

    # First pass
    for i in range(height):
        for j in range(width):
            if binary_image[i, j] == 0 and labels[i, j] == 0:
                L = []
                for ni, nj in neighbors(i, j, height, width):
                    if labels[ni, nj] > 0:
                        L.append(labels[ni, nj])
                
                if len(L) == 0:
                    label += 1
                    labels[i, j] = label
                else:
                    x = min(L)
                    labels[i, j] = x

                    for y in L:
                        if y != x:
                            edges[x].append(y)
                            edges[y].append(x)

    # Second pass
    newlabel = 0
    newlabels = np.zeros(label + 1, dtype=int)

    for i in range(1, label + 1):
        if newlabels[i] == 0:
            newlabel += 1
            Q = Queue()
            newlabels[i] = newlabel
            Q.put(i)

            while not Q.empty():
                x = Q.get()
                for y in edges[x]:
                    if newlabels[y] == 0:
                        newlabels[y] = newlabel
                        Q.put(y)

    # Assign new labels to the original image
    result_image = np.zeros_like(binary_image)
    for i in range(height):
        for j in range(width):
            labels[i, j] = newlabels[labels[i, j]]
            if binary_image[i, j] == 0:
                result_image[i, j] = labels[i, j]

    return result_image

def neighbors(i, j, height, width):
    for ni in range(i-1, i+2):
        for nj in range(j-1, j+2):
            if 0 <= ni < height and 0 <= nj < width and (ni != i or nj != j):
                yield ni, nj

# Example usage
file_path = "./letters.bmp"
image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
binary_image = (image == 255).astype(np.uint8)

connected_components_image = label_connected_components(binary_image)

cv2.imshow("Connected Components", connected_components_image*10)
cv2.waitKey(0)
cv2.destroyAllWindows()
