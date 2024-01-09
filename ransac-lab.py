import cv2
import numpy as np
import random
import math

def find_black_points(image):
    black_points = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] == 0:
                black_points.append((j, i))
    return black_points

def calculate_ransac_parameters(q, t, p, points):
    N = int(math.log(1.0 - p) / math.log(1.0 - q*q))
    T = int(q * len(points))
    return N, T

def ransac_line_fitting(points, t, N, T):
    best_line = (0.0, 0.0, 0.0)  # (a, b, c)
    best_inlier_count = 0

    for _ in range(N):
        random_index1 = random.randint(0, len(points) - 1)
        random_index2 = random.randint(0, len(points) - 1)
        
        while random_index1 == random_index2:
            random_index2 = random.randint(0, len(points) - 1)

        p1 = points[random_index1]
        p2 = points[random_index2]

        a = p1[1] - p2[1]
        b = p2[0] - p1[0]
        c = p1[0] * p2[1] - p2[0] * p1[1]

        inlier_count = 0

        for point in points:
            distance = abs(a * point[0] + b * point[1] + c) / math.sqrt(a * a + b * b)
            if distance < t:
                inlier_count += 1

        if inlier_count > best_inlier_count:
            best_line = (a, b, c)
            best_inlier_count = inlier_count

        if best_inlier_count >= T:
            break

    return best_line

def draw_optimal_line(image, a, b, c):
    result_image = np.ones_like(image) * 255

    # Draw the black points
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] == 0:
                result_image[i, j] = 0

    # Draw the horizontal line
    if abs(b) >= abs(a):
        y1 = int(-c / b)
        y2 = int((-c - a * image.shape[1]) / b)
        cv2.line(result_image, (0, y1), (image.shape[1], y2), 0, 2)
    # Draw the vertical line
    else:
        x1 = int(-c / a)
        x2 = int((-c - b * image.shape[0]) / a)
        cv2.line(result_image, (x1, 0), (x2, image.shape[0]), 0, 2)

    return result_image

def main():
    file_path = "./points2.bmp"

    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    points = find_black_points(image)

    q = 0.3
    t = 10.0
    p = 0.99

    N, T = calculate_ransac_parameters(q, t, p, points)
    print("N =", N)
    print("T =", T)

    a, b, c = ransac_line_fitting(points, t, N, T)
    print("a =", a)
    print("b =", b)
    print("c =", c)

    result_image = draw_optimal_line(image, a, b, c)

    cv2.imshow("RANSAC Line Fitting", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
