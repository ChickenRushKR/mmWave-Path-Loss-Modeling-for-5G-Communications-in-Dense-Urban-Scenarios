from PIL import Image
import numpy as np
import math
import pandas as pd
import os
import argparse

x_min = 324600      # minimum x value
y_min = 4150520     # minimum y value
z_max = 288.43

# for grayscale
under_ground = 255.0
in_terrain = 200.0
in_building = 200.0
in_sky = 0.0


def discriminant(a, b, c):
    return ((-b + math.sqrt(b ** 2 - 4 * a * c)) / (2 * a), (-b - math.sqrt(b ** 2 - 4 * a * c)) / (2 * a))


def line_equation(a, d, gradient, z):
    return (a, a * gradient + d, z)


def sol(x1, y1, z1, gradient, width):
    d = y1 - gradient * x1
    a = gradient ** 2 + 1
    b = -2 * x1 + 2 * gradient * d - 2 * y1 * gradient
    c = x1 ** 2 + d ** 2 - 2 * y1 * d + y1 ** 2 - (width/2)**2
    return (
        line_equation(discriminant(a, b, c)[1], d, gradient, z1),
        line_equation(discriminant(a, b, c)[0], d, gradient, z1))


def angle_diff(Tx, Rx, azi):
    ln1 = Rx[0] - Tx[0]
    ln2 = Rx[1] - Tx[1]
    ln3 = Tx[2] - Rx[2]
    dist2d = math.sqrt(math.pow(ln1, 2) + math.pow(ln2, 2))

    theta = math.atan2(ln2, ln1)
    theta = theta * 180 / math.pi

    theta2 = math.atan2(ln3, dist2d)
    theta2 = theta2 * 180 / math.pi
    case = 0
    if ln2 > 0:
        if ln1 > 0:
            theta = 90 - theta
            case = 1
        else:
            theta = 450 - theta
            case = 2
    else:
        theta = 90 + theta * (-1)
        case = 3
    azi_diff = abs(theta - azi)
    return azi_diff


def gen_image(Tx_coord, Rx_coord, name, img_width, img_height, distance, ver, azimuth):
    Tx_coord[0] = Tx_coord[0] - x_min
    Tx_coord[1] = Tx_coord[1] - y_min
    Rx_coord[0] = Rx_coord[0] - x_min
    Rx_coord[1] = Rx_coord[1] - y_min
    # print(Tx_coord[2], ' -> ', end='')
    # Tx_coord[2] += dim_1[Tx_coord[0]][Tx_coord[1]]
    # print(Tx_coord[2], ' -> ', end='')
    # Rx_coord[2] = dim_1[Rx_coord[1]][Rx_coord[0]]
    if distance % 10 != 0:
        alpha = (Tx_coord[0] - Rx_coord[0]) / (distance - 3)
        beta = (Tx_coord[1] - Rx_coord[1]) / (distance - 3)
        Tx_coord[0] += alpha
        Tx_coord[1] += alpha
        Rx_coord[0] -= beta
        Rx_coord[1] -= beta
    # Tx_coord[2] += dim_1[Tx_coord[0]][Tx_coord[1]]
    # Rx_coord[2] = dim_1[Rx_coord[0]][Rx_coord[1]]
    if Tx_coord[1] == Rx_coord[1]:
        XYs = [[]]
        CentreDots = np.linspace(Tx_coord, Rx_coord, distance)
        for i in range(distance):
            XYs.append(getDots_Of_Orthogonal_Line(CentreDots[i], None, img_width, distance))
        if ver == 0:
            return los(XYs, Tx_coord, Rx_coord, name, img_width, img_height, distance)
        elif ver == 1:
            array_to_npy(XYs, Tx_coord, Rx_coord, name, img_width, img_height, distance)
        elif ver == 2:
            array_to_npy2(XYs, Tx_coord, Rx_coord, name, img_width, img_height, distance, azimuth)
        return
    gradient = (Tx_coord[0] - Rx_coord[0]) / (Tx_coord[1] - Rx_coord[1])
    gradient *= (-1)
    CentreDots = getDottedLines(Tx_coord, Rx_coord, distance)  # CentreDots,

    # Tx와 Rx를 잇는 직선위의 점들(distance로 uniform하게 나눠짐)
    # getDottedLines의 인자의 dim=2였으므로, 2차원 배열 반환
    XYs = [[]]
    for i in range(len(CentreDots)):  # 직선위의 점을 지나고, TxRx 직선을 수직하는 선분의 양 끝점들을 받아오기
        XYs.append(getDots_Of_Orthogonal_Line(CentreDots[i], gradient, img_width, distance))


    if ver == 0:
        return los(XYs, Tx_coord, Rx_coord, name, img_width, img_height, distance)
    elif ver == 1:
        array_to_npy(XYs, Tx_coord, Rx_coord, name, img_width, img_height, distance)
    elif ver == 2:
        array_to_npy2(XYs, Tx_coord, Rx_coord, name, img_width, img_height, distance, azimuth)


def getDottedLines(Tx, Rx, NumOfSlice):
    return np.linspace(Tx, Rx, NumOfSlice)


def getDots_Of_Orthogonal_Line(CenterPoints, gradient, img_width, distance):
    if gradient is None:
        twoDots = []
        twoDots.append((CenterPoints[0], CenterPoints[1] - (img_width / 2), CenterPoints[2]))
        twoDots.append((CenterPoints[0], CenterPoints[1] + (img_width / 2), CenterPoints[2]))
        return getDottedLines(twoDots[0], twoDots[1], img_width)
    twoDots = sol(CenterPoints[0], CenterPoints[1], CenterPoints[2], gradient, img_width)
    return getDottedLines(twoDots[0], twoDots[1], img_width)


def array_to_npy(XYs, Tx, Rx, name, img_width, img_height, distance):
    cube = np.arange(img_width * img_height * distance).reshape((img_width, img_height, distance))
    f = open("temp.txt", "w")
    for i in range(distance):
        for j in range(img_width):
            for k in range(img_height):
                cube[j][k][i] = 0

    for i in range(1, distance + 1):
        for j in range(img_width):
            for k in range(img_height):
                try:
                    x = int(np.rint(XYs[i][j][0]))
                    y = int(np.rint(XYs[i][j][1]))
                    z = int(np.rint(XYs[i][j][2] - (img_height / 2) + k))

                    if z < 0:
                        cube[k][j][i - 1] = under_ground
                    elif z < int(dim_1[y][x]):
                        cube[k][j][i - 1] = in_terrain
                    elif z < int(dim_2[y][x]):
                        cube[k][j][i - 1] = in_building
                    else:
                        cube[k][j][i - 1] = in_sky
                    if cube[k][j][i - 1] > 255:
                        cube[k][j][i - 1] = 255
                    # cube[k][j][i-1] /= 255

                    # print(str(x + x_min) + '\t' + str(y + y_min))
                except:
                    print(i, j, k)
                    print(cube.shape)
                    print(img_width, img_height, distance)
                    exit(1)
    cube = flip_or_not(cube, Tx[0] - Rx[0], Tx[1] - Rx[1], img_width, img_height, distance)
    np.save(name + '.npy', cube)


def flip_or_not(cube, a, b, img_width, img_height, distance):
    case = 0
    if a > 0:
        if b > 0:
            case = 1
    elif a < 0:
        if b >= 0:
            case = 1
    else:
        if b > 0:
            case = 1
    if case == 0:
        return cube
    if case == 1:
        f_cube = np.arange(img_width * img_height * distance).reshape((img_width, img_height, distance))
        for i in range(img_height):
            for j in range(img_width):
                for k in range(distance):
                    f_cube[i][j][k] = 0

        for i in range(img_height):
            for j in range(img_width):
                for k in range(distance):
                    try:
                        f_cube[i][j][k] = cube[i][j][img_width - k - 1]
                    except:
                        print(cube.shape, distance, img_width, img_height)
        return f_cube


def main():
    img_width, img_height, distance, dimension, ver = input("width, height, distance, dimension, version:  ").split(' ')
    img_width = int(img_width)
    img_height = int(img_height)
    distance = int(distance)
    ver = int(ver)
    if dimension == "3d":                   # 33333333333333DDDDDDDDDDDDDDDDDDDD
        try:
            os.stat('lams')
        except:
            os.mkdir('lams')
        train_path = 'lams/RSS_' + str(img_width) + '_' + str(img_height) + '_' + str(distance) + '_v' + str(ver)
        try:
            os.mkdir(train_path)
        except:
            print("덮어쓰기")
            Tx = [args.txx, args.txy, args.txz]
            Rx = [args.rxx, args.rxy, args.rxz]
            azimuth = args.azimuth
            gen_image(Tx, Rx, args.name, img_width, img_height, distance, ver, azimuth)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-txx', required=False, type=float, default=0, help='tx X 좌표')
    parser.add_argument('-txy', required=False, type=float, default=0, help='tx Y 좌표')
    parser.add_argument('-txz', required=False, type=float, default=0, help='tx Z 좌표')
    parser.add_argument('-rxx', required=False, type=float, default=0, help='rx X 좌표')
    parser.add_argument('-rxy', required=False, type=float, default=0, help='rx Y 좌표')
    parser.add_argument('-rxz', required=False, type=float, default=0, help='rx Z 좌표')
    parser.add_argument('-azimuth', required=False, type=float, default=0, help='azimuth')
    parser.add_argument('-name', required=False, type=str, default=0, help='name')
    args = parser.parse_args()
    main(args)