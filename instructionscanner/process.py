"""Process.py functions.
This class implements different functions for image processing.
"""
import argparse
from pathlib import Path
import numpy as np
import cv2
import imutils
import math

WHITE_BALANCE = 60

def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()

def apply_threshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)

    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)

    return matrix

def simplest_cb(img, percent):
    assert img.shape[2] == 3
    assert percent > 0 and percent < 100

    half_percent = percent / 200.0

    channels = cv2.split(img)

    out_channels = []
    for channel in channels:
        assert len(channel.shape) == 2
        height, width = channel.shape
        vec_size = width * height
        flat = channel.reshape(vec_size)

        assert len(flat.shape) == 1

        flat = np.sort(flat)

        n_cols = flat.shape[0]

        low_val  = flat[math.floor(n_cols * half_percent)]
        high_val = flat[math.ceil( n_cols * (1.0 - half_percent))]


        # saturate below the low percentile and above the high percentile
        thresholded = apply_threshold(channel, low_val, high_val)
        # scale the channel
        normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized)

    return cv2.merge(out_channels)


def correct_image_rotation(image):
    """
    Find start point of board and correct the image rotation\n
    Parameters:\n
        image: Board Image,\n
    return: image
    """

    hsv = cv2.cvtColor(np.uint8(image), cv2.COLOR_BGR2HSV)
    blue_pixels = color_pixel_detection(hsv, {'lower': [(100, 50, 0),
                                                        (140, 255, 255)]})
    blue_pixels = cv2.GaussianBlur(blue_pixels, (5, 5), sigmaX=2, sigmaY=2)
    # cv2.imshow("Test", green_pixels)
    # cv2.waitKey(0)
    rows = blue_pixels.shape[0]
    cols = blue_pixels.shape[1]
    if blue_pixels[0][cols-1] > 0:
        rotated = rotated = imutils.rotate_bound(image, 270)
    elif blue_pixels[rows-1][0] > 0:
        rotated = rotated = imutils.rotate_bound(image, 90)
    elif blue_pixels[rows-1][cols-1] > 0:
        rotated = rotated = imutils.rotate_bound(image, 180)
    else:
        rotated = image
    return rotated


def color_pixel_detection(hsv, range_treshold_dict):
    """
    Find only red pixels on picture\n
    Parameters:\n
        hsv: HSV of original image,\n
        range_treshold_dict: Dictionary with lower and/or upper treshold\n
    return: image
    """
    # Apply filter mask
    if 'lower' in range_treshold_dict:
        result_lower = cv2.inRange(hsv, range_treshold_dict['lower'][0],
                                   range_treshold_dict['lower'][1])
        result = result_lower.copy()
    if 'upper' in range_treshold_dict:
        result_upper = cv2.inRange(hsv, range_treshold_dict['upper'][0],
                                   range_treshold_dict['upper'][1])
        result = result_upper.copy()
    if len(range_treshold_dict) == 2:
        result = cv2.addWeighted(result_lower, 1.0, result_upper, 1.0, 0)

    result = cv2.GaussianBlur(result, (5, 5), 0)
    # cv2.imshow("Color Filter", result)
    # cv2.waitKey(0)
    # result = cv2.medianBlur(result, 5)
    return result


def find_largest_rectangle_position(edged, num_rectangles=1):
    """
    Find the largest rectangle on picture (aprox board)\n
    Parameters:\n
        path: edged image,\n
        num_rectangles: Number of rectangles to find\n
    return: rectangle position vertices
    """
    # Find contours in the edged image, keep only the largest ones, and
    # initialize our screen contour
    cnts = cv2.findContours(edged, mode=cv2.RETR_EXTERNAL,
                            method=cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:20]
    screen_cnt = []
    # loop over our contours
    for _c in cnts:
        # approximate the contour
        peri = cv2.arcLength(_c, True)
        approx = cv2.approxPolyDP(_c, 0.015 * peri, True)
        # if our approximated contour has four points, then
        # we can assume that we have found our board
        if len(approx) == 4:
            screen_cnt.append(approx)
            if len(screen_cnt) == num_rectangles:
                break
    return screen_cnt


def find_figure_convex_hull(edged):
    """
    Determine figure on image\n
    Parameters:\n
        edged: edged image with figure\n
    return: image name
    """
    cnts = cv2.findContours(edged, mode=cv2.RETR_EXTERNAL,
                            method=cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    area = cv2.contourArea(cnts)
    convex_hull = cv2.boundingRect(cnts)
    area_ch = convex_hull[2]*convex_hull[3]
    factor = area/area_ch
    figure = []
    # Determine figure from factor
    if factor <= 0.62:
        figure = 'STAR'
    elif 0.62 < factor <= 0.7:
        figure = 'ARROW'
    elif 0.7 < factor <= 0.765:
        figure = 'PENTAGON'
    elif 0.765 < factor <= 0.8:
        figure = 'CIRCLE'
    elif 0.8 < factor <= 0.87:
        figure = 'TRAP'
    elif figure > 0.87:
        figure = 'SQUARE'

    # return figure, factor
    return figure


def get_board_of_image(path):
    """
    Find command board on picture\n
    Parameters:\n
        path: Image directory\n
    return: board as image
    """
    img = cv2.imread(path)
    # cv2.imshow('Original', img)
    img = simplest_cb(img, WHITE_BALANCE)
    # cv2.imshow('Test', img)
    # cv2.waitKey(0)
    # img = path
    # Filter noise
    img = cv2.medianBlur(img, 3)
    # Convert image to hsv
    hsv = cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2HSV)
    # Find red pixels
    range_treshold_dict = {'lower': [(0, 70, 50), (10, 255, 255)],
                           'upper': [(170, 70, 50), (180, 255, 255)]}
    result = color_pixel_detection(hsv, range_treshold_dict)
    # Find edges
    edged = cv2.Canny(result, 30, 200)
    # Find potencial board vertices
    try:
        screen_cnt = find_largest_rectangle_position(edged)[0]
        # Get the board as image
        pts = screen_cnt.reshape(4, 2)
        rectangle = four_point_transform(img, pts)
    except IndexError:
        screen_cnt = None
        rectangle = None

    # Show image
    # cv2.drawContours(img, [screen_cnt], -1, (0, 255, 0), 3)
    # cv2.imshow("Board Detection Result", img)
    # cv2.imshow("Warped Board", rectangle)
    # cv2.waitKey(0)

    # return rectangle, screen_cnt
    return rectangle


def get_figure_area(image):
    """
    Find figure squares\n
    Parameters:\n
        image: Board Image\n
    return: Array with pts of figure squares
    """
    # Detect yellow squares
    hsv = cv2.cvtColor(np.uint8(image), cv2.COLOR_BGR2HSV)
    green_pixels = color_pixel_detection(hsv, {'lower': [(36, 0, 0),
                                                         (86, 255, 255)]})
    # Find edges
    # edged = cv2.Canny(green_pixels, 30, 200)
    # cv2.imshow("Test", green_pixels)
    # cv2.waitKey(0)
    # Find potencial board vertices
    screen_cnt = find_largest_rectangle_position(green_pixels, 12)
    screen_cnt.sort(key=lambda x: get_contour_precedence(x, image.shape[1]))
    figures = []
    for _i, value in enumerate(screen_cnt):
        screen_cnt[_i] = value.reshape(4, 2)
        figures.append(four_point_transform(image, screen_cnt[_i]))
        screen_cnt[_i] = order_points(screen_cnt[_i])
    return figures


def order_points(pts):
    """
    Order the points of the rectangle as top-left, top-right, bottom-right,
    bottom-left.\n
    Author: Adrian Rosebrock\n
    Parameters:\n
        pts: Numpy(4,2) with the for points of the rectangle\n
    Return ordered points
    """
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    _s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(_s)]
    rect[2] = pts[np.argmax(_s)]
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    """
    Wrap the objective image from the original image\n
    Author: Adrian Rosebrock\n
    Parameters:\n
        image: original image,\n
        pts: Numpy(4,2) with the for points of the rectangle\n
    Return wrap image
    """
    # the width of new image
    rect = order_points(pts)
    (_tl, _tr, _br, _bl) = rect
    width_a = np.sqrt(((_br[0] - _bl[0]) ** 2) + ((_br[1] - _bl[1]) ** 2))
    width_b = np.sqrt(((_tr[0] - _tl[0]) ** 2) + ((_tr[1] - _tl[1]) ** 2))
    # height of new image
    height_a = np.sqrt(((_tr[0] - _br[0]) ** 2) + ((_tr[1] - _br[1]) ** 2))
    height_b = np.sqrt(((_tl[0] - _bl[0]) ** 2) + ((_tl[1] - _bl[1]) ** 2))

    # take the maximum of the width and height values to reach
    # our final dimensions
    max_width = max(int(width_a), int(width_b))
    max_height = max(int(height_a), int(height_b))

    # construct our destination points which will be used to
    # map the screen to a top-down, "birds eye" view
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    warp = cv2.warpPerspective(image, cv2.getPerspectiveTransform(rect, dst),
                               (max_width, max_height))

    # return the warped image
    return warp


def get_contour_precedence(contour, cols):
    """
    Beatiful way to set rank for order contours on image\n
    Source: https://stackoverflow.com\n
    Parameters:\n
        contour: Contours of image,\n
        cols: Image cols\n
    Return wrap image
    """
    tolerance_factor = 20
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols\
        + origin[0]


def get_commands(figures):
    """
    Get commands from figures\n
    Parameters:\n
        figures: Image with figures,\n
    Return List of commands
    """
    figure = []
    for fig in figures:
        # Detect blue figure
        hsv = cv2.cvtColor(np.uint8(fig), cv2.COLOR_BGR2HSV)
        blue_pixels = color_pixel_detection(hsv, {'lower': [(100, 50, 0),
                                                            (140, 255, 255)]})
        # cv2.imshow("Test", blue_pixels)
        # cv2.waitKey(0)
        # Determine figure
        try:
            figure.append(find_figure_convex_hull(blue_pixels))
        except IndexError:
            continue
    return figure


def arguments():
    """ Parse command line arguments
    Return:
        array
    """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-p', '--path',
                                 default="C:\\Users\\cardi\\AppData\\Local\\Packages\\CanonicalGroupLimited.UbuntuonWindows_79rhkp1fndgsc\\LocalState\\rootfs\\home\\carlos\\anaconda3\\envs\\visionArtificialEnv\\lib\\python3.7\\site-packages\\pyparrot\\images",
                                 help='path of picture')
    args = vars(argument_parser.parse_args())
    return args


def get_board_commands(path):
    """
    Get commands for drone\n
    Parameters:\n
        path: Images folder path,\n
    Return List of commands
    """
    commands_dic = {}
    max_len = 0
    images = list(Path(path).glob('*.png'))
    for img in images:
        board = get_board_of_image(str(img))
        if board is not None:
            rotated = correct_image_rotation(board)
            figures = get_figure_area(rotated)
            commands = tuple(get_commands(figures))
            if len(commands) >= max_len:
                max_len = len(commands)
                if commands not in commands_dic:
                    commands_dic[commands] = 1
                else:
                    commands_dic[commands] += 1
    commands_dic = {k: v for k, v in commands_dic.items() if len(k) == max_len}
    print(commands_dic)
    if max_len > 0:
        commands = max(commands_dic, key=commands_dic.get)
    else:
        commands = []
    return commands


if __name__ == '__main__':
    ARGS = arguments()
    # CAP = cv2.VideoCapture(ARGS['path'])
    # while True:
    #     RET, FRAME = CAP.read()
    #     if not RET:
    #         break
    #     BOARD, SCREEN_CNT = get_board_of_image(FRAME)
    #     if BOARD is not None or SCREEN_CNT is not None:
    #         ROTATED = correct_image_rotation(BOARD)
    #         cv2.drawContours(FRAME, [SCREEN_CNT], -1, (0, 255, 0), 10)
    #         cv2.imshow("Board Detection Result", FRAME)
    #         FIGURES = get_figure_area(ROTATED)
    #         COMMANDS = get_commands(FIGURES)
    #         print(COMMANDS)
    #     else:
    #         cv2.imshow("Board Detection Result", FRAME)
    #     cv2.waitKey(1)
    # CAP.release()

    print('\n Selected:', get_board_commands(ARGS['path']))
