"""Process.py functions.
This class implements different functions for image processing.
"""
import argparse
import numpy as np
import cv2
import imutils


def correct_image_rotation(image):
    """
    Find start point of board and correct the image rotation\n
    Parameters:\n
        image: Board Image,\n
    return: image
    """

    hsv = cv2.cvtColor(np.uint8(image), cv2.COLOR_BGR2HSV)
    green_pixels = color_pixel_detection(hsv, {'lower': [(36, 25, 25),
                                                         (86, 255, 255)]})
    cv2.namedWindow("Detection Result", cv2.WINDOW_AUTOSIZE)
    rows = green_pixels.shape[0]
    cols = green_pixels.shape[1]
    if green_pixels[0][cols-1] > 0:
        rotated = rotated = imutils.rotate_bound(image, 270)
    elif green_pixels[rows-1][0] > 0:
        rotated = rotated = imutils.rotate_bound(image, 90)
    elif green_pixels[rows-1][cols-1] > 0:
        rotated = rotated = imutils.rotate_bound(image, 180)
    else:
        rotated = image

    cv2.imshow("Detection Result", rotated)
    cv2.waitKey(0)


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

    result = cv2.GaussianBlur(result, (9, 9), sigmaX=2, sigmaY=2)

    return result


def find_largest_rectangle_position(edged):
    """
    Find the largest rectangle on picture (aprox board)\n
    Parameters:\n
        path: edged image,\n
    return: rectangle position vertices
    """
    # Find contours in the edged image, keep only the largest ones, and
    # initialize our screen contour
    cnts = cv2.findContours(edged, cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    screen_cnt = None
    # loop over our contours
    for _c in cnts:
        # approximate the contour
        peri = cv2.arcLength(_c, True)
        approx = cv2.approxPolyDP(_c, 0.015 * peri, True)
        # if our approximated contour has four points, then
        # we can assume that we have found our board
        if len(approx) == 4:
            screen_cnt = approx
            break
    return screen_cnt


def get_board_of_image(path):
    """
    Find command board on picture\n
    Parameters:\n
        path: Image directory\n
    return: board as image
    """
    img = cv2.imread(path)
    # Filter noise
    img = cv2.medianBlur(img, 3)
    # Convert image to hsv
    hsv = cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2HSV)
    # Find red pixels
    range_treshold_dict = {'lower': [(0, 100, 100), (10, 255, 255)],
                           'upper': [(160, 100, 100), (179, 255, 255)]}
    result = color_pixel_detection(hsv, range_treshold_dict)
    # Find edges
    edged = cv2.Canny(result, 30, 200)
    # Find potencial board vertices
    screen_cnt = find_largest_rectangle_position(edged)
    # Get the board as image
    pts = screen_cnt.reshape(4, 2)
    rectangle = four_point_transform(img, pts)

    # Show image
    # cv2.namedWindow("Detection Result", cv2.WINDOW_AUTOSIZE)
    # cv2.imshow("Detection Result", result)
    # cv2.namedWindow("Edged Result", cv2.WINDOW_AUTOSIZE)
    # cv2.imshow("Edged Result", edged)
    cv2.drawContours(img, [screen_cnt], -1, (0, 255, 0), 3)
    cv2.imshow("Board Detection Result", img)
    cv2.imshow("Warped Board", rectangle)
    cv2.waitKey(0)

    return rectangle


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


def arguments():
    """ Parse command line arguments
    Return:
        array
    """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-p', '--path',
                                 default='test_15.png',
                                 help='path of picture')
    args = vars(argument_parser.parse_args())
    return args


if __name__ == '__main__':
    ARGS = arguments()
    BOARD = get_board_of_image(ARGS['path'])
    correct_image_rotation(BOARD)
