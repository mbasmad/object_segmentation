#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 22:05:45 2023

@author: martin
"""
import numpy as np
import cv2
import math
import scipy.special as special
#%%
def get_video_features(cap):
    """
    Get various video features from the given video capture object.

    Parameters:
        cap (cv2.VideoCapture): The video capture object from which to retrieve video features.

    Returns:
        tuple: A tuple containing the following video features:
            width (float): The width of the video frames.
            height (float): The height of the video frames.
            fps (float): The frames per second (fps) of the video.
            frame_count (float): The total number of frames in the video.
    """
    # Get the width of the video frames (Property ID: 3)
    width = cap.get(3)

    # Get the height of the video frames (Property ID: 4)
    height = cap.get(4)

    # Get the frames per second (fps) of the video (Property ID: 5)
    fps = cap.get(5)

    # Get the total number of frames in the video (Property ID: 7)
    frame_count = cap.get(7)

    return width, height, fps, frame_count


def get_mask_and_contours(img):
    """
    Get the binary mask and sorted contours of an input image.
    
    Parameters:
        img (numpy.ndarray): Input image (grayscale or color).

    Returns:
        tuple: A tuple containing:
            - mask (numpy.ndarray): An empty mask with the same shape as the input image.
            - contours (list): A list of contours sorted by their areas in descending order.
    """
    
    # Create an empty mask with the same shape as the input image
    mask = np.zeros_like(img, np.uint8)
    
    # Threshold the image using Otsu's method to obtain a binary mask
    _, temp_mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Perform morphological opening to remove noise and smooth the mask
    kernel = np.ones((5, 5), np.uint8)
    temp_mask = cv2.morphologyEx(temp_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Find contours in the binary mask
    contours, _ = cv2.findContours(temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours based on their areas in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    return mask, contours

def get_filtered_image(img):
    """
    Apply a series of image filtering and corrections to the input image.

    Parameters:
        img (numpy.ndarray): Input image (BGR).

    Returns:
        numpy.ndarray: Filtered image (grayscale).
    """
    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Invert the grayscale image
    inverted_img = cv2.bitwise_not(gray_img)
    
    # Calculate the 1° quartile 
    q3 = np.percentile(inverted_img, 75, interpolation = 'midpoint')
    
    # Subtract the mean and standard deviation from the inverted image to center it around zero
    centered_img = cv2.subtract(inverted_img, q3)
    
    # Apply Gaussian blur to the centered image
    blurred_img = cv2.GaussianBlur(centered_img, (9, 9), 0)
    
    return blurred_img

def get_final_image(cap):
    """
    Get the final frame from the video capture.

    Parameters:
        cap (cv2.VideoCapture): Video capture object.

    Returns:
        numpy.ndarray or None: The final frame of the video as an image (BGR) or None if reading the frame fails.
    """
    try:
        # Get the total number of frames in the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Set the video capture to the frame just before the last one
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 10)
        
        # Read the frame at the current position
        ret, image = cap.read()
        
        # Check if reading the frame was successful
        if not ret:
            raise ValueError("Failed to read the final frame from the video.")
        
        return image
    except Exception as e:
        print(f"Error: {e}")
        return None

def get_contour_features(contour, frame=None):
    """
    Extract features from a given contour.

    Parameters:
        contour (numpy.ndarray): The contour to analyze.
        frame (numpy.ndarray or None): Optional. The frame on which the ellipse will be drawn. Default is None.

    Returns:
        tuple: A tuple containing the features extracted from the contour, the convex hull of the contour,
               and the fitted ellipse (if available).
    """
    # Bounding box features
    x, y, w, h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # Initialize ellipse features to default values
    ellipse_perimeter = shape_ellipse = ellipse_area = extent_ellipse = None
    ellipse = None
    ratio_perimeter = None  # Initialize ratio_perimeter here

    # Ellipse features
    try:
        ellipse = cv2.fitEllipse(contour)
        (x, y), axes, angle = ellipse
        major_axis = max(axes)
        minor_axis = min(axes)
        ellipse_area = math.pi / 4 * major_axis * minor_axis
        e_sq = 1.0 - (minor_axis ** 2) / (major_axis ** 2)
        ellipse_perimeter = 4 * major_axis * special.ellipe(e_sq)

        # Draw the ellipse on the frame (if provided)
        if frame is not None:
            cv2.ellipse(frame, ellipse, (255, 0, 0), 2)

        shape_ellipse = minor_axis / major_axis
        ratio_perimeter = perimeter / ellipse_perimeter

        extent_ellipse = area / ellipse_area

    except cv2.error:
        # If fitting the ellipse fails, leave ellipse features as None
        pass

    # Convex hull features
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area

    # Circularity
    circularity = 4 * math.pi * area / (perimeter ** 2)

    # Return features, convex hull, and ellipse (if available)
    features = [area, solidity, extent_ellipse, ratio_perimeter, shape_ellipse, circularity]
    return features, hull, ellipse


def in_area(nest_x0, nest_y0, nest_x1, nest_y1, cx, cy):
    """
    Check if a point (cx, cy) lies within the specified rectangle defined by its top-left (nest_x0, nest_y0) 
    and bottom-right (nest_x1, nest_y1) coordinates.

    Parameters:
        nest_x0 (int): X-coordinate of the top-left corner of the rectangle.
        nest_y0 (int): Y-coordinate of the top-left corner of the rectangle.
        nest_x1 (int): X-coordinate of the bottom-right corner of the rectangle.
        nest_y1 (int): Y-coordinate of the bottom-right corner of the rectangle.
        cx (int): X-coordinate of the point to check.
        cy (int): Y-coordinate of the point to check.

    Returns:
        bool: True if the point is inside the rectangle, False otherwise.
    """
    is_inside = nest_x0 <= cx <= nest_x1 and nest_y0 <= cy <= nest_y1
    return is_inside

def draw_contour(image, contour, center_x, center_y, color):
    """
    Draw the given contour and a circle at the specified (center_x, center_y) coordinates on the image.

    Parameters:
        image (numpy.ndarray): The input image to draw on.
        contour (numpy.ndarray): The contour to draw.
        center_x (int): The X-coordinate of the center of the circle.
        center_y (int): The Y-coordinate of the center of the circle.
        color (tuple): The RGB tuple representing the color for drawing the contour and circle.

    Returns:
        None: The function modifies the input image in place.
    """
    cv2.drawContours(image, [contour], 0, color, -1)
    cv2.circle(image, (center_x, center_y), 7, color, -1)

def get_coordinates(im, message):
    """
    Get coordinates by drawing a rectangle on the given image.

    Parameters:
        im (numpy.ndarray): Input image (BGR).
        message (str): Message to display on the window.

    Returns:
        tuple or None: A tuple containing two points (p0, p1) representing the top-left and bottom-right coordinates
                       of the drawn rectangle, or None if the selection was canceled.
    """
    RED = (0, 0, 255)
    BLUE = (255, 0, 0)
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.5
    FONT_THICKNESS = 2
    if message == 'arena':
        message = 'Seleccionar la arena con el mouse y presionar ENTER al finalizar'
    else:
        message = 'Seleccionar el nido con el mouse y presionar ENTER al finalizar'

    # Define variables to store the selected coordinates
    p0, p1 = None, None

    def mouse(event, x, y, flags, param):
        nonlocal p0, p1
        if event == cv2.EVENT_LBUTTONDOWN:
            p0 = x, y
            p1 = x, y
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            p1 = x, y
            img[:] = im.copy()
            cv2.rectangle(img, p0, p1, BLUE, 2)
            cv2.imshow('window', img)
        elif event == cv2.EVENT_LBUTTONUP:
            p0, p1 = (min(p0[0], p1[0]), min(p0[1], p1[1])), (max(p0[0], p1[0]), max(p0[1], p1[1]))
            cv2.rectangle(img, p0, p1, RED, 2)
            cv2.imshow('window', img)

    img = im.copy()
    cv2.putText(img, message, (10, 30), FONT, FONT_SCALE, RED, FONT_THICKNESS)  # Add the text here

    # Create the window and set its position on the screen
    cv2.imshow('window', img)
    cv2.moveWindow('window', 700, 0) 

    cv2.setMouseCallback('window', mouse)

    # Wait for a key press and check if the user pressed 'Enter' (key code 13)
    key = cv2.waitKey(0)

    # Close the window after key press or mouse selection
    cv2.destroyAllWindows()

    if key == 13:
        return p0, p1
    else:
        # If the user canceled the selection, return None
        return None




