#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 00:38:07 2023

@author: martin
"""
import cv2
import pandas as pd
from joblib import load
import image_functions
import os
import numpy as np

def analyze_video(video_file):
    """
    Analyze the selected video file, detect particles in the nest area, and save results to CSV.

    Parameters:
        video_file (str): The path of the video file to analyze.

    Returns:
        str: A message describing the analysis results.
    """
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise IOError(f'Unable to open: {video_file}')

    # Generate the image and delimitate nest
    nest_image = image_functions.get_final_image(cap) 

    # Get arena coordinates
    arena_top_left, arena_bottom_right = image_functions.get_coordinates(nest_image, message='arena')
    x0, y0 = arena_top_left
    x1, y1 = arena_bottom_right
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    nest_image = nest_image[y0:y1, x0:x1, :]

    # Set video to the first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    res, frame = cap.read()
    width, height, fps, frame_count = image_functions.get_video_features(cap)

    # Initialize lists to store the bounding box coordinates
    left_top_corner, right_bottom_corner = [], []

    # Get nest coordinates
    left_top_corner, right_bottom_corner = image_functions.get_coordinates(nest_image, message='nest')
    nest_x0, nest_y0 = left_top_corner
    nest_x1, nest_y1 = right_bottom_corner
    nest_x0, nest_x1 = min(nest_x0, nest_x1), max(nest_x0, nest_x1)
    nest_y0, nest_y1 = min(nest_y0, nest_y1), max(nest_y0, nest_y1)

    # Load ML models if needed
    kmeans = load('./kmeans.joblib') 
    scaler = load('./scaler.joblib')

    # Initialize a list to store the 'x' and 'y' coordinates of particles in area
    particles_location = []
    # Initialize FPS of video and conut of frames when the animal is in area
    sum_entries_area = 0
    frames_analyzed = 0
    sum_frames_in_area = 0
    fps = image_functions.get_video_features(cap)[2]
    in_area_flag = False
    current_frame_analyzed = 0
    while True:
        ret, frame = cap.read()
        if frame is None:
            break

        frame = frame[y0:y1, x0:x1, :]
        filtered_image = image_functions.get_filtered_image(frame)
        filtered_image[nest_y0:nest_y1, nest_x0:nest_x1] = cv2.subtract(filtered_image[nest_y0:nest_y1, nest_x0:nest_x1], 15)
        mask, contours = image_functions.get_mask_and_contours(filtered_image)
        last_in_area = in_area_flag
        in_area_flag = False
        for particle in contours:
            M = cv2.moments(particle)

            row, hull, ellipse = image_functions.get_contour_features(particle)
            row[0] = row[0] / (nest_x1 - nest_x0) # Normalize the area to the width of area analyzed
            if ellipse and row[0] > 2:
                cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
                row[0] = scaler.transform(np.asarray(row[0]).reshape(1, -1))[0][0]
                contour_in_area = image_functions.in_area(nest_x0, nest_y0, nest_x1, nest_y1, cx, cy)
                if contour_in_area:
                    # Store 'x' and 'y' coordinates of particle in area
                    particles_location.append((cx, cy, True, current_frame_analyzed))
                    row.append(1) # append the presence of contour in area as a feature of row
                    in_area_flag = True
                    
                else:
                    particles_location.append((cx, cy, False, current_frame_analyzed))
                    row.append(0) # append the ausence of contour in area as a feature of row
                cluster_type = int(kmeans.predict([row]))
                if cluster_type in (1, 6, 0, 5, 8, 2):
                    image_functions.draw_contour(frame, particle, cx, cy, color=(0, 0, 255))
                    if contour_in_area: image_functions.draw_contour(frame, particle, cx, cy, color=(220, 0, 0))
            
        current_frame_analyzed += 1
        if in_area_flag:
            sum_frames_in_area += 1
        
        # Show quantity of seconds when the animal is in area
        frames_analyzed += 1
        sec_in_area = sum_frames_in_area/fps
        total_sec_analyzed = frames_analyzed/fps
        if last_in_area == False and in_area_flag:
            sum_entries_area += 1 
        cv2.putText(frame, f'Seconds analyzed: {int(total_sec_analyzed)}', 
                    (250, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1) 
        cv2.putText(frame, f'Seconds in area: {int(sec_in_area)}', (250, 430), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1) 
        cv2.putText(frame, f'Entries to area: {int(sum_entries_area)}', 
                    (250, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        overlay = frame.copy()
        cv2.rectangle(overlay, left_top_corner, right_bottom_corner, (0, 0, 255), -1)
        alpha = 0.3 
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        cv2.putText(frame, 'Presionar "q" para salir', (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 0), 1) 
        
        cv2.imshow('Frame', frame)
        cv2.moveWindow('Frame', 700, 0) 

        keyboard = cv2.waitKey(1)
        if keyboard == ord('q'):
            break

    cv2.destroyAllWindows()

    # Return an analysis summary message
    sep = os.sep
    video_name = video_file.split(sep)[-1].split('.')[0]
    descr_video_name = f'video_analizado: {video_name}'
    descr_fps = f'FPS: {fps}'
    descr_analysis_area = f'area_arena: x= {x0}-{x1} y= {y0}-{y1}'
    descr_nest_area = f'area_nido: x= {nest_x0}-{nest_x1} y= {nest_y0}-{nest_y1}'
    
    descr_seg_in_area = f'total_segundos_en_nido: {sec_in_area}'
    total_seg_analyzed = f'Total_segundos_analizados: {total_sec_analyzed}'
    
    header_1 = f'{descr_video_name}\n{descr_fps}\n{descr_analysis_area}\n'
    header_2 = f'\n{descr_nest_area}\nTotal entries: {sum_entries_area}'
    header_3 = f'\n{descr_seg_in_area}\n{total_seg_analyzed}'
    header = header_1 + header_2 + header_3
    df_particles_location = pd.DataFrame(particles_location, columns=['x', 'y', 'in_area', 'Frame'])

    # Save the DataFrame to a CSV file
    with open(f'{video_name}.csv', 'w') as f:
        f.write(header + '\n')
        df_particles_location.to_csv(f, index=False)

    return f'Analisis completo!\nEl archivo .csv del analisis se guardó como: {video_name}.csv\n{header}'
