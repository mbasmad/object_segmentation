#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 21:06:20 2023

@author: martin
"""
import tkinter as tk
from tkinter import filedialog
import video_analyzer
import ffmpeg
import os
from sklearn.cluster  import KMeans
from sklearn.preprocessing import RobustScaler

video_file = None

def video_transf(video, options=None):
    """
    Transforms a video file using specified options and saves the output as an AVI file.
    
    Args:
        video (str): The path of the input video file.
        options (dict, optional): A dictionary containing options for video transformation.
            Supported options:
                - 'fps': int, frame rate per second (default: None).
                - 'scale': tuple, containing width and height (default: None).
                - 'codec': str, video codec (default: 'mjpeg').
    
    Returns:
        None
    """
    output_name = video.split('.')[0] + '.avi'
    stream = ffmpeg.input(video)
    
    if options and 'fps' in options:
        stream = ffmpeg.filter(stream, 'fps', options['fps'], round='up')
    
    if options and 'scale' in options:
        stream = ffmpeg.filter(stream, 'scale', *options['scale'])
    
    codec = options['codec'] if options and 'codec' in options else 'mjpeg'
    
    stream = ffmpeg.output(stream, output_name, vcodec=codec)
    try:
        ffmpeg.run(stream, cmd=os.path.join(os.getcwd(), 'ffmpeg', 'bin', 'ffmpeg.exe'))
    except:
        ffmpeg.run(stream)  

def browse_video():
    """
    Opens a file dialog to browse and select a video file.
    
    Returns:
        str or None: The selected video file path or None if no file was selected.
    """
    global video_file
    video_file = filedialog.askopenfilename(title="Select a video file")
    if not video_file:
        return
    return video_file


def analyze_video():
    """
    Analyze the selected video file, detect particles in the nest area, and show results in a message box.
    """
    global video_file

    if video_file is None:
        tk.messagebox.showinfo("Information", "Please select a video file first.")
        return

    try:
        result_message = video_analyzer.analyze_video(video_file)
        tk.messagebox.showinfo("Analysis Complete", result_message)

    except Exception as e:
        tk.messagebox.showerror("Error", str(e))

def update_label_text(label):
    label.config(text=video_file)
    label.after(100, update_label_text, label)

def main():
    root = tk.Tk()
    root.title('Video Analysis')
    root.geometry('400x200+0+0')  # Set the dimensions and location of the root window
    # The format is "widthxheight+X+Y" where X and Y are the coordinates to place the window

    label = tk.Label(root, text='Select a video file to analyze:')
    label.pack(pady=5)

    button = tk.Button(root, text='Seleccionar Video', command=browse_video)
    button.pack(pady=5)
    
    label_file = tk.Label(root, text=video_file)
    label_file.pack(pady=5)
    
    button_video_transform = tk.Button(root, 
                                       text='Transformar video', 
                                       command=lambda: video_transf(video_file, {'fps': 5, 'scale': [720, 540]}))
    button_video_transform.pack(pady=5)

    button_analyze = tk.Button(root, text='Analizar Video', command=analyze_video)
    button_analyze.pack(pady=5)

    button_exit = tk.Button(root, text='Exit', command=root.destroy)
    button_exit.pack(pady=5)

    # Update the label every 100 milliseconds
    update_label_text(label_file)

    root.mainloop()

if __name__ == "__main__":
    main()


