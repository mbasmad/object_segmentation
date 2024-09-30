#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 23:38:03 2023

@author: martin
"""
import os
import cv2
from tkinter import filedialog, Tk, Button, Label
#%%
def open_video():
    video = filedialog.askopenfile(mode='r')
    if video:
       video_path = os.path.abspath(video)
       global cap
       cap = cv2.VideoCapture(video_path)
    
class my_button(Button):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def change_color(self):
        self.configure(bg='steelblue', fg='white')
    
    def reset_color(self):
       self.configure(bg='gainsboro', fg='black')
#%%
root = Tk()
root.title('Análisis de Cuidado parental')

label_inicial = Label(root, text='Seleccione el video a procesar:')
label_inicial.grid(row=0, column=0)
button_inicial = my_button(root, text='Buscar video', command=open_video)
button_inicial.grid(row=0, column=1)

root.mainloop()

