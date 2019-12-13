import os
import glob
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import PathPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from skimage.transform import resize
import cv2 as cv
import PIL.Image
import PIL.ImageOps
import networkx as nx
from tkinter import *
from tkinter import filedialog


def get_letter_directories(parent_dir):
    return [os.path.join(parent_dir, dir) for dir in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, dir))]
    
def get_image_files(parent_dir):
    types = ('*.jpg', '*.png', '*.eps', '*.pdf')
    files_grabbed = []
    for files in types:
        files_grabbed.extend(glob.glob(os.path.join(parent_dir, files)))
    return files_grabbed



def img_to_pie(fn, x, y, zoom=1, ax = None):
    if ax==None: 
        ax=plt.gca()
    
    #im = plt.imread(fn, format='png')
    im = cv.imread(fn)
    im = cv.resize(im, (300, 300), interpolation=cv.INTER_NEAREST)
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    
    oim = OffsetImage(im, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(oim, (x0, y0), xycoords='data', frameon=True)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    

    
    
def generate_print_template(letter_dir):
    # Read the images
    image_files = get_image_files(letter_dir)
    if len(image_files) != 6:
        raise RuntimeError("Wrong number of images! Must be exactly SIX images in either {png, jpg, eps, pdf} format!")
    print(image_files)

    # Init figure
    fig, ax = plt.subplots()
    
    # Data to plot
    size_img = [1] * 6
    size_clamp = [1] * 18
    colors = ['#ff6666', '#ffcc99', '#99ff99', '#66b3ff', '#c2c2f0', '#4e12fa']
    wedgeprops = {'linewidth': 1, 'edgecolor': 'k'}
    
    # Plot
    plt.pie(size_clamp, startangle=0, frame=True, wedgeprops=wedgeprops)
    img_wedges, _ = plt.pie(size_img, startangle=0, colors=colors, radius=0.8, wedgeprops=wedgeprops)
    center_circle = plt.Circle((0,0), 0.3, color='black', fc='white', linewidth=1.5)
    ax.add_artist(center_circle)
    
    for i, image_file in enumerate(image_files):
        R = 0.6
        ang_offs = (360/6/2)/180*np.pi
        ang = (360/6)*i/180*np.pi + ang_offs
        x = R*np.cos(ang)
        y = R*np.sin(ang)
        img_to_pie(image_file, x, y, zoom=0.1, ax=ax)
      
    # Plot settings
    #fig.tight_layout()
    ax.axis('equal')
    ax.axis('off')
    plt.show()
    
    
    
    

if __name__ == '__main__':
    # Settings
    letters_dir = r"./letters"
    letter_dirs = get_letter_directories(letters_dir)
    
    # Template generation
    for letter_dir in letter_dirs:
        generate_print_template(letter_dir)