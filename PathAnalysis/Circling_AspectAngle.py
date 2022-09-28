# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tkinter
from tkinter import messagebox, simpledialog
import tkinter.filedialog as filedialog
from CircleMouseUtils import doIntersect, checkDuplicates, minBoundingRect, checkAngle

    # store results as list of [circle starts, circle end])
circle_starts_ends = []

    # Set parameters for detection
min_duration = 0.35 # seconds
max_duration = 1.45 # seconds
min_consist = 0.56 # 56% of time in correct direction
min_rotation = 0.1 # 36 degrees
min_diameter = 0.5 # bodylengths
max_aspect = 2.5 # ratio; long side of fitted rectangle over short

tkinter.Tk().withdraw()
csvfiles = filedialog.askopenfilenames(title="Please select one or more CSV files of tracking data to analyze!",\
                                     initialdir='./',\
                                     filetypes=(('CSV files', '*.csv'),))

frames_per_sec = int(simpledialog.askfloat("Confirm FPS","What is the framerate of your video(s)?"))

for csvfile in csvfiles:
    csvdata = pd.read_csv(csvfile)
        # Box-angle method uses both snout and tail
        # This script assumes a DLC-style CSV file in which the columns are:
        # frame #, snout X position, snout Y position, snout label confidence,  tail X position, tail Y position, tail label confidence
            # if using a different tracking method, we suggest creating a modified CSV to fit the above format!
            # alternatively, you can edit the '.iloc[]' coordinates below to match your format
    x_snout = np.array(csvdata.iloc[2:,1]).astype(float)
    y_snout = np.array(csvdata.iloc[2:,2]).astype(float)
    x_tail = np.array(csvdata.iloc[2:,4]).astype(float)
    y_tail = np.array(csvdata.iloc[2:,5]).astype(float)
    
        # accumulated turns over time, in units of radians
    xdiff = np.reshape(x_snout-x_tail,(-1,1))
    ydiff = np.reshape(y_snout-y_tail,(-1,1))
    nturns = np.unwrap(np.arctan2(xdiff, ydiff).T)[0]
    
        # how long is the mouse in pixels?
    body_lengths = np.linalg.norm(np.concatenate((xdiff,ydiff),axis=1),axis=1)
    body_len = np.median(body_lengths)
    
    print("Data loaded, beginning circle ID...")
    min_frames = int(round(min_duration * frames_per_sec))
    max_frames = int(round(max_duration * frames_per_sec))
    
    # Use the extremes of <min_frames>, <max_frames> to ensure checking each possibly-viable timepoint
    for start_point in range(len(x_snout) - min_frames):
        for end_point in range(start_point + min_frames, min(len(x_snout), start_point + max_frames)):
            #check if endpoint intersects with start point            
            if doIntersect((x_snout[start_point], y_snout[start_point]),
                           (x_snout[start_point+1], y_snout[start_point+1]),
                           (x_snout[end_point-1], y_snout[end_point-1]),
                           (x_snout[end_point], y_snout[end_point])):
                # don't want multiple intersections at once:
                if not checkDuplicates(x_snout[start_point:end_point], y_snout[start_point:end_point]):
                    ratio, size = minBoundingRect(np.stack((x_snout[start_point:end_point],y_snout[start_point:end_point]),1))
                    size /= body_len
                    num_turns, consistency = checkAngle(nturns[start_point:end_point])
                    duration = end_point - start_point # in frames, not sec
                        
                    # check whether it fits given parameters
                    # given the above constraints, it will automatically fit duration
                    if consistency >= min_consist and num_turns >= min_rotation \
                        and size >= min_diameter and ratio <= max_aspect:
    
                        # make sure to only add it if it is not too soon:
                        # either if it is the first video added
                        # or if the start point is at least <min_frame> later
                        if (not len(circle_starts_ends)) or (start_point >= circle_starts_ends[-1][0]*frames_per_sec + min_frames):
                            circ = [start_point / frames_per_sec, end_point / frames_per_sec]
                            circle_starts_ends.append(circ)
    
    print('\tSaving circle IDs for ', csvfile[csvfile.rfind('/')+1:-4]+"\n")
    circlecsv = pd.DataFrame(circle_starts_ends, columns=['start','end'])
    circlecsv.to_csv('./Circletimes_'+csvfile[csvfile.rfind('/')+1:-4]+".csv")
    print('Saved! Check the CSV in the same folder as your tracking data.')
    