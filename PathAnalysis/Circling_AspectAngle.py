# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tkinter
from tkinter import simpledialog
import tkinter.filedialog as filedialog
from CircleMouseUtils import doIntersect, checkDuplicates, minBoundingRect, checkAngle

    # store results as list of [circle starts, circle end])
circle_starts_ends = []

    # Set parameters for detection
min_duration = 0.35 # seconds
max_duration = 1.13 # seconds
min_consist = 0.74 # % of time in correct direction
min_rotation = 0.64 # fraction of rotation; x360 to get degrees
max_rotation = 1.125# fraction of rotation; x360 to get degrees
min_diameter = 0.55 # bodylengths
max_diameter = 2.14 # bodylengths
max_minor_axis = 1.57 # bodylengths
max_aspect = 2.12 # ratio; long side of fitted rectangle over short

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
    
        # clean up places where labels jump too large a fraction of the estimated mouse size
        # i.e. the mouse will never truly move at 40x their body size per second
    snout_velocity = np.sqrt(np.power(np.diff(x_snout),2) + np.power(np.diff(y_snout),2)) / body_len * frames_per_sec # accounts for framerate
    tail_velocity = np.sqrt(np.power(np.diff(x_tail),2) + np.power(np.diff(y_tail),2)) / body_len * frames_per_sec
    
    bad_indices = np.argwhere(snout_velocity >= 40)+1
    bad_indices = np.sort(np.unique(np.concatenate((bad_indices-2, bad_indices-1, bad_indices, bad_indices+1, bad_indices+2))))                    
    bad_indices = np.delete(bad_indices, np.argwhere(bad_indices < 0))
    bad_indices = np.delete(bad_indices, np.argwhere(bad_indices >= len(x_snout)))
    x_snout[bad_indices] = np.NaN
    y_snout[bad_indices] = np.NaN
    real_indices = np.squeeze(np.argwhere(~np.isnan(x_snout)))
    x_snout[bad_indices] = np.interp(bad_indices,real_indices,x_snout[real_indices])
    y_snout[bad_indices] = np.interp(bad_indices,real_indices,y_snout[real_indices])
    
    bad_indices = np.argwhere(tail_velocity >= 40)+1
    bad_indices = np.sort(np.unique(np.concatenate((bad_indices-2, bad_indices-1, bad_indices, bad_indices+1, bad_indices+2))))                    
    bad_indices = np.delete(bad_indices, np.argwhere(bad_indices < 0))
    bad_indices = np.delete(bad_indices, np.argwhere(bad_indices >= len(x_snout)))
    y_tail[bad_indices] = np.NaN
    x_tail[bad_indices] = np.NaN
    real_indices = np.squeeze(np.argwhere(~np.isnan(x_tail)))
    x_tail[bad_indices] = np.interp(bad_indices,real_indices,x_tail[real_indices])
    y_tail[bad_indices] = np.interp(bad_indices,real_indices,y_tail[real_indices])
    
    # recalculate after cleaning data
    xdiff = np.reshape(x_snout-x_tail,(-1,1))
    ydiff = np.reshape(y_snout-y_tail,(-1,1))
    
    # accumulated turns over time, in units of radians
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
                    aspect, major_axis = minBoundingRect(np.stack((x_snout[start_point:end_point],y_snout[start_point:end_point]),1))
                    major_axis /= body_len
                    minor_axis = major_axis / aspect
                    
                    num_turns, consistency = checkAngle(nturns[start_point:end_point])
                    duration = end_point - start_point # in frames, not sec
                        
                    # check whether it fits given parameters
                    # given the above constraints, it will automatically fit duration
                    if consistency >= min_consist and num_turns >= min_rotation and num_turns <= max_rotation \
                        and minor_axis >= min_diameter and minor_axis <= max_minor_axis\
                        and major_axis <= max_diameter and aspect <= max_aspect:
    
                        # make sure to only add it if it is not too soon:
                        # either if it is the first video added
                        # or if the start point is at least <min_frame> later
                        
                        if (not len(circle_starts_ends)) or (start_point >= circle_starts_ends[-1][0]*frames_per_sec + min_frames):
                            circ = [start_point / frames_per_sec, end_point / frames_per_sec, consistency, num_turns, minor_axis, major_axis, aspect]
                            circle_starts_ends.append(circ)
    
    print('\tSaving circle IDs for ', csvfile[csvfile.rfind('/')+1:-4]+"\n")
    circlecsv = pd.DataFrame(circle_starts_ends, columns=['start','end','consistency','turns','minor axis','major axis','aspect ratio'])
    circlecsv.to_csv('./Circletimes_'+csvfile[csvfile.rfind('/')+1:-4]+".csv")
    print('Saved! Check the CSV in the same folder as this script.')
    