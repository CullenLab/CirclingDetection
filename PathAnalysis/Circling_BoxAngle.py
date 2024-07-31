# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tkinter
from tkinter import simpledialog
import tkinter.filedialog as filedialog
from CircleMouseUtils import doIntersect, checkDuplicates, minBoundingRect, checkAngle, fit_Turns_dist, fit_Minor_dist, fit_Major_dist

    # store results as list of [circle starts, circle end])
collision_stats = [] # collisions are candidate circles
circle_stats = [] # candidates that pass the filters

    # constrain how far to search (seconds)
min_duration = 0.1 
max_duration = 2
    # Set parameters for detection (SD from mean)
p_Turns_min= -1.0658927118695667
p_Turns_max= 0.23836228645569535 
p_Minor_min= -1.2792097710034533 
p_Major_max= -0.1411313323043651

tkinter.Tk().withdraw()
csvfiles = filedialog.askopenfilenames(title="Please select one or more CSV files of tracking data to analyze!",initialdir='./',filetypes=(('CSV files', '*.csv'),))

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
    
    # First find all collisions in the snout path
    for start_point in range(len(x_snout) - 4):
        for end_point in range(start_point + 4, min(len(x_snout), start_point + (frames_per_sec*2) + 1)):
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
                    
                    num_turns = checkAngle(nturns[start_point:end_point])
                    
                    first = start_point / frames_per_sec
                    last = end_point / frames_per_sec
                    
                    circ = [first, last, last - first, num_turns, minor_axis, major_axis]
                    collision_stats.append(circ)
    print("Found ",len(collision_stats)," collisions.")
    # Next assess parameter distributions to set thresholds
    print("Fitting parameter distributions")
    popt = fit_Turns_dist([circ[3] for circ in collision_stats])
    Turns_min = popt[3] + p_Turns_min*popt[4]
    Turns_max = popt[3] + p_Turns_max*popt[4]
    
    popt = fit_Minor_dist([circ[4] for circ in collision_stats])
    Minor_min = popt[3] + p_Minor_min*popt[4]
    popt = fit_Major_dist([circ[5] for circ in collision_stats])
    Major_max = popt[3] + p_Major_max*popt[4]
    
    print("Parameters for this video:",Turns_min, Turns_max, Minor_min, Major_max)
    # Finally filter out false positives using the selected parameters
    for circ in collision_stats:
        num_turns = circ[3]
        minor_axis = circ[4]
        major_axis = circ[5]
        if num_turns >= Turns_min and num_turns <= Turns_max \
        and minor_axis >= Minor_min and major_axis <= Major_max:
        
            # make sure to only add it if it is not too soon:
            # either if it is the first video added
            # or if the start point is at least <min_frame> later
            if (not len(circle_stats)) or (circ[0] >= circle_stats[-1][0] + 0.1):
                circle_stats.append(circ)
                
    print("Found ",len(circle_stats)," circles.")
    print('\tSaving circle IDs for ', csvfile[csvfile.rfind('/')+1:-4]+"\n")
    circlecsv = pd.DataFrame(circle_stats, columns=['start','end','duration','turns','minor axis','major axis'])
    circlecsv.to_csv('./Circletimes_'+csvfile[csvfile.rfind('/')+1:-4]+".csv")
    print('Saved! Check the CSV in the same folder as this script.')
    