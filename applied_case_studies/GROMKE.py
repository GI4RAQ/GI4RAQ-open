#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:04:01 2019

@author: HelenPearce
"""
###############################################################################
# Replicating Gromke et al 2016 study as closely as possible

###############################################################################
# HOUSE-KEEPING
###############################################################################

# import necessary packages
import numpy as np
from scipy.linalg import solve
import math as math
from statistics import mean


# set container for error flags - this will be checked before final calculations
error = np.zeros((10,10))

###############################################################################
# for easily generating different combinations, only specify here and carried
# through code

# universal values - change as required
gi_height = 2.25
gi_height_filename = "225"
gi_obstruction_filename = "90"


# can only have 1 barrier - uncomment/comment as required - ensure values match
# those specified above 

#upwind
gi_upwind = 6.75
gi_upwind_obstruction = 90
gi_location_text = "upwind"
#gi_upwind = 0
#gi_upwind_obstruction = 0


#downwind
#gi_downwind = 29.25
#gi_downwind_obstruction = 90
#gi_location_text = "downwind"
gi_downwind = 0
gi_downwind_obstruction = 0




###############################################################################
# SET VALUES MANUALY
###############################################################################

# ___________ extract background pollution concentrations ___________
# only concerned with 1 pollutant
# Sulfur hexaflouride (SF6) used in Gromke et al 2016
# will be specified within code as NO2 just to avoid renaming lots of variables
# but is based on the molecular weight of SF6 rather than NO2

cB_no2 = 0 # UNIT: ug/m3 


# ___________ total street width ___________

roadw=36  

# ___________ climatological wind characteristics ___________

# wind speed left to right (m/s) 
# 4.65 m/s at building height (18m) specified, rather than using met station
# therefore altered the wind methodology to account for this


# ___________ set horizontal defining values ___________

# row_original = storage container initialised with zeros
row_original = np.array([0,0,0,0,0,0,0,0,0,0], dtype = float)

# left hand side building height
row_original[0]=18

# right hand side building height
row_original[1]=18

# hedge height
row_original[2] = gi_height

# proposed grey infra height
row_original[3] = 0
        
# tree base height at maturity
row_original[4] = 0

# tree top height at maturity
row_original[5] = 0

# existing barrier upwind height
row_original[8] = 0

# existing barrier downwind height
row_original[9] = 0


# ___________ set vertical defining values ___________

# zone_original = storage container
zone_original = np.array([0,0,0,0,0], dtype = float)

# upwind street boundary 
zone_original[0] = 2

# determine where emission zones start and end
ez1_start = 13
ez1_finish = 16
ez2_start = 20
ez2_finish = 23

ez1_w = 3 
ez2_w = 3

# determine number of emission zones
nez = 2
    
# determine upwind near edge and downwind far edge of emission zones 
zone_original[1] = 13   
zone_original[2] = 23

# downwind street boundary
zone_original[3] = 34

# downwind building
zone_original[4] = roadw 

# barrier locations (m from left of street)
# bar_original = a strage container
bar_original = np.array([0,0,0,0], dtype = float)

# existing barrier upwind
bar_original[0] = 0

# existing barrier downwind
bar_original[2] = 0

# new GI upwind
bar_original[1] = gi_upwind

# new GI downwind
bar_original[3] = gi_downwind


# ___________ obstruction values for barriers ___________

# obstruction array
obs_original = np.array([0,0,0,0], dtype = float)

# defined in terms of HOW MUCH THE AIR IS STOPPED 
# e.g. 80% obstruction = only 20% of air can get through barrier
 
# upwind existing
obs_original[0] = 0
obs_original[0] = obs_original[0]/100

# downwind existing
obs_original[2] = 0
obs_original[2] = obs_original[2]/100

# upwind new GI
obs_original[1] = gi_upwind_obstruction
obs_original[1] = obs_original[1]/100

# downwind new GI
obs_original[3] = gi_downwind_obstruction
obs_original[3] = obs_original[3]/100


# ___________ emissions ___________
# variable called no2 to save renaming all variables, but based on CF6 molecular weight
ez1_emis_no2 = 232.1
ez2_emis_no2 = 232.1

###############################################################################
# AUTOMATICALLY CALCULATED VALUES
###############################################################################
# check presence of barriers in locations (1 = true, 0 = false)
check_original = np.array([0,0,0,0], dtype = float)

# existing barrier upwind
if bar_original[0] > 0:
    check_original[0] = 1
else:
    check_original[0] = 0

# new barrier upwind
if bar_original[1] > 0:
    check_original[1] = 1
else:
    check_original[1] = 0

# existing barrier downwind 
if bar_original[2] > 0:
    check_original[2] = 1
else:
    check_original[2] = 0

# new barrier downwind 
if bar_original[3] > 0:
    check_original[3] = 1
else:
    check_original[3] = 0


# recirculation locations of barriers (m from left of street)
# rec_original = a storage container
rec_original = np.array([0,0,0,0,0], dtype = float)
recirc = 0

# upwind building recirc
# if there is building upwind the recirculation is calculated by 3H-3
# however if this is less than or equal to 0, 0.01 is assigned
# if there is no building upwind the recirc = 0 
# (done this way to avoid adding 0.01 as a recirc if there's no building)
if row_original[0] > 0:
    recirc = (row_original[0]*3)-3
    if recirc <= 0:
        recirc = 0.01    
    rec_original[0] = recirc
else:
    rec_original[0] = 0

# upwind existing barrier recirc
# reassign recirc to not carry over anything calculated previously
recirc = 0
# if there is an existing barrier upwind
if check_original[0] == 1:
    # the recirculation = 3H-3
    recirc = (row_original[8]*3)-3
    # if the above calculation results <= 0 a default of 0.01 is applied
    if recirc <= 0:
        recirc = 0.01
    # finally, vertical reference of recirculation touch-down is: 
    # the width of recirculation + the location of the barrier
    rec_original[1] = recirc+bar_original[0]
else:
    rec_original[1] = 0
    

# upwind new barrier recirc 
# reassign recirc to not carry over anything calculated previously
recirc = 0
# if there is a new barrier upwind
if check_original[1] == 1:
    # the recirculation = 3H-3
    recirc = (max(row_original[2:6])*3)-3
    # if the above calculation results <= 0 default of 0.01 is applied
    if recirc <= 0:
        recirc = 0.01
    # finally, vertical reference of recirculation touch-down is: 
    # the width of recirculation + the location of the barrier
    rec_original[2] = recirc+bar_original[1]
else:
    rec_original[2] = 0

# downwind existing barrier recirc
# reassign recirc to not carry over anything calculated previously
recirc = 0
# if there is an existing barrier downwind
if check_original[2] == 1:
    # the recirculation = 3H-3
    recirc = (row_original[9]*3)-3
    # if the above calculation results <= 0 default of 0.01 is applied
    if recirc <= 0:
        recirc = 0.01
    # finally, vertical reference of recirculation touch-down is: 
    # the width of recirculation + the location of the barrier
    rec_original[3] = recirc+bar_original[2]
else:
    rec_original[3] = 0

# downwind new barrier recirc 
# reassign recirc to not carry over anything calculated previously
recirc = 0
# if there is a new barrier downwind
if check_original[3] == 1:
    # the recirculation = 3H-3
    recirc = (max(row_original[2:6])*3)-3
    # if the above calculation resuls <= 0 default of 0.01 is applied
    if recirc <= 0:
        recirc = 0.01
    # finally, vertical reference of recirculation touch-down is: 
    # the width of recirculation + the location of the barrier
    rec_original[4] = recirc+bar_original[3]
else:
    rec_original[4] = 0



###############################################################################
# Make sure all values are rounded before progressing
    # otherwise when checking i x == x can erraneously report False

def rounding_row(row_input):
    row_input[0] = round(row_input[0], 4)
    row_input[1] = round(row_input[1], 4)
    row_input[2] = round(row_input[2], 4)
    row_input[3] = round(row_input[3], 4)
    row_input[4] = round(row_input[4], 4)
    row_input[5] = round(row_input[5], 4)
    row_input[8] = round(row_input[8], 4)
    row_input[9] = round(row_input[9], 4)
    return row_input

row_original = rounding_row(row_input=row_original)

def rounding_5(data):
    data[0] = round(data[0], 4)
    data[1] = round(data[1], 4)
    data[2] = round(data[2], 4)
    data[3] = round(data[3], 4)
    data[4] = round(data[4], 4)
    return data

zone_original = rounding_5(data=zone_original)
rec_original = rounding_5(data=rec_original)

def rounding_4(data):
    data[0] = round(data[0], 4)
    data[1] = round(data[1], 4)
    data[2] = round(data[2], 4)
    data[3] = round(data[3], 4)
    return data

bar_original = rounding_4(data=bar_original)

#print(bar_original, bar_mirror)
    
    
###############################################################################
# ROW DIMENSIONING
###############################################################################
# NOTE: ignore 0 index to avoid confusion with row 1, row 2, row 3

def row_dimensioning(row):
    # h = a storage container
    h = np.array([0,0,0,0], dtype = float)
    
    
    # first row height
    if row[5] > 0:
        # if there's a tree present, row height = crown top at maturity
        h[1] = row[5]
    elif row[2] > 0:
        # if there's a hedge, row height = height of barrier
        h[1] = row[2]
    elif row[3] > 0:
        # if there's a grey barrier, row height = height of barrier
        h[1] = row[3]
    else:
        # if there's no barrier, default of 2 m
        h[1] = 2
    
        
    # second and third row heights
    
    # if there are no buildings    
    if row[0] == 0 and row[1] == 0:
      error[0,1] = 1
      
    # if no lhs building
    elif row[0] == 0 and row[1] > 0:
      error[0,2] = 1
    
    # if no rhs building
    elif row[1] == 0 and row[0] > 0:
      error[0,2] = 1
    
    # if lhs building > rhs building
    elif row[0] > row[1]:
      # if GI is shorter than both buildings:
      if h[1] < row[0] and h[1] < row[1]:
          h[2] = row[1] - h[1]
          h[3] = row[0] - (h[1] + h[2])
      # if GI the same height or taller than either building:
      else:
          error[0,3] = 1
    
    
    # if rhs building > lhs building  
    elif row[1] > row[0]:
      # if GI is shorter than both buildings
      if h[1] < row[0] and h[1] < row[1]:
          h[2] = row[0] - h[1]
          h[3] = row[1] - (h[1] + h[2])
      # if GI the same height or taller than either building:
      else:
          error[0,3] = 1
    
    # if buildings are equal height     
    elif row[1] == row[0]:
      # if GI is shorter than the buildings
      if h[1] < row[0]:
          h[2] = (row[0] - h[1])/2
          h[3] = (row[1] - h[1])/2
      # if GI is the same height or taller than the buildings
      else:
          error[0,3] = 1
    
    # catch all:
    else:
      error[0,0] = 1
    
    
    # final check for existing barriers - must be shorter than buildings
    if row[8] >= min(row[0:2]):
        error[0,4] = 1
    
    elif row[9] >= min(row[0:2]):
        error[0,4] = 1
    
    h[1] = round(h[1], 4)
    h[2] = round(h[2], 4)
    h[3] = round(h[3], 4)
    
    return h

h_original = row_dimensioning(row = row_original)

#print(h_original)
#print(h_mirror)

###############################################################################
# COLUMN DIMENSIONING
###############################################################################
# NOTE: ignore 0 index to avoid confusion with row 1, row 2, row 3

def column_dimensioning(rec, zone, check, bar, row):
        
    # l = a storage container
    l = np.array([0,0,0,0,0,0], dtype = float)
    
    # 1) CASE 1: building recirc to upwind street boundary
    # 2) CASE 2: building recirc between upwind SB and near edge of emission zone
    # 3) CASE 3: building recirc in emission zone
    # 4) CASE 4: building recirc between downwind edge of emission zone and SB 
    # 5) CASE 5: building recirc between downwind SB and downwind building
    # 6) CASE 6: building recirc reaches downwind building (true street canyon)
    
    if rec[0] < zone[0]:
        # CASE 1
        # upwind
        if check[0] == 1:
            if rec[1] > zone[1]:
                l[1] = zone[0]
                l[2] = rec[1] - l[1]
            else:
                l[1] = zone[0]
                l[2] = zone[1] - l[1]
            if check[1] == 1:
                if bar[1] < rec[1]:
                    if max(row[2:6]) < row[8]:
                        l[1] = l[1]
                        l[2] = l[2]
                    else:
                        l[1] = bar[1]
                        if rec[2] > zone[1]:
                            l[2] = rec[2] - l[1]
                        else:
                            l[2] = zone[1] - l[1]
                elif bar[1] > rec[0]:
                    l[1] = bar[1]
                    if rec[2] > zone[1]:
                        l[2] = rec[2] - l[1]
                    else:
                        l[2] = zone[1] - l[1]
            elif check[1] == 0:
                l[1] = l[1]
                l[2] = l[2]
        elif check[0] == 0:
            l[1] = rec[0]
            l[2] = zone[1] - l[1]
            if check[1] == 1:
                l[1] = bar[1]
                if rec[2] > zone[1]:
                    l[2] = rec[2] - l[1]
                else:
                    l[2] = zone[1] - l[1]
            elif check[1] == 0:
                l[1] = l[1]
                l[2] = l[2]
        
        # downwind
        if l[2] > zone[2]-l[1]:
            error[1,1] = 1
        else:
            l[3] = zone[2] - l[1] - l[2]
            if check[2] == 1:
                if check[3] == 1:
                    l[3] = bar[3] - l[1] - l[2]
                    if rec[4] >= zone[3]:
                        if max(row[2:6]) < row[9]:
                            l[4] = zone[3] - l[1] - l[2] - l[3]
                        else:
                            if rec[4] >= zone[4]:
                                l[4] = zone[3] - l[1] - l[2] - l[3]
                            else:
                                l[4] = rec[4] - l[1] - l[2] - l[3]
                    else:
                        l[4] = rec[4] - l[1] - l[2] - l[3]
                else:
                    l[4] = zone[3] - l[1] - l[2] - l[3]
            
            elif check[2] == 0:
                if check[3] == 1:
                    if rec[4] >= zone[4]:
                        l[3] = zone[2] - l[1] - l[2]
                        l[4] = bar[3] - l[1] - l[2] - l[3]
                    else:
                        l[3] = bar[3] - l[1] - l[2]
                        l[4] = rec[4] - l[1] - l[2] - l[3]
                else:
                    l[4] = zone[3] - l[1] - l[2] - l[3]
        
        # final check
        if l[4] > zone[4]-l[1]-l[2]-l[3]:
            error[1,2] = 1
        else:
            l[5] = zone[4] - l[1] - l[2] - l[3] - l[4]
                    
    elif rec[0] > zone[0] and rec[0] < zone[1]:
        # CASE 2
        # upwind: yes, no
        if check[0] == 1 and row[8] < row[0]:
            l[1] = rec[0]
            l[2] = zone[1] - l[1]
            if check[1] == 1:
                if bar[1] <= rec[0]:
                    if max(row[2:6]) < row[0]:
                        l[1] = l[1]
                        l[2] = l[2]
                    else:
                        l[1] = bar[1]
                        if rec[2] > zone[1]:
                            l[2] = rec[2] - l[1]
                        else:
                            l[2] = zone[1] - l[1]
                else:
                    l[1] = bar[1]
                    if rec[2] > zone[1]:
                        l[2] = rec[2] - l[1]
                    else:
                        l[2] = zone[1] - l[1]
            else:
                l[1] = l[1]
                l[2] = l[2]
        
        # upwind: no
        elif check[0] == 0:
            l[1] = rec[0]
            l[2] = zone[1] - l[1]
            if check[1] == 1:
                if bar[1] <= rec[0]:
                    if max(row[2:6]) < row[0]:
                        l[1] = l[1]
                        l[2] = l[2]
                    else:
                        l[1] = bar[1]
                        if rec[2] > zone[1]:
                            l[2] = rec[2] - l[1]
                        else:
                            l[2] = zone[1] - l[1]
                else:
                    l[1] = bar[1]
                    if rec[2] > zone[1]:
                        l[2] = rec[2] - l[1]
                    else:
                        l[2] = zone[1] - l[1]
            else:
                l[1] = l[1]
                l[2] = l[2]
        
        # upwind: yes, yes    
        elif check[0] == 1 and row[8] > row[0]:
            l[1] = bar[0]
            if rec[1] > zone[1]:
                l[2] = rec[1] - l[1]
            else:
                l[2] = zone[1] - l[1]
            if check[1] == 1:
                if bar[1] <= rec[1]:
                    if max(row[2:6]) < row[8]:
                        l[1] = l[1]
                        l[2] = l[2]
                    else:
                        l[1] = bar[1]
                        if rec[2] > zone[1]:
                            l[2] = rec[2] - l[1]
                        else:
                            l[2] = zone[1] - l[1]
                else:
                    l[1] = bar[1]
                    if rec[2] > zone[1]:
                        l[2] = rec[2] - l[1]
                    else:
                        l[2] = zone[1] - l[1]
            else:
                l[1] = l[1]
                l[2] = l[2]
                
        # downwind
        if l[2] > zone[2]-l[1]:
            error[2,1] = 1
        else:
            l[3] = zone[2] - l[1] - l[2]
            if check[2] == 1:
                if check[3] == 1:
                    l[3] = bar[3] - l[1] - l[2]
                    if rec[4] > zone[3]:
                        if max(row[2:6]) < row[9]:
                            l[4] = zone[3] - l[1] - l[2] - l[3]
                        else:
                            if rec[4] >= zone[4]:
                                l[4] = zone[3] - l[1] - l[2] - l[3]
                            else:
                                l[4] = rec[4] - l[1] - l[2] - l[3]
                    else:
                        l[4] = rec[4] - l[1] - l[2] - l[3]
                else:
                    l[4] = zone[3] - l[1] - l[2] - l[3]
            else:
                if check[3] == 1:
                    if rec[4] >= zone[4]:
                        l[4] = bar[3] - l[1] - l[2] - l[3]
                    else:
                        l[3] = bar[3] - l[1] - l[2]
                        l[4] = rec[4] - l[1] - l[2] - l[3]
                else:
                    l[4] = zone[3] - l[1] - l[2] - l[3]
                    
        # final check
        if l[4] > zone[4]-l[1]-l[2]-l[3]:
            error[2,2] = 1
        else:
            l[5] = zone[4] - l[1] - l[2] - l[3] - l[4]
        
    elif rec[0] > zone[1] and rec[0] < zone[2]:
        # CASE 3
        # upwind (no)
        if check[0] == 0:
            l[1] = zone[0]
            l[2] = rec[0] - l[1]
            if check[1] == 1:
                if max(row[2:6]) < row[0]:
                    if check[0] == 1:
                        l[2] = bar[1] - l[1]
                        l[3] = rec[0] - l[1] - l[2]
                    else:
                        l[1] = bar[1]
                        l[2] = rec[0] - l[1]
                        l[3] = zone[2] - l[1] - l[2]
                else:
                    if rec[2] > zone[2]:
                        error[3,2] = 1
                    else:
                        if check[0] == 1:
                            l[2] = bar[1] - l[1]
                            l[3] = rec[2] - l[1] - l[2]
                        else:
                            l[1] = bar[1]
                            l[2] = rec[2] - l[1]
                            l[3] = zone[2] - l[1] - l[2]
            else:
                l[1] = l[1]
                l[2] = l[2]
                l[3] = zone[2] - l[1] - l[2]
            
        # upwind (yes, no)
        if check[0] == 1 and row[8] < row[0]:
            l[1] = zone[0]
            l[2] = rec[0] - l[1]
            if check[1] == 1:
                if max(row[2:6]) < row[0]:
                    if check[0] == 1:
                        l[2] = bar[1] - l[1]
                        l[3] = rec[0] - l[1] - l[2]
                    else:
                        l[1] = bar[1]
                        l[2] = rec[0] - l[1]
                        l[3] = zone[2] - l[1] - l[2]
                else:
                    if rec[2] > zone[2]:
                        error[3,2] = 1
                    else:
                        if check[0] == 1:
                            l[2] = bar[1] - l[1]
                            l[3] = rec[2] - l[1] - l[2]
                        else:
                            l[1] = bar[1]
                            l[2] = rec[2] - l[1]
                            l[3] = zone[2] - l[1] - l[2]
            else:
                l[1] = l[1]
                l[2] = l[2]
                l[3] = zone[2] - l[1] - l[2]
                
        # upwind (yes, yes)
        if check[0] == 1 and row[8] > row[0]:
            l[1] = bar[0]
            if rec[1] > zone[2]:
                error[3,1] = 1
            else:
                l[2] = rec[1] - l[1]
                if check[1] == 1:
                    if max(row[2:6]) < row[8]:
                        l[2] = bar[1] - l[1]
                        if rec[1] > zone[1]:
                            l[3]= rec[1] - l[1] - l[2]
                        else:
                            l[3] = zone[1] - l[1] - l[2]
                    else:
                        l[2] = bar[1]
                        if rec[2] > zone[2]:
                            error[3,3] = 1
                        else:
                            if rec[2] > zone[1]:
                                l[3] = rec[2] - l[1] - l[2]
                            else:
                                l[3] = zone[1] - l[1] - l[2]
                else:
                    l[1] = l[1]
                    l[2] = l[2]
                    l[3] = zone[2] - l[1] - l[2]
        
        # downwind
        if l[3] == zone[2]-l[1]-l[2]:
            if check[2] == 1:
                if check[3] == 1:
                    l[3] = bar[3] - l[1] - l[2]
                    if rec[4] > zone[3]:
                        if max(row[2:6]) < row[9]:
                            l[4] = zone[3] - l[1] - l[2] - l[3]
                        else:
                            if rec[4] >= zone[4]:
                                l[4] = zone[3] - l[1] - l[2] - l[3]
                            else:
                                l[4] = rec[4] - l[1] - l[2] - l[3]
                    else:
                        l[4] = rec[4] - l[1] - l[2] - l[3]
                else:
                    l[4] = zone[3] - l[1] - l[2] - l[3]
            else:
                if check[3] == 1:
                    if rec[4] >= zone[4]:
                        l[4] = bar[3] - l[1] - l[2] - l[3]
                    else:
                        l[3] = bar[3] - l[1] - l[2]
                        l[4] = rec[4] - l[1] - l[2] - l[3]
                else:
                    l[4] = zone[3] - l[1] - l[2] - l[3]
        else:
            l[4] = zone[3] - l[1] - l[2] - l[3]
            
        # final check
        if l[4] > zone[4]-l[1]-l[2]-l[3]:
            error[3,4] = 1
        else:
            l[5] = zone[4] - l[1] - l[2] - l[3] - l[4]
            
    elif rec[0] > zone[2] and rec[0] < zone[3]:
        # CASE 4
        # upwind (yes yes)
        if check[0] == 1 and row[8] > row[0]:
                error[4,1] = 1
                   
        # upwind (no) or yes,no
        else:
            l[1] = zone[0]
            l[2] = rec[0] - l[1]
            
        if check[1] == 1 and max(row[2:6]) > row[0]:
            error[4,2] = 1
            
        if check[1] == 1 and max(row[2:6]) < row[0] and check[0] == 0:
            l[1] = bar[1]
            l[2] = rec[0] - l[1]
            
        # downwind
        if check[2] == 1:
            if check[3] == 1:
                if bar[3] <= rec[0]:
                    l[2] = bar[3] - l[1]
                    l[3] = rec[0] - l[1] - l[2]
                    l[4] = zone[3] - l[1] - l[2] - l[3]
                else:
                    l[3] = bar[3] - l[1] - l[2]
                    l[4] = zone[3] - l[1] - l[2] - l[3]
            else:
                l[2] = zone[2] - l[1]
                l[3] = rec[0] - l[1] - l[2]
                l[4] = zone[3] - l[1] - l[2] - l[3]
        else:
            if check[3] == 1:
                if bar[3] <= rec[0]:
                    l[2] = bar[3] - l[1]
                    l[3] = rec[0] - l[1] - l[2]
                    l[4] = zone[3] - l[1] - l[2] - l[3]
                else:
                    if rec[4] >= zone[4]:
                        l[2] = zone[2] - l[1]
                        l[3] = rec[0] - l[1] - l[2]
                        l[4] = bar[3] - l[1] - l[2] - l[3]
                    else:
                        l[3] = bar[3] - l[1] - l[2]
                        l[4] = rec[4] - l[1] - l[2] - l[3]
            else:
                l[2] = zone[2] - l[1]
                l[3] = rec[0] - l[1] - l[2] - l[3]
                l[4] = zone[3] - l[1] - l[2] - l[3]
                
        # case 4.4.1
        if check[1] == 1 and max(row[2:6]) < row[0] and check[0] == 1:
            l[1] = zone[0]
            l[2] = bar[1] - l[1]
            l[3] = rec[0] - l[1] - l[2]
            l[4] = zone[3] - l[1] - l[2] - l[3]
      
        # and finally..
        l[5] = zone[4] - l[1] - l[2] - l[3] - l[4]
    
    elif rec[0] > zone[3] and rec[0] < zone[4]:
        # CASE 5
        if check[0] == 1:
            if row[8] > row[0]:
                error[5,1] = 1
            else:
                l[1] = zone[0]
        else:
            l[1] = zone[0]
            
        if check[1] == 1:
            if max(row[2:6]) < row[0]:
                if check[0] == 1:
                    l[2] = bar[1] - l[1]
                    if check[2] == 1:
                        l[3] = zone[3] - l[1] - l[2]
                        l[4] = rec[0] - l[1] - l[2] - l[3]
                    else:
                        l[3] = zone[2] - l[1] - l[2]
                        l[4] = rec[0] - l[1] - l[2] - l[3]
                else:
                    l[1] = bar[1]
                    if check[2] == 1:
                        if check[3] == 1:
                            l[2] = bar[3] - l[1]
                            l[3] = zone[3] - l[1] - l[2]
                            l[4] = rec[0] - l[1] - l[2] - l[3]
                        else:
                            l[2] = zone[2] - l[1]
                            l[3] = zone[3] - l[1] - l[2]
                            l[4] = rec[0] - l[1] - l[2] - l[3]
                    else:
                        if check[3] == 1:
                            l[2] = bar[3] - l[1]
                            l[3] = zone[3] - l[1] - l[2]
                            l[4] = rec[0] - l[1] - l[2] - l[3]
                        else:
                            l[2] = zone[2] - l[1]
                            l[3] = zone[3] - l[1] - l[2]
                            l[4] = rec[0] - l[1] - l[2] - l[3]
            else:
                error[5,2] = 1
        else:
            l[1] = l[1]
            if check[2] == 1:
                if check[3] ==1:
                    l[2] = bar[3] - l[1]
                    l[3] = zone[3] - l[1] - l[2]
                    l[4] = rec[0] - l[1] - l[2] - l[3]
                else:
                    l[2] = zone[2] - l[1]
                    l[3] = zone[3] - l[1] - l[2]
                    l[4] = rec[0] - l[1] - l[2] - l[3]
            else:
                if check[3] == 1:
                    l[2] = bar[3] - l[1]
                    l[3] = zone[3] - l[1] - l[2]
                    l[4] = rec[0] - l[1] - l[2] - l[3]
                else:
                    l[2] = zone[2] - l[1]
                    l[3] = zone[3] - l[1] - l[2]
                    l[4] = rec[0] - l[1] - l[2] - l[3]
        
        # finally...
        l[5] = zone[4] - l[1] - l[2] - l[3] - l[4]
                    
         
    elif rec[0] >= zone[4]:
        # CASE 6
        l[1] = zone[0]
        if check[1] == 1:
            l[2] = bar[1] - l[1]
        else:
            l[2] = zone[1] - l[1]
            
        if check[3] == 1:
            l[3] = bar[3] - l[1] - l[2]
        else:
            l[3] = zone[2] - l[1] - l[2]
        
        l[4] = zone[3] - l[1] - l[2] - l[3]
        l[5] = zone[4] - l[1] - l[2] - l[3] - l[4]
    
    # add in for no new barriers:
    elif check[1] == 0 and check[3] == 0:
        l[1] = zone[0]
        l[2] = zone[1] - l[1]
        l[3] = zone[2] - l[1] - l[2]
        l[4] = zone[3] - l[1] - l[2] - l[3]
        l[5] = zone[4] - l[1] - l[2] - l[3] - l[4]
    
    l[1] = round(l[1], 4)
    l[2] = round(l[2], 4)
    l[3] = round(l[3], 4)
    l[4] = round(l[4], 4)
    l[5] = round(l[5], 4)
    
        
    return l

l_original = column_dimensioning(rec=rec_original, zone=zone_original, 
                                 check=check_original, bar=bar_original, 
                                 row=row_original)
    
#print(l_original)

# check that all the columns add up to total road width
if round(sum(l_original),4) != roadw:
    error[7,1] = 1

# ___________ format column and row output ___________

# convert to l and h to lists, otherwise you get the error: 
# Object of type 'ndarray' is not JSON serializable

# these provide a list of widths/heights of individual columns/rows
l_original_list = l_original.tolist()
h_original_list = h_original.tolist()


# calculate cumulative distance in m from left / bottom:
l_cumu_original = np.array([0,0,0,0,0,0], dtype = float)
l_cumu_original[1] = round(l_original[1], 4)
l_cumu_original[2] = round(l_original[1]+l_original[2], 4)
l_cumu_original[3] = round(l_original[1]+l_original[2]+l_original[3], 4)
l_cumu_original[4] = round(l_original[1]+l_original[2]+l_original[3]+l_original[4], 4)
l_cumu_original[5] = round(l_original[1]+l_original[2]+l_original[3]+l_original[4]+l_original[5], 4)
l_cumu_original_list = l_cumu_original.tolist()

h_cumu_original = np.array([0,0,0,0], dtype = float)
h_cumu_original[1] = round(h_original[1],4)
h_cumu_original[2] = round(h_original[1]+h_original[2],4)
h_cumu_original[3] = round(h_original[1]+h_original[2]+h_original[3],4)
h_cumu_original_list = h_cumu_original.tolist()

#print(l_cumu_original_list)
#print(h_cumu_original_list)
###############################################################################
# ADVECTION & DISPERSION PATTERNS
###############################################################################


# ___________ SCALE obstruction values for barriers to row heights ___________
        
# this will need looking at again should we change the rules around heights
        # of barriers (e.g. GI has to be less than the building heights);
        # currently assumed any barrier is WITHIN FIRST ROW - existing & new
        
# upwind existing
obs_original[0] = obs_original[0]*(row_original[8]/h_original[1])

# downwind existing
obs_original[2] = obs_original[2]*(row_original[9]/h_original[1])

# upwind new
obs_original[1] = obs_original[1]*(max(row_original[2:6])/h_original[1])

# downwind new
obs_original[3] = obs_original[3]*(max(row_original[2:6])/h_original[1])



# if any barriers have 100% coverage, change obstruction to 0.99 otherwise solver 
# issues occur because of knock-on effect on advection and dispersion patterns;
# now dispersion is calculated from advection values (adv*0.1)
# but if obs=1, advection will = 0 in some places and throw out the calculations

for i in range(len(obs_original)):
    if obs_original[i]==1:
        obs_original[i]=0.99
    else:
        obs_original[i]=obs_original[i]
        

# ___________ determine how many columns are FULLY spanned by a recirc  ___________

full_rec = 0

def recirc_col(recirc, l_cumu):
    if recirc < l_cumu[1]:
        full_rec = 0
        return full_rec
    elif recirc == l_cumu[1]:
        full_rec = 1
        return full_rec
    elif recirc > l_cumu[1] and recirc < l_cumu[2]:
        full_rec = 1
        return full_rec
    elif recirc == l_cumu[2]:
        full_rec = 2
        return full_rec
    elif recirc > l_cumu[2] and recirc < l_cumu[3]:
        full_rec = 2
        return full_rec
    elif recirc == l_cumu[3]:
        full_rec = 3
        return full_rec
    elif recirc > l_cumu[3] and recirc < l_cumu[4]:
        full_rec = 3
        return full_rec
    elif recirc == l_cumu[4]:
        full_rec = 4
        return full_rec
    elif recirc > l_cumu[4] and recirc < l_cumu[5]:
        full_rec = 4
        return full_rec
    elif recirc == l_cumu[5]:
        full_rec = 5
        return full_rec
    elif recirc > l_cumu[5]:
        full_rec = 5
        return full_rec
  
rec_ncol_orig = 0
rec_nrow_orig = 0
rec_ncol_mir = 0
rec_nrow_mir = 0

# if the upwind building is aligned with row 2:
if row_original[0] == h_cumu_original[2]:
    rec_nrow_orig = 2
    rec_ncol_orig = recirc_col(recirc = rec_original[0], l_cumu = l_cumu_original)
# if the upwind building is aligned with row 3:
elif row_original[0] == h_cumu_original[3]:
    # ft is reduced; flow within canyon is slower than background wind speed
    rec_nrow_orig = 3
    rec_ncol_orig = recirc_col(recirc = rec_original[0], l_cumu = l_cumu_original)
    


# ___________ dynamic assignment of wind speeds within canyon ___________
    


# H = upwind building height
H_orig = row_original[0]

# w = street width
w = roadw

# function to determine wind speed (u) at any height (z)
def ws_point(ubg,z,H,w):
     # firstly determine value of d based on street dimensions
    if w <= (1.5*H):
        d = (0.7*H)
    elif w > (1.5*H) and w <= (5*H):
        d = H-(0.2*w)
    else:
        d = 0
    
    # calculate wind speed (u) at the height of upwind building (h)
    #uh = ubg*(math.log(5000)/math.log(500))*(math.log(5*H-5*d)/math.log(500-5*d))
    # this is directly provided
    uh = 4.65
    min_u = 0.1*uh
    
    if d >= z:
        uz = min_u
    else:
        uz = ubg*(math.log(5000)/math.log(500))*(math.log(5*z-5*d)/math.log(500-5*d))
                                                    
    # check and ensure minimum of 0.1*h
    if uz < min_u:
        u = min_u
    else:
        u = uz
    
    return u


# function to determine average wind speed (u) across a row
def ws_average(row_min, row_max, ubg, H):
    dif = row_max - row_min
    dif9 = dif/9
    
    u1 = ws_point(ubg=ubg, z = row_min, H=H, w=w)
    u2 = ws_point(ubg=ubg, z = row_min+(1*dif9), H=H, w=w)
    u3 = ws_point(ubg=ubg, z = row_min+(2*dif9), H=H, w=w)
    u4 = ws_point(ubg=ubg, z = row_min+(3*dif9), H=H, w=w)
    u5 = ws_point(ubg=ubg, z = row_min+(4*dif9), H=H, w=w)
    u6 = ws_point(ubg=ubg, z = row_min+(5*dif9), H=H, w=w)
    u7 = ws_point(ubg=ubg, z = row_min+(6*dif9), H=H, w=w)
    u8 = ws_point(ubg=ubg, z = row_min+(7*dif9), H=H, w=w)
    u9 = ws_point(ubg=ubg, z = row_min+(8*dif9), H=H, w=w)
    u10 = ws_point(ubg=ubg, z = row_max, H=H, w=w)
    
    data = (u1,u2,u3,u4,u5,u6,u7,u8,u9,u10)
    avg_u = mean(data)
    
    return avg_u
        
# calculate ubg by working back from known speed at Uh
ubg_orig = 4.65/((math.log(5000)/math.log(500))*(math.log(5*18-5*10.8)/math.log(500-5*10.8)))
#print(ubg_orig)
# check this results in a Uh of 4.65
#Uh_orig = ws_point(ubg = ubg_orig, z = 18, H = 18, w = w)
#print(Uh_orig)

U3_orig = ws_average(row_min = h_cumu_original[2], row_max = h_cumu_original[3], ubg = ubg_orig, H = H_orig)
U2_orig = ws_average(row_min = h_cumu_original[1], row_max = h_cumu_original[2], ubg = ubg_orig, H = H_orig)
U1_orig = ws_average(row_min = 0, row_max = h_cumu_original[1], ubg = ubg_orig, H = H_orig)
Uh_orig = ws_point(ubg = ubg_orig, z = H_orig, H = H_orig, w = w)
Ur_orig = (0.1*Uh_orig)*(H_orig/(2*h_original[rec_nrow_orig]))
#print("Ur original:",Ur_orig)
Ut_orig = ws_point(ubg = ubg_orig,z=max(row_original[0],row_original[1]),H = H_orig, w = w)



# ___________ prepare empty containers ___________

# base patterns: left to right wind
ua1_orig = np.zeros((5,6)) 	# horizontal advection velocities
wa1_orig = np.zeros((5,6))   # vertical advection velocities
ue1_orig = np.zeros((5,6))   # horizontal dispersion velocities
we1_orig = np.zeros((5,6))   # vertical dispersion velocities



# ___________ Advection & Dispersion Assignment: NO BARRIERS ___________

def no_barriers_pattern(row, h_cumu, rec_ncol, wa1, we1, ua1, ue1, U1, U2, U3, Ut, Uh, Ur, l, h,):
    # assign air flow outside of recirc
    # upwind building covers 2 rows
    if row[0] == h_cumu[2]:
        # recirc region covers less than 4 columns
        if rec_ncol < 4:
            if rec_ncol == 0:
                wa1[4,1] = -(((U1*h[1])/l[1])+((U2*h[2])/l[1])+((U3*h[3])/l[1]))
            else:
                wa1[4,1] = -((U3*h[3])/l[1])
                wa1[4,rec_ncol+1] = -(((U1*h[1])/l[rec_ncol+1])+((U2*h[2])/l[rec_ncol+1]))
            
            # assign downwards wind to column after recirc zone
            wa1[2,rec_ncol+1] = -((U1*h[1])/l[rec_ncol+1])
            wa1[3,rec_ncol+1] = -(((U2*h[2])/l[rec_ncol+1])+abs(wa1[2,rec_ncol+1]))
            
            # dispersion for vertical down
            we1[4,1] = abs(wa1[4,1])*0.1
            we1[3,rec_ncol+1] = abs(wa1[3,rec_ncol+1])*0.1
            we1[2,rec_ncol+1] = abs(wa1[2,rec_ncol+1])*0.1
            
            #assign mirror upwards wind in column 5
            wa1[2,5] = ((U1*h[1])/l[5])
            wa1[3,5] = ((U2*h[2])/l[5]) + wa1[2,5]
            wa1[4,5] = ((U3*h[3])/l[5]) + wa1[3,5]
            
            # dispersion for vertical up
            we1[4,5] = abs(wa1[4,5])*0.1
            we1[3,5] = abs(wa1[3,5])*0.1
            we1[2,5] = abs(wa1[2,5])*0.1
            
            # vertical dispersion in columns between up and down flows
            if rec_ncol < 3:
                we1[3,(rec_ncol+2):5] = (abs(wa1[3,rec_ncol+1])+abs(wa1[3,5]))/2*0.1
                we1[2,(rec_ncol+2):5] = (abs(wa1[2,rec_ncol+1])+abs(wa1[2,5]))/2*0.1
                
            # vertical dispersion at top (above recirc - only 2 rows covered)
            # driven by exchanges with air above canyon
            we1[4,2:5] = 0.1*Ut
            # apart from the box with the addition input for U1 and U2
            we1[4,rec_ncol+1] = abs(wa1[4,rec_ncol+1])*0.1
            
            # horizontal
            # assign horizontal advection for f1,f2 and f3
            ua1[3,2:] = U3
            ua1[2,(rec_ncol+2):] = U2
            ua1[1,(rec_ncol+2):] = U1
            
            # horizontal dispersion
            ue1[1,(rec_ncol+2):] = abs(ua1[1,(rec_ncol+2):])*0.1
            ue1[2,(rec_ncol+2):] = abs(ua1[2,(rec_ncol+2):])*0.1
            ue1[3,:] = abs(ua1[3,:])*0.1
            
        elif rec_ncol >= 4:
            # cases 5 & 6
            wa1[4,1] = -((U3*h[3])/l[1])
            ua1[3,2:] = U3
            wa1[4,5] = ((U3*h[3])/l[5])
            
            # dispersion at the top of boxes
            we1[4,:] = 0.1*Ut
            we1[4,1] = abs(wa1[4,1])*0.1
            we1[4,5] = abs(wa1[4,5])*0.1
            
            # horizontal dispersion
            ue1[3,2:] = abs(ua1[3,2:])*0.1
            
            if rec_ncol == 4:
                # vertical dispersion for slack regions
                we1[2,5] = abs(wa1[4,5])*0.1
                we1[3,5] = abs(wa1[4,5])*0.1
                
                
            
    # if the upwind building covers 3 rows
    elif row[0] == h_cumu[3]:
        if rec_ncol < 4:
            # downward: flow into canyon
            wa1[2,rec_ncol+1] = -((U1*h[1])/l[rec_ncol+1])
            wa1[3,rec_ncol+1] = -(((U2*h[2])/l[rec_ncol+1])+abs(wa1[2,rec_ncol+1]))
            wa1[4,rec_ncol+1] = -(((U3*h[3])/l[rec_ncol+1])+abs(wa1[3,rec_ncol+1]))
            
            # dispersion for vertical down
            we1[4,rec_ncol+1] = abs(wa1[4,rec_ncol+1])*0.1
            we1[3,rec_ncol+1] = abs(wa1[3,rec_ncol+1])*0.1
            we1[2,rec_ncol+1] = abs(wa1[2,rec_ncol+1])*0.1
            
            # upwards: flow out of canyon
            wa1[2,5] = ((U1*h[1])/l[5])
            wa1[3,5] = ((U2*h[2])/l[5]) + wa1[2,5]
            wa1[4,5] = ((U3*h[3])/l[5]) + wa1[3,5]
            
            # dispersion for vertical up
            we1[2,5] = wa1[2,5]*0.1
            we1[3,5] = wa1[3,5]*0.1
            we1[4,5] = wa1[4,5]*0.1
            
            # vertical dispersion in between advection flows up and down
            if rec_ncol < 3:
                we1[4,(rec_ncol+2):5] = 0.1*Uh
                we1[3,(rec_ncol+2):5] = (we1[3,rec_ncol+1]+we1[3,5])/2
                we1[2,(rec_ncol+2):5] = (we1[2,rec_ncol+1]+we1[2,5])/2
            
            # horizontal flows
            ua1[3,(rec_ncol+2):] = U3
            ua1[2,(rec_ncol+2):] = U2
            ua1[1,(rec_ncol+2):] = U1
            
            # horizontal dispersion
            ue1[3,(rec_ncol+2):] = abs(ua1[3,(rec_ncol+2):])*0.1
            ue1[2,(rec_ncol+2):] = abs(ua1[2,(rec_ncol+2):])*0.1
            ue1[1,(rec_ncol+2):] = abs(ua1[1,(rec_ncol+2):])*0.1
            
        # else - for cases 5 & 6 there are no flows outside the recirc zone
        elif rec_ncol == 4:
            # no horizontal dispersion but vertical dispersion based on ACH
            # no advection flows outside of recirc
            we1[2,5] = 0.1*Uh
            we1[3,5] = 0.1*Uh
            we1[4,5] = 0.1*Uh
    
    # flows within recirc region
    if row[0] == h_cumu[2]:
        if rec_ncol >= 2:
            # advection: top row horizontal
            ua1[2,2:(rec_ncol+1)] = Ur
            
            # advection: vertical drop down
            wa1[2,rec_ncol] = -((Ur*h[2])/l[rec_ncol])
            
            # advection: bottom row horizontal
            ua1[1,2:(rec_ncol+1)] = -((abs(wa1[2,rec_ncol])*l[rec_ncol])/h[1])
            
            # advection: vertical up
            wa1[2,1] = (abs(ua1[1,2])*h[1])/l[1]
            
            # horizontal dispersion
            ue1[2,2:(rec_ncol+1)] = abs(ua1[2,2:(rec_ncol+1)])*0.1
            ue1[1,2:(rec_ncol+1)] = abs(ua1[1,2:(rec_ncol+1)])*0.1
            
            # vertical dispersion = 10% of average up and down velocities
            # vertical dispersion within recirc
            we1[2,1] = abs(wa1[2,1])*0.1
            we1[2,rec_ncol] = abs(wa1[2,rec_ncol])*0.1
            we1[2,2:rec_ncol] =  (we1[2,1]+we1[2,rec_ncol])/2
            #we1[2,1:(rec_ncol+1)] = ((abs(wa1[2,1])+abs(wa1[2,rec_ncol]))/2)*0.1
            
        elif rec_ncol == 1:
            we1[2,1] = 0.1*Ur
            
    elif row[0] == h_cumu[3]:
        if rec_ncol >= 2:
            # advection: top row horizontal
            ua1[3,2:(rec_ncol+1)] = Ur
            
            # vertical advection: drop down
            wa1[3,rec_ncol] = -((Ur*h[3])/l[rec_ncol])
            wa1[2,rec_ncol] = wa1[3,rec_ncol]
            
            # advection: bottom row horizontal
            ua1[1,2:(rec_ncol+1)] = -((abs(wa1[2,rec_ncol])*l[rec_ncol])/h[1])
            
            # vertical advection: up
            wa1[2,1] = (abs(ua1[1,2])*h[1])/l[1]
            wa1[3,1] = wa1[2,1]
            
            
            # horizontal dispersion
            ue1[3,2:(rec_ncol+1)] = abs(ua1[3,2:(rec_ncol+1)])*0.1
            ue1[1,2:(rec_ncol+1)] = abs(ua1[1,2:(rec_ncol+1)])*0.1
            
            # horizontal dispersion through slack middle = average of top and bottom
            ue1[2,2:(rec_ncol+1)] = (ue1[3,2:(rec_ncol+1)]+ue1[1,2:(rec_ncol+1)])/2
            
            
            # vertical dispersion
            # default each row to average of up/down velocities *0.1
            we1[2,1:(rec_ncol+1)] = ((abs(wa1[2,rec_ncol])+abs(wa1[2,1]))/2)*0.1
            we1[3,1:(rec_ncol+1)] = ((abs(wa1[3,rec_ncol])+abs(wa1[3,1]))/2)*0.1
            
            # then specify each edge driven by local advection
            we1[2,rec_ncol] = abs(wa1[2,rec_ncol])*0.1
            we1[3,rec_ncol] = abs(wa1[3,rec_ncol])*0.1
            
            we1[2,1] = abs(wa1[2,1])*0.1
            we1[3,1] = abs(wa1[3,1])*0.1
        
        elif rec_ncol == 1:
            we1[2,1] = 0.1*Ur
            we1[3,1] = 0.1*Ur
        
    
    
    
    # decrease dispersion at edge of recirc
    dis = 0.05*Uh

    
    if rec_ncol > 0 and rec_ncol < 5:
        if row[0] == h_cumu[2]:
            ue1[1,(rec_ncol+1)] = dis
            ue1[2,(rec_ncol+1)] = dis
            we1[3,1:(rec_ncol+1)] = dis
        elif row[0] == h_cumu[3]:
            ue1[1:4,(rec_ncol+1)] = dis
            we1[4,1:(rec_ncol+1)] = dis
    elif rec_ncol == 5:
        if row[0] == h_cumu[2]:
            we1[3,1:] = dis
        elif row[0] == h_cumu[3]:
            we1[4,1:] = dis
        
        
    
    # assign values that should be 0 by default
    ue1[:,0] = 0
    ue1[:,1] = 0 # to lhs building
    ue1[0,:] = 0 # all columns within 'ground'
    ue1[4,:] = 0 # horizontal exhanges above street level
    
    we1[0,:] = 0
    we1[1,:] = 0
    we1[:,0] = 0
        
    return ue1, ua1, we1, wa1        

no_barriers_orig = no_barriers_pattern(row=row_original, h_cumu=h_cumu_original, 
                                rec_ncol=rec_ncol_orig, wa1=wa1_orig, we1=we1_orig, 
                                ua1=ua1_orig, ue1=ue1_orig, U1=U1_orig, U2=U2_orig, 
                                U3=U3_orig, Ut=Ut_orig, Uh=Uh_orig, Ur=Ur_orig, 
                                l=l_original, h=h_original)
ue1_orig = no_barriers_orig[0]
ua1_orig = no_barriers_orig[1]
we1_orig = no_barriers_orig[2]
wa1_orig = no_barriers_orig[3]

# print("ue1_orig")
# print(ue1_orig)
# print("ua1_orig")
# print(ua1_orig)
# print("we1_orig")
# print(we1_orig)
# print("wa1_orig")
# print(wa1_orig)



# ___________ Advection & Dispersion Assignment: FUNCTIONS for barriers ___________

# 1) ----- define functions -----
# 1a) flow effected by barrier placed outside the recirculation zone

def bar_outside(obs, bar_col, bar_recirc, ue, ua, we, wa, h, l):
    # dispersion at barrier: reduce dispersion that it currently is by barrier effect
    #ue[1,(bar_col+1)] = (1-obs)*ue[1,(bar_col+1)]
    # switched off the above as this is accounted for by reduced advection term,
    # and dispersion is then calculated as 10% of this
    
    # calculate change in U1 (i.e. the amount slowed down)
    # altered this so it doesn't refer back to the constant of U1 but instead
    # to the current advection term across boundary that barrier is on
    delta_u1 = obs*ua[1,bar_col+1]

    # slow down U1' (vertical) by adding the remainder of mass flux
    delta_u1_1 = (delta_u1*h[1])/l[bar_col]
    # assign to correct vertical boundary
    wa[2,bar_col] = wa[2,bar_col]+delta_u1_1
    # respective dispersion:
    we[2,bar_col] = abs(wa[2,bar_col])*0.1

    # calculate change in U2
    delta_u2 = (delta_u1_1*l[bar_col])/h[2]

    # determine number of columns fully spanned by barrier recirc
    rec_ncol_bar = recirc_col(recirc = bar_recirc)
    
    if rec_ncol_bar <= 3: 
        # carry on delta u2 until drops down
        # remember convention within python to be right exclusive when using :
        ua[2,(bar_col+1):(rec_ncol_bar+2)] = ua[2,(bar_col+1):(rec_ncol_bar+2)]+delta_u2
        # respective dispersion:
        ue[2,(bar_col+1):(rec_ncol_bar+2)] = abs(ua[2,(bar_col+1):(rec_ncol_bar+2)])*0.1
        
        # carry on delta u1 until the rejoin
        ua[1,(bar_col+1):(rec_ncol_bar+2)] = ua[1,(bar_col+1):(rec_ncol_bar+2)]-delta_u1
        # respective dispersion:
        ue[1,(bar_col+1):(rec_ncol_bar+2)] = abs(ua[1,(bar_col+1):(rec_ncol_bar+2)])*0.1
    
        # bring delta U2 back down to ground
        delta_u2_down = ((delta_u2*h[2])/l[rec_ncol_bar+1])
        wa[2,rec_ncol_bar+1] = -delta_u2_down # negative convention for downward flows
        # respective dispersion:
        we[2,rec_ncol_bar+1] = abs(wa[2,rec_ncol_bar+1])*0.1
    
        # remaining flows should now have returned to pre barrier conditions
    
    
    else:
        # carry on delta U2 across 2nd row
        ua[2,(bar_col+1):] = ua[2,(bar_col+1):]+delta_u2
        # respective dispersion:
        ue[2,(bar_col+1):] = abs(ua[2,(bar_col+1):])*0.1
    
        # carry delta u1 across 1st row
        ua[1,(bar_col+1):] = ua[1,(bar_col+1):]-delta_u1
        # respective dispersion:
        ue[1,(bar_col+1):] = abs(ua[1,(bar_col+1):])*0.1

    
        # adjust vertical values in column 5
        delta_u1_11 = (delta_u1*h[1])/l[5]
        #delta_u2_11 = (delta_u2*h[2])/l[5]
    
        # only bottom vertical flow adjusted; by the time they have rejoined,
        # the flows should have returned to pre-barrier conditions
        wa[2,5] = wa[2,5]-delta_u1_11
        # respective dispersion:
        we[2,5] = abs(wa[2,5])*0.1


    return 


# 1b) Flow effected by barrier placed within recirculation zone

def bar_inside(obs, bar_col, ue, ua, we, wa, h, l, rec_ncol, rec_nrow):
    # calculate horizontal change in row 1 - if there is a recirculation flow
    # there will always be a value for U12
    #delta_ur_11 = obs*abs(ua[1,2])
    # calculate vertial change near going up
    #delta_ur_111 = (delta_ur_11*h[1])/l[1]
    # calculate vertical change going down
    #delta_ur_1 = (delta_ur_11*h[1])/l[rec_ncol]
    
    # no change in Ur to simulate no slowing of Ur due to barrier
    delta_ur_1 = 0
    delta_ur_11 = 0
    delta_ur_111 = 0

    # calculate horizontal change in top row (could be either row 2 or 3)
    #delta_ur_2 = (delta_ur_111*l[1])/h[2]
    #delta_ur_3 = (delta_ur_111*l[1])/h[3]
    
    delta_ur_2 = 0
    delta_ur_3 = 0

    # flows within recirc region
    if rec_nrow == 2:
        # advection: top row horizontal
        ua[2,2:(rec_ncol+1)] = ua[2,2:(rec_ncol+1)]-delta_ur_2
    
        # advection: vertical drop down: added change as this is a negative flow
        wa[2,rec_ncol] = wa[2,rec_ncol]+delta_ur_1
    
        # advection: bottom row horizontal: addition of change as this is defined as a negative flow (right to left) but needs to be slowed
        ua[1,2:(rec_ncol+1)] = ua[1,2:(rec_ncol+1)]+delta_ur_11
    
        # advection: vertical up
        wa[2,1] = wa[2,1]-delta_ur_111
    
        # horizontal dispersion
        ue[2,2:(rec_ncol+1)] = abs(ua[2,2:(rec_ncol+1)])*0.1
        ue[1,2:(rec_ncol+1)] = abs(ua[1,2:(rec_ncol+1)])*0.1
    
        # vertical dispersion = 10% of average up and down velocities
        # vertical dispersion within recirc
        we[2,1] = abs(wa[2,1])*0.1
        we[2,rec_ncol] = abs(wa[2,rec_ncol])*0.1
        we[2,2:rec_ncol] =  (we[2,1]+we[2,rec_ncol])/2
        #we1[2,1:(rec_ncol+1)] = ((abs(wa1[2,1])+abs(wa1[2,rec_ncol]))/2)*0.1
    
        # additional dispersion reduction at site of barrier
        ue[1,bar_col+1] = (1-obs)*ue[1,bar_col+1]

        
    elif rec_nrow == 3:
        # advection: top row horizontal
        ua[3,2:(rec_ncol+1)] = ua[3,2:(rec_ncol+1)]-delta_ur_3
    
        # vertical advection: drop down: addition of change due to definition of positive/negative flows
        wa[3,rec_ncol] = wa[3,rec_ncol]+delta_ur_1
        wa[2,rec_ncol] = wa[2,rec_ncol]+delta_ur_1
    
        # advection: bottom row horizontal: addition of change due to definition of positive/negative flows
        ua[1,2:(rec_ncol+1)] = ua[1,2:(rec_ncol+1)]+delta_ur_11
    
        # vertical advection: up
        wa[2,1] = wa[2,1]-delta_ur_111
        wa[3,1] = wa[3,1]-delta_ur_111
    
        # horizontal dispersion
        ue[3,2:(rec_ncol+1)] = abs(ua[3,2:(rec_ncol+1)])*0.1
        ue[1,2:(rec_ncol+1)] = abs(ua[1,2:(rec_ncol+1)])*0.1
    
        # horizontal dispersion through slack middle = average of top and bottom
        ue[2,2:(rec_ncol+1)] = (ue[3,2:(rec_ncol+1)]+ue[1,2:(rec_ncol+1)])/2
    
    
        # vertical dispersion
        # default each row to average of up/down velocities *0.1
        we[2,1:(rec_ncol+1)] = ((abs(wa[2,rec_ncol])+abs(wa[2,1]))/2)*0.1
        we[3,1:(rec_ncol+1)] = ((abs(wa[3,rec_ncol])+abs(wa[3,1]))/2)*0.1
    
        # then specify each edge driven by local advection
        we[2,rec_ncol] = abs(wa[2,rec_ncol])*0.1
        we[3,rec_ncol] = abs(wa[3,rec_ncol])*0.1
    
        we[2,1] = abs(wa[2,1])*0.1
        we[3,1] = abs(wa[3,1])*0.1
    
        # additional dispersion reduction at site of barrier
        ue[1,bar_col+1] = (1-obs)*ue[1,bar_col+1]
    
    return




def bar_outside_check(bar, obs, rec, ue, ua, we, wa, l_cumu, h, l):
    if bar == l_cumu[1]:
        bar_outside(obs=obs, bar_col=1, bar_recirc=rec, ue=ue, ua=ua, we=we, wa=wa, h=h, l=l)
    if bar == l_cumu[2]:
        bar_outside(obs=obs, bar_col=2, bar_recirc=rec, ue=ue, ua=ua, we=we, wa=wa, h=h, l=l)
    if bar == l_cumu[3]:
        bar_outside(obs=obs, bar_col=3, bar_recirc=rec, ue=ue, ua=ua, we=we, wa=wa, h=h, l=l)
    if bar == l_cumu[4]:
        bar_outside(obs=obs, bar_col=4, bar_recirc=rec, ue=ue, ua=ua, we=we, wa=wa, h=h, l=l)
        
    return

def bar_inside_check(bar, obs, ue, ua, we, wa, l_cumu, h, l, rec_ncol, rec_nrow):
    if bar == l_cumu[1]:
        bar_inside(obs=obs, bar_col=1, ue=ue, ua=ua, we=we, wa=wa, h=h, l=l, rec_ncol=rec_ncol, rec_nrow=rec_nrow)
    if bar == l_cumu[2]:
        bar_inside(obs=obs, bar_col=2, ue=ue, ua=ua, we=we, wa=wa, h=h, l=l, rec_ncol=rec_ncol, rec_nrow=rec_nrow)
    if bar == l_cumu[3]:
        bar_inside(obs=obs, bar_col=3, ue=ue, ua=ua, we=we, wa=wa, h=h, l=l, rec_ncol=rec_ncol, rec_nrow=rec_nrow)
    if bar == l_cumu[4]:
        bar_inside(obs=obs, bar_col=4, ue=ue, ua=ua, we=we, wa=wa, h=h, l=l, rec_ncol=rec_ncol, rec_nrow=rec_nrow)
        
    return

def bar_inside_disp(bar, ue, obs, l_cumu):
    if bar == l_cumu[1]:
        ue[1,2] = ue[1,2]*(1-obs)
    if bar == l_cumu[2]:
        ue[1,3] = ue[1,3]*(1-obs)
    if bar == l_cumu[3]:
        ue[1,4] = ue[1,4]*(1-obs)
    if bar == l_cumu[4]:
        ue[1,5] = ue[1,5]*(1-obs)     
        
    return


# ___________ Advection & Dispersion Assignment: EXISTING BARRIERS ___________

# copy existing patterns into new storage containers
ue2_orig = ue1_orig.copy()
ua2_orig = ua1_orig.copy()
we2_orig = we1_orig.copy()
wa2_orig = wa1_orig.copy()


def existing_barrier_pattern(check, l_cumu, rec_ncol, rec_nrow, bar, obs, rec, ue2, ua2, we2, wa2, h, l, row):
        # 1. OUTSIDE RECIRC
    # barriers considered in order of impact on advection patterns
    # upwind existing
    if check[0] == 1:
        # if the barrier is outside the recirc, use the outside recirc function
        if l_cumu[rec_ncol] < bar[0]:
            bar_outside_check(bar=bar[0], obs=obs[0], rec=rec[1], ue=ue2, ua=ua2, we=we2, wa=wa2, l_cumu=l_cumu, h=h, l=l)
    
    # downwind existing            
    if check[2] == 1:
        # if the barrier is outside the recirc, use the outside recirc function
        if l_cumu[rec_ncol] < bar[2]:
            bar_outside_check(bar=bar[2], obs=obs[2], rec=rec[3], ue=ue2, ua=ua2, we=we2, wa=wa2, l_cumu=l_cumu, h=h, l=l)
        
            
    # 2. INSIDE RECIRC
    # for the inside recirc adjustments, if there are 2 existing barriers within
            # the recirc, take the largest to account for advection changes, but
            # include the effect on dispersion from the smaller barrier
    
    # if there's only one existing barrier present and this is upwind
    if check[0] == 1 and check[2] == 0:
        # if this barrier lies inside the recirc
        if l_cumu[rec_ncol] > bar[0]:
            bar_inside_check(bar=bar[0], obs=obs[0], ue=ue2, ua=ua2, we=we2, wa=wa2, h=h, l=l, rec_ncol=rec_ncol, rec_nrow=rec_nrow, l_cumu=l_cumu)
    
    # if there's only one existing barrier present and this is downwind       
    elif check[0] == 0 and check[2] == 1:
        # if this barrier lies inside the recirc
        if l_cumu[rec_ncol] > bar[2]:
            bar_inside_check(bar=bar[2], obs=obs[2], ue=ue2, ua=ua2, we=we2, wa=wa2, h=h, l=l, rec_ncol=rec_ncol, rec_nrow=rec_nrow, l_cumu=l_cumu)
    
    # if there are two barriers present           
    elif check[0] == 1 and check [2] == 1:
        # and they are both inside the recirc 
        if l_cumu[rec_ncol] > bar[0] and l_cumu[rec_ncol] > bar[2]:
            # if existing upwind is largest:
            if row[8] > row[9]:
                # decrease advection based on upwind existing
                bar_inside_check(bar=bar[0], obs=obs[0], ue=ue2, ua=ua2, we=we2, wa=wa2, h=h, l=l, rec_ncol=rec_ncol, rec_nrow=rec_nrow, l_cumu=l_cumu)
                
                # decrease dispersion based on downwind existing
                bar_inside_disp(bar=bar[2], ue=ue2, obs=obs[2], l_cumu=l_cumu)
    
                
            # if existing downwind is largest:
            if row[9] >= row[8]:
                # decrease advection based on downwind existing
                bar_inside_check(bar=bar[2], obs=obs[2], ue=ue2, ua=ua2, we=we2, wa=wa2, h=h, l=l, rec_ncol=rec_ncol, rec_nrow=rec_nrow, l_cumu=l_cumu)
                    
                # decrease dispersion based on upwind existing
                bar_inside_disp(bar=bar[0], ue=ue2, obs=obs[0], l_cumu=l_cumu)                
            
            # if they are the same height, decrease advection based on downwind 
                    # barrier as this is the first that the flow will encounter
        
        # if only the upwind existing is inside the recirc 
                # note: by definition, the upwind exsiting cannot be outside the
                # recirc and the downwind existing be inside the recirc
        elif l_cumu[rec_ncol] > bar[0] and l_cumu[rec_ncol] < bar[2]:
            # decrease advection based on upwind existing
            bar_inside_check(bar=bar[0], obs=obs[0], ue=ue2, ua=ua2, we=we2, wa=wa2, h=h, l=l, rec_ncol=rec_ncol, rec_nrow=rec_nrow, l_cumu=l_cumu)
    
    
    # 3. EDGE OF RECIRC
        # if the barrier lies on the recirc boundary: decrease dispersion
    if l_cumu[rec_ncol] == bar[0]:
        ue2[1,rec_ncol+1] = ue2[1,rec_ncol+1]*(1-obs[0])
        
    elif l_cumu[rec_ncol] == bar[2]:
        ue2[1,rec_ncol+1] = ue2[1,rec_ncol+1]*(1-obs[2])
    
    return ue2, ua2, we2, wa2 

existing_barriers_orig = existing_barrier_pattern(check=check_original, l_cumu=l_cumu_original, 
                                                  rec_ncol=rec_ncol_orig, rec_nrow=rec_nrow_orig, 
                                                  bar=bar_original, obs=obs_original, rec=rec_original, 
                                                  ue2=ue2_orig, ua2=ua2_orig, we2=we2_orig, wa2=wa2_orig, 
                                                  h=h_original, l=l_original, row=row_original)

ue2_orig = existing_barriers_orig[0]
ua2_orig = existing_barriers_orig[1]
we2_orig = existing_barriers_orig[2]
wa2_orig = existing_barriers_orig[3]

# print("ue2_orig")
# print(ue2_orig)
# print("ua2_orig")
# print(ua2_orig)
# print("we2_orig")
# print(we2_orig)
# print("wa2_orig")
# print(wa2_orig)



# ___________ Advection & Dispersion Assignment: NEW GI BARRIER ___________

# make a fresh copy of base advection patterns
ue3_orig = ue1_orig.copy()
ua3_orig = ua1_orig.copy()
we3_orig = we1_orig.copy()
wa3_orig = wa1_orig.copy()


def new_barrier_pattern(check, l_cumu, bar, obs, rec, ue3, ua3, we3, wa3, h, l, rec_ncol, zone, row, rec_nrow):
        # 1. OUTSIDE RECIRC
    
    # by definition, if all 3 were present, they would have to be in the order of:
    # upwind existing, new GI (upwind or downwind), downwind existing
    # hence the code will look at each in turn of impact on advection patterns
    # and therefore go through from left to right 
    
    # 1a) upwind existing
    if check[0] == 1:
        # if the barrier is outside the recirc, use the outside recirc function
        if l_cumu[rec_ncol] < bar[0]:
            bar_outside_check(bar=bar[0], obs=obs[0], rec=rec[1], ue=ue3, ua=ua3, we=we3, wa=wa3, l_cumu=l_cumu, h=h, l=l)
        
    # 1b) new GI upwind
    if check[1] == 1:
        # if the new barrier is outside the upwind building recirc
        if l_cumu[rec_ncol] < bar[1]:
            bar_outside_check(bar=bar[1], obs=obs[1], rec=rec[2], ue=ue3, ua=ua3, we=we3, wa=wa3, l_cumu=l_cumu, h=h, l=l)
    
    # 1c) new GI downwind
    if check[3] == 1:
        # if the new barrier is outside the upwind building recirc
        if l_cumu[rec_ncol] < bar[3]:
            bar_outside_check(bar=bar[3], obs=obs[3], rec=rec[4], ue=ue3, ua=ua3, we=we3, wa=wa3, l_cumu=l_cumu, h=h, l=l)
                
    # 1d) existing downind
    if check[2] == 1:
        # if the barrier is outside the recirc, use the outside recirc function
        if l_cumu[rec_ncol] < bar[2]:
            bar_outside_check(bar=bar[2], obs=obs[2], rec=rec[3], ue=ue3, ua=ua3, we=we3, wa=wa3, l_cumu=l_cumu, h=h, l=l)
                    
                
    # 2. INSIDE RECIRCULATION         
    
    # 2a) if the recirc covers all possible barriers it must be beyond the downwind street boundary
    # therefore including: upwind existing, upwind/downwind new, downwind existing
    if l_cumu[rec_ncol] > zone[3]:
        # if all barriers are the same height
        if max(row[2:6]) == row[8] and row[8] == row[9]:
            # apply advection changes based on GI
            # if the barrier is upwind
            if check[1] == 1:
                # apply advection changes based on new barrier upwind
                bar_inside_check(bar=bar[1], obs=obs[1], ue=ue3, ua=ua3, we=we3, wa=wa3, l_cumu=l_cumu, h=h, l=l, rec_ncol=rec_ncol, rec_nrow=rec_nrow)
                
                # apply dispersion to the existing barriers
                bar_inside_disp(bar=bar[0], ue=ue3, obs=obs[0], l_cumu=l_cumu)
                bar_inside_disp(bar=bar[2], ue=ue3, obs=obs[2], l_cumu=l_cumu)
                
            # else if the new barrier is downwind
            elif check[3] == 1:
                # apply advection changes based on new barrier downwind
                bar_inside_check(bar=bar[3], obs=obs[3], ue=ue3, ua=ua3, we=we3, wa=wa3, l_cumu=l_cumu, h=h, l=l, rec_ncol=rec_ncol, rec_nrow=rec_nrow)
                
                # apply dispersion to the existing barriers
                bar_inside_disp(bar=bar[0], ue=ue3, obs=obs[0], l_cumu=l_cumu)
                bar_inside_disp(bar=bar[2], ue=ue3, obs=obs[2], l_cumu=l_cumu)
                
        # if the GI is equal to the tallest existing barrier 
        elif max(row[2:6]) == max(row[8:]):
            # apply advection changes based on GI
            # if the barrier is upwind
            if check[1] == 1:
                # apply advection changes based on new barrier upwind
                bar_inside_check(bar=bar[1], obs=obs[1], ue=ue3, ua=ua3, we=we3, wa=wa3, l_cumu=l_cumu, h=h, l=l, rec_ncol=rec_ncol, rec_nrow=rec_nrow)
                
                # apply dispersion to the existing barriers
                bar_inside_disp(bar=bar[0], ue=ue3, obs=obs[0], l_cumu=l_cumu)
                bar_inside_disp(bar=bar[2], ue=ue3, obs=obs[2], l_cumu=l_cumu)
                
            # else if the new barrier is downwind
            elif check[3] == 1:
                # apply advection changes based on new barrier downwind
                bar_inside_check(bar=bar[3], obs=obs[3], ue=ue3, ua=ua3, we=we3, wa=wa3, l_cumu=l_cumu, h=h, l=l, rec_ncol=rec_ncol, rec_nrow=rec_nrow)
                
                # apply dispersion to the existing barriers
                bar_inside_disp(bar=bar[0], ue=ue3, obs=obs[0], l_cumu=l_cumu)
                bar_inside_disp(bar=bar[2], ue=ue3, obs=obs[2], l_cumu=l_cumu)
                
        # if the existing barriers are equal heights, and they are both greater than the GI
        elif row[8] == row[9] and row[8] > max(row[2:6]):
            # advection changes on downwind barrier
            bar_inside_check(bar=bar[2], obs=obs[2], ue=ue3, ua=ua3, we=we3, wa=wa3, l_cumu=l_cumu, h=h, l=l, rec_ncol=rec_ncol, rec_nrow=rec_nrow)
                
            # dispersion changes based on others:
            bar_inside_disp(bar=bar[0], ue=ue3, obs=obs[0], l_cumu=l_cumu)
            bar_inside_disp(bar=bar[1], ue=ue3, obs=obs[1], l_cumu=l_cumu)
            bar_inside_disp(bar=bar[3], ue=ue3, obs=obs[3], l_cumu=l_cumu)
        
        # if upwind existing is exclusively the tallest
        elif row[8] == max(row[2:]):
            # apply advection using upwind existing barrier info
            bar_inside_check(bar=bar[0], obs=obs[0], ue=ue3, ua=ua3, we=we3, wa=wa3, l_cumu=l_cumu, h=h, l=l, rec_ncol=rec_ncol, rec_nrow=rec_nrow)
            
            # apply dispersion changes for all other barriers
            bar_inside_disp(bar=bar[1], ue=ue3, obs=obs[1], l_cumu=l_cumu)
            bar_inside_disp(bar=bar[2], ue=ue3, obs=obs[2], l_cumu=l_cumu)
            bar_inside_disp(bar=bar[3], ue=ue3, obs=obs[3], l_cumu=l_cumu)
            
        # if the downwind existing is exclusively the tallest    
        elif row[9] == max(row[2:]):
            # apply advection using downwind existing barrier info
            bar_inside_check(bar=bar[2], obs=obs[2], ue=ue3, ua=ua3, we=we3, wa=wa3, l_cumu=l_cumu, h=h, l=l, rec_ncol=rec_ncol, rec_nrow=rec_nrow)
            
            # apply dispersion changes for all other barriers
            bar_inside_disp(bar=bar[0], ue=ue3, obs=obs[0], l_cumu=l_cumu)
            bar_inside_disp(bar=bar[1], ue=ue3, obs=obs[1], l_cumu=l_cumu)
            bar_inside_disp(bar=bar[3], ue=ue3, obs=obs[3], l_cumu=l_cumu)
        
        # else if the GI is exclusively the tallest    
        elif max(row[2:6]) == max(row[2:]):
            # if the barrier is upwind
            if check[1] == 1:
                # apply advection changes based on new barrier upwind
                bar_inside_check(bar=bar[1], obs=obs[1], ue=ue3, ua=ua3, we=we3, wa=wa3, l_cumu=l_cumu, h=h, l=l, rec_ncol=rec_ncol, rec_nrow=rec_nrow)
                
                # apply dispersion to the existing barriers
                bar_inside_disp(bar=bar[0], ue=ue3, obs=obs[0], l_cumu=l_cumu)
                bar_inside_disp(bar=bar[2], ue=ue3, obs=obs[2], l_cumu=l_cumu)
                
            # else if the new barrier is downwind
            elif check[3] == 1:
                # apply advection changes based on new barrier downwind
                bar_inside_check(bar=bar[3], obs=obs[3], ue=ue3, ua=ua3, we=we3, wa=wa3, l_cumu=l_cumu, h=h, l=l, rec_ncol=rec_ncol, rec_nrow=rec_nrow)
                
                # apply dispersion to the existing barriers
                bar_inside_disp(bar=bar[0], ue=ue3, obs=obs[0], l_cumu=l_cumu)
                bar_inside_disp(bar=bar[2], ue=ue3, obs=obs[2], l_cumu=l_cumu)
                
    
    
    # 2b) if the recirc covers the upwind existing and new GI placed downwind, 
                    # (recirc doesn't cover downwind existing)
    if l_cumu[rec_ncol] > bar[3] and l_cumu[rec_ncol] <= zone[3]:
        # check if there is a barrier downwind (bar[3] would be 0 if there is no barrier downwind)
        if check[3] == 1:
            # if they are both the same height
            if row[8] == max(row[2:6]):
                # apply advection using new barrier
                bar_inside_check(bar=bar[3], obs=obs[3], ue=ue3, ua=ua3, we=we3, wa=wa3, l_cumu=l_cumu, h=h, l=l, rec_ncol=rec_ncol, rec_nrow=rec_nrow)
                # apply dispersion using upwind existing
                bar_inside_disp(bar=bar[0], ue=ue3, obs=obs[0], l_cumu=l_cumu)
            
            # if upwind existing is the tallest
            elif row[8] > max(row[2:6]):
                # apply advection using upwind existing barrier info
                bar_inside_check(bar=bar[0], obs=obs[0], ue=ue3, ua=ua3, we=we3, wa=wa3, l_cumu=l_cumu, h=h, l=l, rec_ncol=rec_ncol, rec_nrow=rec_nrow)
                # apply dispersion changes using new barrier info
                bar_inside_disp(bar=bar[3], ue=ue3, obs=obs[3], l_cumu=l_cumu)
                
            # else if the new barrier is the tallest
            elif max(row[2:6]) > row[8]:
                # apply advection using new barrier
                bar_inside_check(bar=bar[3], obs=obs[3], ue=ue3, ua=ua3, we=we3, wa=wa3, l_cumu=l_cumu, h=h, l=l, rec_ncol=rec_ncol, rec_nrow=rec_nrow)
                # apply dispersion using upwind existing
                bar_inside_disp(bar=bar[0], ue=ue3, obs=obs[0], l_cumu=l_cumu)
    
    
    # 2c) if the recirc covers the upwind existing and new GI placed upwind, 
                    # (recirc doesn't cover downwind existing)
    if l_cumu[rec_ncol] > bar[1] and l_cumu[rec_ncol] <= zone[3]:
        if check[1] == 1:
            # if they are both the same height
            if row[8] == max(row[2:6]):
                # apply advection using new barrier
                bar_inside_check(bar=bar[1], obs=obs[1], ue=ue3, ua=ua3, we=we3, wa=wa3, l_cumu=l_cumu, h=h, l=l, rec_ncol=rec_ncol, rec_nrow=rec_nrow)
                # apply dispersion using upwind existing
                bar_inside_disp(bar=bar[0], ue=ue3, obs=obs[0], l_cumu=l_cumu)
            
            # if upwind existing is the tallest
            elif row[8] > max(row[2:6]):
                # apply advection using upwind existing barrier info
                bar_inside_check(bar=bar[0], obs=obs[0], ue=ue3, ua=ua3, we=we3, wa=wa3, l_cumu=l_cumu, h=h, l=l, rec_ncol=rec_ncol, rec_nrow=rec_nrow)
                # apply dispersion changes using new barrier info
                bar_inside_disp(bar=bar[1], ue=ue3, obs=obs[1], l_cumu=l_cumu)
                
            # else if the new barrier is the tallest
            elif max(row[2:6]) > row[8]:
                # apply advection using new barrier
                bar_inside_check(bar=bar[1], obs=obs[1], ue=ue3, ua=ua3, we=we3, wa=wa3, l_cumu=l_cumu, h=h, l=l, rec_ncol=rec_ncol, rec_nrow=rec_nrow)
                # apply dispersion using upwind existing
                bar_inside_disp(bar=bar[0], ue=ue3, obs=obs[0], l_cumu=l_cumu)
    
    # 2d) if the recirc only covers the upwind existing barrier
    if l_cumu[rec_ncol] > bar[0] and (l_cumu[rec_ncol] <= bar[1] or l_cumu[rec_ncol] <= bar[3]):
        bar_inside_check(bar=bar[0], obs=obs[0], ue=ue3, ua=ua3, we=we3, wa=wa3, l_cumu=l_cumu, h=h, l=l, rec_ncol=rec_ncol, rec_nrow=rec_nrow)
    
    else:
        ue3 = ue3
        ua3 = ua3
        we3 = we3
        wa3 = wa3
    
    
    # 3. EDGE OF RECIRC
    # if a barrier lies on the recirc boundary: decrease dispersion
    # upwind existing
    if l_cumu[rec_ncol] == bar[0]:
        ue3[1,rec_ncol+1] = ue3[1,rec_ncol+1]*(1-obs[0])
    
    # upwind GI
    elif l_cumu[rec_ncol] == bar[1]:
        ue3[1,rec_ncol+1] = ue3[1,rec_ncol+1]*(1-obs[1])
    
    # downwind GI
    elif l_cumu[rec_ncol] == bar[3]:
        ue3[1,rec_ncol+1] = ue3[1,rec_ncol+1]*(1-obs[3])
    
    # downwind existing        
    elif l_cumu[rec_ncol] == bar[2]:
        ue3[1,rec_ncol+1] = ue3[1,rec_ncol+1]*(1-obs[2])
        
    return ue3, ua3, we3, wa3 



new_barrier_orig = new_barrier_pattern(check=check_original, l_cumu=l_cumu_original, 
                                       bar=bar_original, obs=obs_original, rec=rec_original, 
                                       ue3=ue3_orig, ua3=ua3_orig, we3=we3_orig, wa3=wa3_orig, 
                                       h=h_original, l=l_original, rec_ncol=rec_ncol_orig, 
                                       zone=zone_original, row=row_original, rec_nrow=rec_nrow_orig)

ue3_orig = new_barrier_orig[0]
ua3_orig = new_barrier_orig[1]
we3_orig = new_barrier_orig[2]
wa3_orig = new_barrier_orig[3]

# print("ue3_orig")
# print(ue3_orig)
# print("ua3_orig")
# print(ua3_orig)
# print("we3_orig")
# print(we3_orig)
# print("wa3_orig")
# print(wa3_orig)



# ___________ Parallel wind: dispersion only ___________

# we don't want any influence of recirculation zones; for these to occur wind 
# would need to be across the street

def ws_point_parallel(ubg,z,H):
    # firstly set d
    d = 0
    
    # calculate wind speed (u) at the height of shortest building (where log profile is still valid to)
    # in this study uh is provided
    uh = 4.65
    
    if z <= H:
        # if determining wind speed at point below both building heights (H is set to minimum)
        uz = uh*(z/H)
    else:
        # if determining wind speed at a point above smallest building (e.g. between building heights) employ log profile
        #uz = ubg*(math.log(5000)/math.log(500))*(math.log(5*H-5*d)/math.log(500-5*d))
        uz = ubg*(math.log(5000)/math.log(500))*(math.log(5*z-5*d)/math.log(500-5*d))
        
    return uz



# function to determine average wind speed (u) across a row
def ws_average_parallel(row_min, row_max, ubg, H):
    dif = row_max - row_min
    dif9 = dif/9
    
    u1 = ws_point_parallel(ubg=ubg, z = row_min, H=H)
    u2 = ws_point_parallel(ubg=ubg, z = row_min+(1*dif9), H=H)
    u3 = ws_point_parallel(ubg=ubg, z = row_min+(2*dif9), H=H)
    u4 = ws_point_parallel(ubg=ubg, z = row_min+(3*dif9), H=H)
    u5 = ws_point_parallel(ubg=ubg, z = row_min+(4*dif9), H=H)
    u6 = ws_point_parallel(ubg=ubg, z = row_min+(5*dif9), H=H)
    u7 = ws_point_parallel(ubg=ubg, z = row_min+(6*dif9), H=H)
    u8 = ws_point_parallel(ubg=ubg, z = row_min+(7*dif9), H=H)
    u9 = ws_point_parallel(ubg=ubg, z = row_min+(8*dif9), H=H)
    u10 = ws_point_parallel(ubg=ubg, z = row_max, H=H)
    
    data = (u1,u2,u3,u4,u5,u6,u7,u8,u9,u10)
    avg_u = mean(data)
    
    return avg_u


# assume ubg_parallel is the same as ubg_orig
ubg_parallel = ubg_orig
#print(ubg_parallel)
#Uh_test = ws_point_parallel(ubg = ubg_parallel, z=18,H=18)
#print(test)

ue_row1 = ws_average_parallel(row_min = 0, row_max = h_cumu_original[1], ubg = ubg_parallel, H = min(row_original[0],row_original[1]))
ue_row2 = ws_average_parallel(row_min = h_cumu_original[1], row_max = h_cumu_original[2], ubg = ubg_parallel, H = min(row_original[0],row_original[1]))
ue_row3 = ws_average_parallel(row_min = h_cumu_original[2], row_max = h_cumu_original[3], ubg = ubg_parallel, H = min(row_original[0],row_original[1]))

we_row12 = ws_point_parallel(ubg = ubg_parallel, z=h_cumu_original[1], H=min(row_original[0],row_original[1]))
we_row23 = ws_point_parallel(ubg = ubg_parallel, z=h_cumu_original[2], H=min(row_original[0],row_original[1]))
we_row3b = ws_point_parallel(ubg = ubg_parallel, z=h_cumu_original[3], H=min(row_original[0],row_original[1]))

#print(ue_row1, ue_row2, ue_row3)
#print(we_row12, we_row23, we_row3b)

# create empty containers
# existing barriers, dimensioning based on wind left to right
ue1_par = np.zeros((5,6)) 	
ua1_par = np.zeros((5,6))   # advection - will remain 0
we1_par = np.zeros((5,6))  
wa1_par = np.zeros((5,6))   # advection - will remain 0

# existing barriers only, dimensioning based on wind right to left
ue2_par = np.zeros((5,6)) 	
ua2_par = np.zeros((5,6))   # advection - will remain 0
we2_par = np.zeros((5,6))  
wa2_par = np.zeros((5,6))   # advection - will remain 0

# apply linear reductions of ACH from roof top for both horizontal and vertical
ue1_par[1,1] = 0 #c11 to wall
ue1_par[1,2] = ue_row1*0.1 #c11 and c12
ue1_par[1,3] = ue_row1*0.1 #c12 and c13
ue1_par[1,4] = ue_row1*0.1 #c13 and c14
ue1_par[1,5] = ue_row1*0.1 #c14 and c15
ue1_par[2,1] = 0 #c21 to wall
ue1_par[2,2] = ue_row2*0.1 #c21 and c22
ue1_par[2,3] = ue_row2*0.1 #c22 and c23
ue1_par[2,4] = ue_row2*0.1 #c23 and c24
ue1_par[2,5] = ue_row2*0.1 #c24 and c25
ue1_par[3,1] = 0 #c31 to wall
ue1_par[3,2] = ue_row3*0.1 #c31 and c32
ue1_par[3,3] = ue_row3*0.1 #c32 and c33
ue1_par[3,4] = ue_row3*0.1 #c33 and c34
ue1_par[3,5] = ue_row3*0.1 #c34 and c35

#vertical dispersion co-efficients
#we[1,1] = 0 #ground to c11
#we[1,2] = 0 #ground to c12
#we[1,3] = 0 #ground to c13
#we[1,4] = 0 #ground to c14
#we[1,5] = 0 #ground to c15
we1_par[2,1] = we_row12*0.1 #c11 to c21
we1_par[2,2] = we_row12*0.1 #c12 and c22
we1_par[2,3] = we_row12*0.1 #c13 and c23
we1_par[2,4] = we_row12*0.1 #c13 and c24
we1_par[2,5] = we_row12*0.1 #c15 and c25
we1_par[3,1] = we_row23*0.1 #c21 and c31
we1_par[3,2] = we_row23*0.1 #c22 and c32
we1_par[3,3] = we_row23*0.1 #c23 and c33
we1_par[3,4] = we_row23*0.1 #c24 and c34
we1_par[3,5] = we_row23*0.1 #c25 and c35
we1_par[4,1] = we_row3b*0.1 #c31 and cb
we1_par[4,2] = we_row3b*0.1 #c32 and cb
we1_par[4,3] = we_row3b*0.1 #c33 and cb
we1_par[4,4] = we_row3b*0.1 #c24 and cb
we1_par[4,5] = we_row3b*0.1 #c35 and cb




# create function to determine dispersion reduction based on obs at box boundary
# where a barrier lies
# use the bar_inside_disp function to assess at each box boundary

# apply function to existing barriers
# upwind
if check_original[0] == 1:
    bar_inside_disp(bar=bar_original[0], ue=ue1_par, obs=obs_original[0], l_cumu=l_cumu_original)
# downwind    
if check_original[2] == 1:
    bar_inside_disp(bar=bar_original[2], ue=ue1_par, obs=obs_original[2], l_cumu=l_cumu_original)


# make copies of existing dispersion patterns
ue3_par = ue1_par.copy()
ua3_par = ua1_par.copy()
we3_par = we1_par.copy()
wa3_par = wa1_par.copy()


# apply function to new barriers
if check_original[1] == 1:
    bar_inside_disp(bar=bar_original[1], ue=ue3_par, obs=obs_original[1], l_cumu=l_cumu_original)

if check_original[3] == 1:
    bar_inside_disp(bar=bar_original[3], ue=ue3_par, obs=obs_original[3], l_cumu=l_cumu_original)


    
###############################################################################
# EMISSIONS
###############################################################################


# partition emissions into respective boxes
# value represents % of emissions in box - should all sum to 1
emis_par_orig = np.array([0,0,0,0,0], dtype = float) 
emis_par_mir = np.array([0,0,0,0,0], dtype = float) 

ez1_par_orig = np.array([0,0,0,0,0,0], dtype = float)
ez2_par_orig = np.array([0,0,0,0,0,0], dtype = float)

ez1_par_mir = np.array([0,0,0,0,0,0], dtype = float)
ez2_par_mir = np.array([0,0,0,0,0,0], dtype = float)


def emission_partition(l_cumu, l, ez1_start, ez1_finish, ez2_start, ez2_finish, ez1_par, ez2_par, ez1_w, ez2_w):
        
    # go through each column and see if it crosses into the emissions zone
    # potentially reformat here into function to be more efficient
    
    # column 1
    if l_cumu[1] <= ez1_start:
        ez1_par[1] = 0
        ez2_par[1] = 0
    elif l_cumu[1] > ez1_start:
        if l_cumu[1] <= ez1_finish:
            ez1_par[1] = (l_cumu[1]-ez1_start)/ez1_w
            ez2_par[1] = 0
    elif l_cumu[1] > ez1_finish:
        if l_cumu[1] <= ez2_start:
            ez1_par[1] = 1
            ez2_par[1] = 0
    elif l[1] > ez2_start:
        if l_cumu[1] <= ez2_finish:
            ez1_par[1] = 1
            ez2_par[1] = (l_cumu[1]-ez2_start)/ez2_w
    elif l_cumu[1] > ez2_finish:
        ez1_par[1] = 1
        ez2_par[1] = 1
    
    
    # column 2
    if l_cumu[1] < ez1_start and l_cumu[2] <= ez1_start:
        ez1_par[2] = 0
        ez2_par[2] = 0
    elif l_cumu[1] <= ez1_start and l_cumu[2] > ez1_start:
        if l_cumu[2] <= ez1_finish:
            ez1_par[2] = (l_cumu[2]-ez1_start)/ez1_w
            ez2_par[2] = 0
        elif l_cumu[2] > ez1_finish and l_cumu[2] <= ez2_start:
            ez1_par[2] = 1
            ez2_par[2] = 0
        elif l_cumu[2] > ez2_start and l_cumu[2] <= ez2_finish:
            ez1_par[2] = 1
            ez2_par[2] = (l_cumu[2]-ez2_start)/ez2_w
        elif l_cumu[2] > ez2_finish:
            ez1_par[2] = 1
            ez2_par[2] = 1
    elif l_cumu[1] > ez1_start and l_cumu[1] <= ez1_finish:
        if l_cumu[2] <= ez1_finish:
            ez1_par[2] = (l_cumu[2]-l_cumu[1])/ez1_w
            ez2_par[2] = 0
        elif l_cumu[2] > ez1_finish and l_cumu[2] <= ez2_start:
            ez1_par[2] = (ez1_finish-l_cumu[1])/ez1_w
            ez2_par[2] = 0
        elif l_cumu[2] > ez2_start and l_cumu[2] <= ez2_finish:
            ez1_par[2] = (ez1_finish-l_cumu[1])/ez1_w
            ez2_par[2] = (l_cumu[2]-ez2_start)/ez2_w
        elif l_cumu[2] > ez2_finish:
            ez1_par[2] = (ez1_finish-l_cumu[1])/ez1_w
            ez2_par[2] = 1
    elif l_cumu[1] >= ez1_finish and l_cumu[1] <= ez2_start:
        if l_cumu[2] <= ez2_start:
            ez1_par[2] = 0
            ez1_par[2] = 0
        elif l_cumu[2] > ez2_start and l_cumu[2] <= ez2_finish:
            ez1_par[2] = 0
            ez2_par[2] = (l_cumu[2]-ez2_start)/ez2_w
        elif l_cumu[2] > ez2_finish:
            ez1_par[2] = 0
            ez2_par[2] = 1
    elif l_cumu[1] >= ez2_start and l_cumu[1] <= ez2_finish:
        if l_cumu[2] <= ez2_finish:
            ez1_par[2] = 0
            ez2_par[2] = (l_cumu[2]-l_cumu[1])/ez2_w
        if l_cumu[2] > ez2_finish:
            ez1_par[2] = 0
            ez2_par[2] = (ez2_finish-l_cumu[1])/ez2_w
    elif l_cumu[1] > ez2_finish:
        ez1_par[2] = 0
        ez2_par[2] = 0
        
    # column 3
    if l_cumu[2] < ez1_start and l_cumu[3] <= ez1_start:
        ez1_par[3] = 0
        ez2_par[3] = 0
    elif l_cumu[2] <= ez1_start and l_cumu[3] > ez1_start:
        if l_cumu[3] <= ez1_finish:
            ez1_par[3] = (l_cumu[3]-ez1_start)/ez1_w
            ez2_par[3] = 0
        elif l_cumu[3] > ez1_finish and l_cumu[3] <= ez2_start:
            ez1_par[3] = 1
            ez2_par[3] = 0
        elif l_cumu[3] > ez2_start and l_cumu[3] <= ez2_finish:
            ez1_par[3] = 1
            ez2_par[3] = (l_cumu[3]-ez2_start)/ez2_w
        elif l_cumu[3] > ez2_finish:
            ez1_par[3] = 1
            ez2_par[3] = 1
    elif l_cumu[2] > ez1_start and l_cumu[2] <= ez1_finish:
        if l_cumu[3] <= ez1_finish:
            ez1_par[3] = (l_cumu[3]-l_cumu[2])/ez1_w
            ez2_par[3] = 0
        elif l_cumu[3] > ez1_finish and l_cumu[3] <= ez2_start:
            ez1_par[3] = (ez1_finish-l_cumu[2])/ez1_w
            ez2_par[3] = 0
        elif l_cumu[3] > ez2_start and l_cumu[3] <= ez2_finish:
            ez1_par[3] = (ez1_finish-l_cumu[2])/ez1_w
            ez2_par[3] = (l_cumu[3]-ez2_start)/ez2_w
        elif l_cumu[3] > ez2_finish:
            ez1_par[3] = (ez1_finish-l_cumu[2])/ez1_w
            ez2_par[3] = 1
    elif l_cumu[2] >= ez1_finish and l_cumu[2] <= ez2_start:
        if l_cumu[3] <= ez2_start:
            ez1_par[3] = 0
            ez1_par[3] = 0
        elif l_cumu[3] > ez2_start and l_cumu[3] <= ez2_finish:
            ez1_par[3] = 0
            ez2_par[3] = (l_cumu[3]-ez2_start)/ez2_w
        elif l_cumu[3] > ez2_finish:
            ez1_par[3] = 0
            ez2_par[3] = 1
    elif l_cumu[2] >= ez2_start and l_cumu[2] <= ez2_finish:
        if l_cumu[3] <= ez2_finish:
            ez1_par[3] = 0
            ez2_par[3] = (l_cumu[3]-l_cumu[2])/ez2_w
        if l_cumu[3] > ez2_finish:
            ez1_par[3] = 0
            ez2_par[3] = (ez2_finish-l_cumu[2])/ez2_w
    elif l_cumu[2] > ez2_finish:
        ez1_par[3] = 0
        ez2_par[3] = 0
    
    # column 4
    if l_cumu[3] < ez1_start and l_cumu[4] <= ez1_start:
        ez1_par[4] = 0
        ez2_par[4] = 0
    elif l_cumu[3] <= ez1_start and l_cumu[4] > ez1_start:
        if l_cumu[4] <= ez1_finish:
            ez1_par[4] = (l_cumu[4]-ez1_start)/ez1_w
            ez2_par[4] = 0
        elif l_cumu[4] > ez1_finish and l_cumu[4] <= ez2_start:
            ez1_par[4] = 1
            ez2_par[4] = 0
        elif l_cumu[4] > ez2_start and l_cumu[4] <= ez2_finish:
            ez1_par[4] = 1
            ez2_par[4] = (l_cumu[4]-ez2_start)/ez2_w
        elif l_cumu[4] > ez2_finish:
            ez1_par[4] = 1
            ez2_par[4] = 1
    elif l_cumu[3] > ez1_start and l_cumu[3] <= ez1_finish:
        if l_cumu[4] <= ez1_finish:
            ez1_par[4] = (l_cumu[4]-l_cumu[3])/ez1_w
            ez2_par[4] = 0
        elif l_cumu[4] > ez1_finish and l_cumu[4] <= ez2_start:
            ez1_par[4] = (ez1_finish-l_cumu[3])/ez1_w
            ez2_par[4] = 0
        elif l_cumu[4] > ez2_start and l_cumu[4] <= ez2_finish:
            ez1_par[4] = (ez1_finish-l_cumu[3])/ez1_w
            ez2_par[4] = (l_cumu[4]-ez2_start)/ez2_w
        elif l_cumu[4] > ez2_finish:
            ez1_par[4] = (ez1_finish-l_cumu[3])/ez1_w
            ez2_par[4] = 1
    elif l_cumu[3] >= ez1_finish and l_cumu[3] <= ez2_start:
        if l_cumu[4] <= ez2_start:
            ez1_par[4] = 0
            ez1_par[4] = 0
        elif l_cumu[4] > ez2_start and l_cumu[4] <= ez2_finish:
            ez1_par[4] = 0
            ez2_par[4] = (l_cumu[4]-ez2_start)/ez2_w
        elif l_cumu[4] > ez2_finish:
            ez1_par[4] = 0
            ez2_par[4] = 1
    elif l_cumu[3] >= ez2_start and l_cumu[3] <= ez2_finish:
        if l_cumu[4] <= ez2_finish:
            ez1_par[4] = 0
            ez2_par[4] = (l_cumu[4]-l_cumu[3])/ez2_w
        if l_cumu[4] > ez2_finish:
            ez1_par[4] = 0
            ez2_par[4] = (ez2_finish-l_cumu[3])/ez2_w
    elif l_cumu[3] > ez2_finish:
        ez1_par[4] = 0
        ez2_par[4] = 0
        
    # column 5
    if l_cumu[4] >= ez2_finish:
        ez1_par[5] = 0
        ez2_par[5] = 0
    elif l_cumu[4] >= ez2_start and l_cumu[4] < ez2_finish:
        ez1_par[5] = 0
        ez2_par[5] = (ez2_finish-l_cumu[4])/ez2_w
    elif l_cumu[4] >= ez1_finish and l_cumu[4] <= ez2_start:
        ez1_par[5] = 0
        ez2_par[5] = 1
    elif l_cumu[4] >= ez1_start and l_cumu[4] < ez1_finish:
        ez1_par[5] = (ez1_finish - l_cumu[4])/ez1_w
        ez2_par[5] = 1
    elif l_cumu[4] < ez1_start:
        ez1_par[5] = 1
        ez2_par[5] = 1
        
    return ez1_par, ez2_par

emis_par_fun_orig = emission_partition(l_cumu=l_cumu_original, l=l_original, 
                                       ez1_start=ez1_start, ez1_finish=ez1_finish, ez2_start=ez2_start, ez2_finish=ez2_finish, 
                                       ez1_par=ez1_par_orig, ez2_par=ez2_par_orig, 
                                       ez1_w=ez1_w, ez2_w=ez2_w)

ez1_par_orig = emis_par_fun_orig[0]
ez2_par_orig = emis_par_fun_orig[1]

#print(ez1_par_orig)
#print(ez2_par_orig)


ez_tot_no2_orig = np.array([0,0,0,0,0,0], dtype = float)
ez_tot_no2_orig[1] = (ez1_par_orig[1]*float(ez1_emis_no2)) + (ez2_par_orig[1]*float(ez2_emis_no2))
ez_tot_no2_orig[2] = (ez1_par_orig[2]*float(ez1_emis_no2)) + (ez2_par_orig[2]*float(ez2_emis_no2))
ez_tot_no2_orig[3] = (ez1_par_orig[3]*float(ez1_emis_no2)) + (ez2_par_orig[3]*float(ez2_emis_no2))
ez_tot_no2_orig[4] = (ez1_par_orig[4]*float(ez1_emis_no2)) + (ez2_par_orig[4]*float(ez2_emis_no2))
ez_tot_no2_orig[5] = (ez1_par_orig[5]*float(ez1_emis_no2)) + (ez2_par_orig[5]*float(ez2_emis_no2))


# check the sum of partitioned emissions equal the orginal
if round(sum(ez_tot_no2_orig),0) != round((float(ez1_emis_no2) + float(ez2_emis_no2))):
    error[8,1] = 1
    

###############################################################################
# A MATRIX 
###############################################################################

# ___________ Functions for positive/negative conventions ___________

# postive flows: left to right, and upwards (bottom to top)
# negative flows: right to left, and downwards (top to bottom)
def alpha(x):
    if x > 0:
        return 1
    elif x < 0:
        return 0
    elif x == 0:
        return 0
    
def beta(x):
    if x > 0:
        return 0
    elif x < 0:
        return 1
    elif x == 0:
        return 0
    
        
# ___________ Ratios ___________
# ratios worked out to make equations more readable in a matrix
r1 = np.zeros((4,6))
r1[1,1] = h_original[1]/l_original[1] 
r1[1,2] = h_original[1]/l_original[2] 
r1[1,3] = h_original[1]/l_original[3]
r1[1,4] = h_original[1]/l_original[4] 
r1[1,5] = h_original[1]/l_original[5] 
r1[2,1] = h_original[2]/l_original[1] 
r1[2,2] = h_original[2]/l_original[2]
r1[2,3] = h_original[2]/l_original[3] 
r1[2,4] = h_original[2]/l_original[4]
r1[2,5] = h_original[2]/l_original[5]
r1[3,1] = h_original[3]/l_original[1]
r1[3,2] = h_original[3]/l_original[2] 
r1[3,3] = h_original[3]/l_original[3] 
r1[3,4] = h_original[3]/l_original[4]
r1[3,5] = h_original[3]/l_original[5] 


# ___________ Large 5x3 A-Matrix function ___________

def a_matrix(r, u, U, w, W):
    a = np.zeros((15,15), dtype = float) # a matrix
    # C11
    a[0,0] = r[1,1]*(u[1,2] + alpha(U[1,2])*U[1,2]) + w[2,1] + alpha(W[2,1])*W[2,1]
    a[0,1] = r[1,1]*(beta(U[1,2])*U[1,2] - u[1,2])
    a[0,2] = 0
    a[0,3] = 0
    a[0,4] = 0
    a[0,5] = beta(W[2,1])*W[2,1] - w[2,1]
    a[0,6] = 0
    a[0,7] = 0
    a[0,8] = 0
    a[0,9] = 0
    a[0,10] = 0
    a[0,11] = 0
    a[0,12] = 0
    a[0,13] = 0
    a[0,14] = 0
    
    # C12
    a[1,0] = r[1,2]*(-alpha(U[1,2])*U[1,2] - u[1,2])
    a[1,1] = r[1,2]*(alpha(U[1,3])*U[1,3] + u[1,3] - beta(U[1,2])*U[1,2] + u[1,2]) + w[2,2] + alpha(W[2,2])*W[2,2]
    a[1,2] = r[1,2]*(beta(U[1,3])*U[1,3] - u[1,3])
    a[1,3] = 0
    a[1,4] = 0
    a[1,5] = 0
    a[1,6] = beta(W[2,2])*W[2,2] - w[2,2]
    a[1,7] = 0
    a[1,8] = 0
    a[1,9] = 0
    a[1,10] = 0
    a[1,11] = 0
    a[1,12] = 0
    a[1,13] = 0
    a[1,14] = 0
    
    # C13
    a[2,0] = 0
    a[2,1] = r[1,3]*(-alpha(U[1,3])*U[1,3] - u[1,3])
    a[2,2] = r[1,3]*(alpha(U[1,4])*U[1,4] + u[1,4] - beta(U[1,3])*U[1,3] + u[1,3]) + alpha(W[2,3])*W[2,3] + w[2,3]
    a[2,3] = r[1,3]*(beta(U[1,4])*U[1,4] - u[1,4])
    a[2,4] = 0
    a[2,5] = 0
    a[2,6] = 0
    a[2,7] = beta(W[2,3])*W[2,3] - w[2,3]
    a[2,8] = 0
    a[2,9] = 0
    a[2,10] = 0
    a[2,11] = 0
    a[2,12] = 0
    a[2,13] = 0
    a[2,14] = 0
    
    # C14
    a[3,0] = 0
    a[3,1] = 0
    a[3,2] = r[1,4]*(-alpha(U[1,4])*U[1,4] - u[1,4])
    a[3,3] = r[1,4]*(alpha(U[1,5])*U[1,5] + u[1,5] - beta(U[1,4])*U[1,4] + u[1,4]) + alpha(W[2,4])*W[2,4] + w[2,4]
    a[3,4] = r[1,4]*(beta(U[1,5])*U[1,5] - u[1,5])
    a[3,5] = 0
    a[3,6] = 0
    a[3,7] = 0
    a[3,8] = beta(W[2,4])*W[2,4] - w[2,4]
    a[3,9] = 0
    a[3,10] = 0
    a[3,11] = 0
    a[3,12] = 0
    a[3,13] = 0
    a[3,14] = 0
    
    # C15
    a[4,0] = 0
    a[4,1] = 0
    a[4,2] = 0
    a[4,3] = r[1,5]*(-alpha(U[1,5])*U[1,5] - u[1,5])
    a[4,4] = r[1,5]*(u[1,5] - beta(U[1,5])*U[1,5]) + alpha(W[2,5])*W[2,5] + w[2,5]
    a[4,5] = 0
    a[4,6] = 0
    a[4,7] = 0
    a[4,8] = 0
    a[4,9] = beta(W[2,5])*W[2,5] - w[2,5]
    a[4,10] = 0
    a[4,11] = 0
    a[4,12] = 0
    a[4,13] = 0
    a[4,14] = 0
    
    # C21
    a[5,0] = -alpha(W[2,1])*W[2,1] - w[2,1]
    a[5,1] = 0
    a[5,2] = 0
    a[5,3] = 0
    a[5,4] = 0
    a[5,5] = r[2,1]*(alpha(U[2,2])*U[2,2] + u[2,2]) + w[2,1] - beta(W[2,1])*W[2,1] + w[3,1] + alpha(W[3,1])*W[3,1]
    a[5,6] = r[2,1]*(beta(U[2,2])*U[2,2] - u[2,2])
    a[5,7] = 0
    a[5,8] = 0
    a[5,9] = 0
    a[5,10] = beta(W[3,1])*W[3,1] - w[3,1]
    a[5,11] = 0
    a[5,12] = 0
    a[5,13] = 0
    a[5,14] = 0
    
    # C22
    a[6,0] = 0
    a[6,1] = -alpha(W[2,2])*W[2,2] - w[2,2]
    a[6,2] = 0
    a[6,3] = 0
    a[6,4] = 0
    a[6,5] = r[2,2]*(-alpha(U[2,2])*U[2,2] - u[2,2])
    a[6,6] = r[2,2]*(alpha(U[2,3])*U[2,3] + u[2,3] - beta(U[2,2])*U[2,2] + u[2,2]) + w[2,2] - beta(W[2,2])*W[2,2] + w[3,2] + alpha(W[3,2])*W[3,2]
    a[6,7] = r[2,2]*(beta(U[2,3])*U[2,3] - u[2,3])
    a[6,8] = 0
    a[6,9] = 0
    a[6,10] = 0
    a[6,11] = beta(W[3,2])*W[3,2] - w[3,2]
    a[6,12] = 0
    a[6,13] = 0
    a[6,14] = 0
    
    # C23  
    a[7,0] = 0
    a[7,1] = 0
    a[7,2] = -alpha(W[2,3])*W[2,3] - w[2,3]
    a[7,3] = 0
    a[7,4] = 0
    a[7,5] = 0
    a[7,6] = r[2,3]*(-alpha(U[2,3])*U[2,3] - u[2,3])
    a[7,7] = r[2,3]*(alpha(U[2,4])*U[2,4] + u[2,4] - beta(U[2,3])*U[2,3] + u[2,3]) + w[2,3] - beta(W[2,3])*W[2,3] + w[3,3] + alpha(W[3,3])*W[3,3]
    a[7,8] = r[2,3]*(beta(U[2,4])*U[2,4] - u[2,4])
    a[7,9] = 0
    a[7,10] = 0
    a[7,11] = 0
    a[7,12] = beta(W[3,3])*W[3,3] - w[3,3]
    a[7,13] = 0
    a[7,14] = 0
    
    # C24
    a[8,0] = 0
    a[8,1] = 0
    a[8,2] = 0
    a[8,3] = -alpha(W[2,4])*W[2,4] - w[2,4]
    a[8,4] = 0
    a[8,5] = 0
    a[8,6] = 0
    a[8,7] = r[2,4]*(-alpha(U[2,4])*U[2,4] - u[2,4])
    a[8,8] = r[2,4]*(alpha(U[2,5])*U[2,5] + u[2,5] - beta(U[2,4])*U[2,4] + u[2,4]) + w[2,4] - beta(W[2,4])*W[2,4] + w[3,4] + alpha(W[3,4])*W[3,4]
    a[8,9] = r[2,4]*(beta(U[2,5])*U[2,5] - u[2,5])
    a[8,10] = 0
    a[8,11] = 0
    a[8,12] = 0
    a[8,13] = beta(W[3,4])*W[3,4] - w[3,4]
    a[8,14] = 0
    
    # C25
    a[9,0] = 0
    a[9,1] = 0
    a[9,2] = 0
    a[9,3] = 0
    a[9,4] = -alpha(W[2,5])*W[2,5] - w[2,5]
    a[9,5] = 0
    a[9,6] = 0
    a[9,7] = 0
    a[9,8] = r[2,5]*(-alpha(U[2,5])*U[2,5] - u[2,5])
    a[9,9] = r[2,5]*(u[2,5] - beta(U[2,5])*U[2,5]) + w[2,5] - beta(W[2,5])*W[2,5] + w[3,5] + alpha(W[3,5])*W[3,5]
    a[9,10] = 0
    a[9,11] = 0
    a[9,12] = 0
    a[9,13] = 0
    a[9,14] = beta(W[3,5])*W[3,5] - w[3,5]
    
    # C31
    a[10,0] = 0
    a[10,1] = 0
    a[10,2] = 0
    a[10,3] = 0
    a[10,4] = 0
    a[10,5] = -alpha(W[3,1])*W[3,1] - w[3,1]
    a[10,6] = 0
    a[10,7] = 0
    a[10,8] = 0
    a[10,9] = 0
    a[10,10] = r[3,1]*(u[3,2] + alpha(U[3,2])*U[3,2]) + w[3,1] - beta(W[3,1])*W[3,1] + w[4,1] + alpha(W[4,1])*W[4,1]
    a[10,11] = r[3,1]*(beta(U[3,2])*U[3,2] - u[3,2])
    a[10,12] = 0
    a[10,13] = 0
    a[10,14] = 0
    
    # C32
    a[11,0] = 0
    a[11,1] = 0
    a[11,2] = 0
    a[11,3] = 0
    a[11,4] = 0
    a[11,5] = 0
    a[11,6] = -alpha(W[3,2])*W[3,2] - w[3,2]
    a[11,7] = 0
    a[11,8] = 0
    a[11,9] = 0
    a[11,10] = r[3,2]*(-alpha(U[3,2])*U[3,2] - u[3,2])
    a[11,11] = r[3,2]*(u[3,2] - beta(U[3,2])*U[3,2] + u[3,3] + alpha(U[3,3])*U[3,3]) + w[3,2] - beta(W[3,2])*W[3,2] + w[4,2] + alpha(W[4,2])*W[4,2]
    a[11,12] = r[3,2]*(beta(U[3,3])*U[3,3] - u[3,3])
    a[11,13] = 0
    a[11,14] = 0
    
    # C33
    a[12,0] = 0
    a[12,1] = 0
    a[12,2] = 0
    a[12,3] = 0
    a[12,4] = 0
    a[12,5] = 0
    a[12,6] = 0
    a[12,7] = -alpha(W[3,2])*W[3,2] - w[3,3]
    a[12,8] = 0
    a[12,9] = 0
    a[12,10] = 0
    a[12,11] = r[3,3]*(-alpha(U[3,3])*U[3,3] - u[3,3])
    a[12,12] = r[3,3]*(u[3,3] - beta(U[3,3])*U[3,3] + u[3,4] + alpha(U[3,4])*U[3,4]) + w[3,3] - beta(W[3,3])*W[3,3] + w[4,3] + alpha(W[4,3])*W[4,3]
    a[12,13] = r[3,3]*(beta(U[3,4])*U[3,4] -u[3,4])
    a[12,14] = 0
    
    # C34
    a[13,0] = 0
    a[13,1] = 0
    a[13,2] = 0
    a[13,3] = 0
    a[13,4] = 0
    a[13,5] = 0
    a[13,6] = 0
    a[13,7] = 0
    a[13,8] = -alpha(W[3,4])*W[3,4] - w[3,4]
    a[13,9] = 0
    a[13,10] = 0
    a[13,11] = 0
    a[13,12] = r[3,4]*(-alpha(U[3,4])*U[3,4] - u[3,4])
    a[13,13] = r[3,4]*(u[3,4] - beta(U[3,4])*U[3,4] + u[3,5] + alpha(U[3,5])*U[3,5]) + w[3,4] - beta(W[3,4])*W[3,4] + w[4,4] + alpha(W[4,4])*W[4,4]
    a[13,14] = r[3,4]*(beta(U[3,5])*U[3,5] -u[3,5])
    
    # C35
    a[14,0] = 0
    a[14,1] = 0
    a[14,2] = 0
    a[14,3] = 0
    a[14,4] = 0
    a[14,5] = 0
    a[14,6] = 0
    a[14,7] = 0
    a[14,8] = 0
    a[14,9] = -alpha(W[3,5])*W[3,5] - w[3,5]
    a[14,10] = 0
    a[14,11] = 0
    a[14,12] = 0
    a[14,13] = r[3,5]*(-alpha(U[3,5])*U[3,5] - u[3,5])
    a[14,14] = r[3,5]*(u[3,5] - beta(U[3,5])*U[3,5]) + w[3,5] - beta(W[3,5])*W[3,5] + w[4,5] + alpha(W[4,5])*W[4,5]
    
    return np.array(a)

# ___________ Define for different scenarios (before/after) ___________

# solver issue ' Matrix is singular' is only with:
    # a1_orig and a1_mir
# r1 and r2 are used in others that do work
    # perhaps it's the advection and dispersion patterns before a new barrier

# solution to above solver issue: it was the existing barrier patterns causing
    # the issue as the barrier had an obstruction of 100% and therefore advection
    # and dispersion values were set to 0 ue2/we2/ua2/wa2 arrays
    # changed obstruction value to 0.99 in those cases

# use A-matrix with different dispersion and advection values
# wind left to right, before new barrier
a1_orig = a_matrix(r = r1, u = ue2_orig, w = we2_orig, U = ua2_orig, W = wa2_orig)

# wind left to right, after new barrier
a2_orig = a_matrix(r = r1, u = ue3_orig, w = we3_orig, U = ua3_orig, W = wa3_orig)

# wind parallel, before new barrier, dimensioning from wind left to right
a1_par_1 = a_matrix(r = r1, u = ue1_par, U = ua1_par, w = we1_par, W = wa1_par)

# wind parallel, after new barrier, dimensioning from wind left to right
a2_par_1 = a_matrix(r = r1, u = ue3_par, U = ua3_par, w = we3_par, W = wa3_par)


# d1_orig = inputs for existing conditions, NO2
d1_orig = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype = float)
d1_orig[0] = ez_tot_no2_orig[1]/l_original[1] # averaging line emission source over area of 'ground' = (ug/m2/s)
d1_orig[1] = ez_tot_no2_orig[2]/l_original[2]
d1_orig[2] = ez_tot_no2_orig[3]/l_original[3]
d1_orig[3] = ez_tot_no2_orig[4]/l_original[4]
d1_orig[4] = ez_tot_no2_orig[5]/l_original[5]
d1_orig[5] = 0      # no inputs into middle boxes from either emissions or background
d1_orig[6] = 0
d1_orig[7] = 0
d1_orig[8] = 0
d1_orig[9] = 0
d1_orig[10] = (we2_orig[4,1] - beta(wa2_orig[4,1])*wa2_orig[4,1])*cB_no2  # background inputs into top boxes
d1_orig[11] = (we2_orig[4,2] - beta(wa2_orig[4,2])*wa2_orig[4,2])*cB_no2  # negative for beta because if true, the sign of W[4,x] would be negative
d1_orig[12] = (we2_orig[4,3] - beta(wa2_orig[4,3])*wa2_orig[4,3])*cB_no2  # yet as here being treated as an input, it should be positive, hence signs cancel out
d1_orig[13] = (we2_orig[4,4] - beta(wa2_orig[4,4])*wa2_orig[4,4])*cB_no2
d1_orig[14] = (we2_orig[4,5] - beta(wa2_orig[4,5])*wa2_orig[4,5])*cB_no2


# d2_orig = inputs for new conditions, NO2
d2_orig = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype = float)
d2_orig[0] = ez_tot_no2_orig[1]/l_original[1] # averaging line emission source over area of 'ground' = (ug/m2/s)
d2_orig[1] = ez_tot_no2_orig[2]/l_original[2]
d2_orig[2] = ez_tot_no2_orig[3]/l_original[3]
d2_orig[3] = ez_tot_no2_orig[4]/l_original[4]
d2_orig[4] = ez_tot_no2_orig[5]/l_original[5]
d2_orig[5] = 0      # no inputs into middle boxes from either emissions or background
d2_orig[6] = 0
d2_orig[7] = 0
d2_orig[8] = 0
d2_orig[9] = 0
d2_orig[10] = (we3_orig[4,1] - beta(wa3_orig[4,1])*wa3_orig[4,1])*cB_no2  # background inputs into top boxes
d2_orig[11] = (we3_orig[4,2] - beta(wa3_orig[4,2])*wa3_orig[4,2])*cB_no2  # negative for beta because if true, the sign of W[4,x] would be negative
d2_orig[12] = (we3_orig[4,3] - beta(wa3_orig[4,3])*wa3_orig[4,3])*cB_no2  # yet as here being treated as an input, it should be positive, hence signs cancel out
d2_orig[13] = (we3_orig[4,4] - beta(wa3_orig[4,4])*wa3_orig[4,4])*cB_no2
d2_orig[14] = (we3_orig[4,5] - beta(wa3_orig[4,5])*wa3_orig[4,5])*cB_no2



###############################################################################
# d1_orig_par = inputs for existing conditions, NO2
d1_orig_par = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype = float)
d1_orig_par[0] = ez_tot_no2_orig[1]/l_original[1] # averaging line emission source over area of 'ground' = (ug/m2/s)
d1_orig_par[1] = ez_tot_no2_orig[2]/l_original[2]
d1_orig_par[2] = ez_tot_no2_orig[3]/l_original[3]
d1_orig_par[3] = ez_tot_no2_orig[4]/l_original[4]
d1_orig_par[4] = ez_tot_no2_orig[5]/l_original[5]
d1_orig_par[5] = 0      # no inputs into middle boxes from either emissions or background
d1_orig_par[6] = 0
d1_orig_par[7] = 0
d1_orig_par[8] = 0
d1_orig_par[9] = 0
d1_orig_par[10] = (we1_par[4,1] - beta(wa1_par[4,1])*wa1_par[4,1])*cB_no2  # background inputs into top boxes
d1_orig_par[11] = (we1_par[4,2] - beta(wa1_par[4,2])*wa1_par[4,2])*cB_no2  # negative for beta because if true, the sign of W[4,x] would be negative
d1_orig_par[12] = (we1_par[4,3] - beta(wa1_par[4,3])*wa1_par[4,3])*cB_no2  # yet as here being treated as an input, it should be positive, hence signs cancel out
d1_orig_par[13] = (we1_par[4,4] - beta(wa1_par[4,4])*wa1_par[4,4])*cB_no2
d1_orig_par[14] = (we1_par[4,5] - beta(wa1_par[4,5])*wa1_par[4,5])*cB_no2


# d2_orig_par = inputs for new conditions, NO2
d2_orig_par = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype = float)
d2_orig_par[0] = ez_tot_no2_orig[1]/l_original[1] # averaging line emission source over area of 'ground' = (ug/m2/s)
d2_orig_par[1] = ez_tot_no2_orig[2]/l_original[2]
d2_orig_par[2] = ez_tot_no2_orig[3]/l_original[3]
d2_orig_par[3] = ez_tot_no2_orig[4]/l_original[4]
d2_orig_par[4] = ez_tot_no2_orig[5]/l_original[5]
d2_orig_par[5] = 0      # no inputs into middle boxes from either emissions or background
d2_orig_par[6] = 0
d2_orig_par[7] = 0
d2_orig_par[8] = 0
d2_orig_par[9] = 0
d2_orig_par[10] = (we3_par[4,1] - beta(wa3_par[4,1])*wa3_par[4,1])*cB_no2  # background inputs into top boxes
d2_orig_par[11] = (we3_par[4,2] - beta(wa3_par[4,2])*wa3_par[4,2])*cB_no2  # negative for beta because if true, the sign of W[4,x] would be negative
d2_orig_par[12] = (we3_par[4,3] - beta(wa3_par[4,3])*wa3_par[4,3])*cB_no2  # yet as here being treated as an input, it should be positive, hence signs cancel out
d2_orig_par[13] = (we3_par[4,4] - beta(wa3_par[4,4])*wa3_par[4,4])*cB_no2
d2_orig_par[14] = (we3_par[4,5] - beta(wa3_par[4,5])*wa3_par[4,5])*cB_no2



###############################################################################
# CALCULATE & SEND DATA BACK
###############################################################################

# error check before sending data back
# if there are any error flags pass back the error message and an array of NaN
# otherwise, calculate the percentage change

empty_array = np.empty((3,5))
empty_array[:] = np.NaN
empty_array = empty_array.tolist()


if error[0,0] == 1:
    print("Row dimensioning error: generic")
    print(empty_array)
elif error[0,1] == 1:
    print("Row dimensioning error 1: no buildings present")
    print(empty_array)
elif error[0,2] == 1:
    print("Row dimensioning error 2: only 1 building present")
    print(empty_array)
elif error[0,3] == 1:
    print("Row dimensioning error 3: green infrastructure proposed is the same height or taller than at least one building")
    print(empty_array)
elif error[0,4] == 1:
    print("Row dimensioning error 4: existing barrier is the same height or taller than at least one building")
    print(empty_array)
elif error[1,1] == 1:
    print("Column dimensioning error 1.1: column 2 further than far edge of emission zone")
    print(empty_array)
elif error[1,2] == 1:
    print("Column dimensioning error 1.2: column 4 extends further than downwind building")
    print(empty_array)
elif error[2,1] == 1:
    print("Column dimensioning error 2.1: column 2 extends beyond far edge of emission zone")
    print(empty_array)
elif error[2,2] == 1:
    print("Column dimensioning error 2.2: column 4 extends beyond downwind building")
    print(empty_array)
elif error[3,1] == 1:
    print("Column dimensioning error 3.1: upwind existing barrier recirculation extends beyond far limit of emission zone")
    print(empty_array)
elif error[3,2] == 1:
    print("Column dimensioning error 3.2: upwind new barrier recirculation extends beyond far limit of emission zone")
    print(empty_array)
elif error[3,3] == 1:
    print("Column dimensioning error 3.3: upwind new barrier recirculation extends beyond far limit of emission zone")
    print(empty_array)
elif error[3,4] == 1:
    print("Column dimensioning error 3.4: column 4 extends beyond downwind building")
    print(empty_array)
elif error[4,1] == 1:
    print("Column dimensioning error 4.1: existing barrier upwind is taller than upwind building")
    print(empty_array)
elif error[4,2] == 1:
    print("Column dimensioning error 4.2: new barrier upwind taller than upwind building")
    print(empty_array)
elif error[5,1] == 1:
    print("Column dimensioning error 5.1: existing barrier upwind taller than upwind building")
    print(empty_array)
elif error[5,2] == 1:
    print("Column dimensioning error 5.2: new barrier upwind is taller than upwind building")
    print(empty_array)
elif error[7,1] == 1:
    print("Column dimensioning error: 7.1: total column widths do not equal road width (wind left to right)")
    print(empty_array)
elif error[7,2] == 1:
    print("Column dimensioning error: 7.2: total column widths do not equal road width  (wind right to left)")
    print(empty_array)
elif error[8,1] == 1:
    print("Emissions error 1: partitioned emissions do not equal total emissions")
    print(empty_array)
else:
    # if there are no errors, solve for existing and new conditions
    print("")
    print("********************************************")
    print("IMPORTANT NOTE: % changes are reported in box order from bottom left to top right. Therefore first 5 results refer to R1 C1-5, results 5-10 refer to R2 C1-5, and results 10-15 refer to R3 C1-5")
    print("********************************************")
    print("")
    # _________________________________________________________________________
    # WIND: LEFT TO RIGHT
    # NO2
    C1_orig = solve(a1_orig, d1_orig)
    C2_orig = solve(a2_orig, d2_orig)
    
    #print("Concentrations before:", C1_orig, sep = "\n")
    #print("Concentration after:", C2_orig, sep = "\n")
    
    # calculate the percentage changes in concentrations before and after
    # NO2
    per_change_no2_orig = ((C2_orig - C1_orig)/C1_orig)*100
    print("SF6 % Change (wind L->R)", per_change_no2_orig, sep = '\n')
    print("")
 
    

    # _________________________________________________________________________
    # WIND: PARALLEL
    # NO2
    # dimensioning from left to right
    C1_par_1 = solve(a1_par_1, d1_orig_par)
    C2_par_1 = solve(a2_par_1, d2_orig_par)
    

    # % change of NO2 based on dispersion only (parallel wind) with dimensioning from wind left-to-right
    per_change_no2_par_1 = ((C2_par_1 - C1_par_1)/C1_par_1)*100
   
    print("SF6 % Change (Parallel wind, dimensioning: L->R)", per_change_no2_par_1, sep = '\n')
   
    


# #print("SF6 % Change (Parallel wind, dimensioning: L->R)", per_change_no2_par_1, sep = '\n')
# import csv
# colnames = ["barrier_position", "barrier_height", "barrier_obstruction", "wind", "C11", "C12", "C13", "C14", "C15", "C21", "C22", "C23", "C24", "C25", "C31", "C32", "C33", "C34", "C35"]
# par_array = np.array([gi_location_text, gi_height_filename, gi_obstruction_filename, "parallel"])
# per_array = np.array([gi_location_text, gi_height_filename, gi_obstruction_filename, "perpendicular"])
# output_par_array = np.append(par_array, per_change_no2_par_1)
# output_per_array = np.append(per_array, per_change_no2_orig)
# #print(output_array)

# filename_made = gi_location_text + "_" + gi_height_filename + "_" + gi_obstruction_filename + ".csv"
# print(filename_made)

# with open(filename_made, 'w', newline='') as csvfile:
#     aqwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
#     aqwriter.writerow(colnames)
#     aqwriter.writerow(output_par_array)
#     aqwriter.writerow(output_per_array)
    
###############################################################################
# USEFUL PARAMETERS TO PRINT TO THE DISPLAY TO CHECK THAT THE CODE RUNS CORRECTLY
    # - UNCOMMENT AS NECESSARY
    
#print(gi)
#print("Row:", row)
#print("Zone:", zone)
#print("Bar:", bar)
#print("Check", check)
#print("Recirc:", rec)
#print("Columns:",l_cumu_original)
#print("Rows:", h_cumu_original)
    
#print("Rows covered by recirculation of upwind building:",rec_nrow) 
#print("Columns fully covered by recirculation of upwind building:",rec_ncol)

#print("U1:", U1)
#print("U2:", U2)
#print("U3:", U3)
#print("Uh:", Uh)
#print("Ur:", Ur)
#print("Ut:", Ut)

#print("Horizontal advection NO BARRIERS:", np.around(ua1_orig,3), sep = '\n')
#print("Vertical advection NO BARRIERS:", np.around(wa1_orig,3), sep = '\n')
#print("Horizontal dispersion NO BARRIERS:", np.around(ue1,3), sep = '\n')
#print("Vertical dispersion NO BARRIERS:", np.around(we1,3), sep = '\n')

#print("Horizontal advection after existing barriers:", np.around(ua2,3), sep = '\n')
#print("Vertical advection after existing barriers:", np.around(wa2,3), sep = '\n')
#print("Horizontal dispersion after existing barriers:", np.around(ue2,3), sep = '\n')
#print("Vertical dispersion after existing barriers:", np.around(we2,3), sep = '\n')

#print("Horizontal advection after GI:", np.around(ua3_orig,3),sep = '\n')
#print("Vertical advection after GI:", np.around(wa3_orig,3),sep = '\n')
#print("Horizontal dispersion after GI:", np.around(ue3,3),sep = '\n')
#print("Vertical dispersion after GI:", np.around(we3,3),sep = '\n')
