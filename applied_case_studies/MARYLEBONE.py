#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:04:01 2019

@author: HelenPearce
"""
###############################################################################
# MARYLEBONE
# ADAPTED TO COMPARE DIRECTLY TO CERC (HOOD et al 2021)

###############################################################################
# HOUSE-KEEPING
###############################################################################

# import necessary packages
import numpy as np
from scipy.linalg import solve
import math as math
from statistics import mean


# for met data
import pandas as pd


# set container for error flags - this will be checked before final calculations
error = np.zeros((10,10))

###############################################################################
# BACKGROUND
###############################################################################

# ___________ extract background pollution concentrations ___________
# MANUAL
# source: Defra background - same link as used in the platform form

# 2012 background (to tie in with CERC paper)
cB_no2 = 40.03 # UNIT: ug/m3 
cB_pm25 = 17.15 # UNIT: ug/m3 


print("BACKGROUND CONCS.")
print("NO2:", cB_no2)
print("PM2.5:", cB_pm25)
print("")


###############################################################################
# WIND
###############################################################################

# ___________ climatological wind characteristics ___________

# ___________ choose closest met station ______________
# unnecessary in manual version - just read in file


# _______________ calculate weighted L-R and R-L cross-canyon wind speed (u) _______________

# convert street direction to degrees
# MANUAL
street_dir = 254

# read in Heathrow 2012 meteorology
csv_path = 'https://raw.githubusercontent.com/GI4RAQ/GI4RAQ-open/master/applied_case_studies/met_heathrow_2012.csv'

# read in wind data from folder
wind = pd.read_csv(csv_path)

# visual inspection
#print(wind)

# wind stored in a pandas dataframe
# developing with a pandas dataframe is different to in R, safest way to 
# calculate things is to define a function first and then apply it using 
# the lambda method. 
# Also remember .copy() if making a new dataframe - ensures not just 'viewing'
# a dataframe but actually making alterations


#print(wind)
# REPLACE slack wind speed with 0.5 m/s and assign the same direction as the street
# so this is representative of total along-street component
slack = wind['wind_speed'][0]
wind['wind_speed'] = wind['wind_speed'].replace([slack],0.5)
wind['wind_direction'] = wind['wind_direction'].replace([0],street_dir)
#print(wind)

# firstly, create a new column in the wind dataframe that contains modified angles as if 
# the street was pointing northerly (i.e. at 0 degrees)
# the was done to help calculate the wind sectors that are left-to-right and
# right to left, as these angles will always been in the same relative locations
# there is also a check to ensure the angle calculated is always between 0 and 360
def rotate_street(wind_d):
    angle =  wind_d - street_dir
    if angle >= 360:
        angle2 = angle-360
        return angle2
    elif angle <0:
        angle2 = angle+360
        return angle2
    else:
        return angle

wind['street_dir_as_N'] = wind.apply(lambda wind: rotate_street(wind['wind_direction']), axis=1)
#print(wind)

# secondly, calculate theta: the difference between the perpendicular angle 
# (relative to the street) and the relative wind directions for both the L-R and R-L cases
# relative perpendicular wind at 270 degrees for L-R, and 90 degrees for R-L
def dif_to_perp(perp,wind_d):
    if perp == 360:
        if wind_d < 180:
            a = 0+wind_d
        else:
            a=360-wind_d
        
        return a
        
    else:
        return abs(perp-wind_d)

wind['LR_difference'] = wind.apply(lambda wind: dif_to_perp(270, wind['street_dir_as_N']), axis=1)
#print(wind)
wind['RL_difference'] = wind.apply(lambda wind: dif_to_perp(90, wind['street_dir_as_N']), axis=1)
#print(wind)
wind['par1_difference'] = wind.apply(lambda wind: dif_to_perp(360, wind['street_dir_as_N']), axis=1)
#print(wind)
wind['par2_difference'] = wind.apply(lambda wind: dif_to_perp(180, wind['street_dir_as_N']), axis=1)
#print(wind)

# create subsets that only include data that is relevant for each sector 
# i.e. 45 +/- from the perpendicular
LR_only = wind.loc[wind.LR_difference <= 45].copy()
RL_only = wind.loc[wind.RL_difference <= 45].copy()
par1_only = wind.loc[wind.par1_difference <= 45].copy()
par2_only = wind.loc[wind.par2_difference <= 45].copy()

#print(LR_only)
#print(RL_only)

# convert theta into radians as this is what math.cos assumes
def calculate_radians(degrees):
    return math.radians(degrees)

LR_only["LR_dif_rad"] = LR_only.apply(lambda LR_only: calculate_radians(LR_only['LR_difference']), axis=1)
RL_only["RL_dif_rad"] = RL_only.apply(lambda RL_only: calculate_radians(RL_only['RL_difference']), axis=1)
par1_only["par_dif_rad"] = par1_only.apply(lambda par1_only: calculate_radians(par1_only['par1_difference']), axis=1)
par2_only["par_dif_rad"] = par2_only.apply(lambda par2_only: calculate_radians(par2_only['par2_difference']), axis=1)

# print(LR_only)
# print(RL_only)
# print(par1_only)
# print(par2_only)

# calculate the perpendicular component (across canyon) of each of the wind vectors

# IMPORTANT: the angles calculated only need to be relative to the perpendicular
# but need to still use the original wind speed values from the original wind directions
# that they represent

# cosine used as the hypotenuse of a right-angled triangle is the original wind speed
# the acute angle has been calculated (theta - relative to the perpendicular)
# and the adjacent side of the right-angled triangle represents the cross-street
# component of the wind vector

# for the along street speeds, cosine is still used because the adjacent component is now
# orientated parallel to the street (i.e. the reference angle has changed through 90)
def calculate_u(theta, wind_speed):
    return math.cos(theta)*wind_speed

LR_only["LR_u"] = LR_only.apply(lambda LR_only: calculate_u(LR_only['LR_dif_rad'],LR_only['wind_speed']), axis=1)
RL_only["RL_u"] = RL_only.apply(lambda RL_only: calculate_u(RL_only['RL_dif_rad'],RL_only['wind_speed']), axis=1)
par1_only["par_u"] = par1_only.apply(lambda par1_only: calculate_u(par1_only['par_dif_rad'],par1_only['wind_speed']), axis=1)
par2_only["par_u"] = par2_only.apply(lambda par2_only: calculate_u(par2_only['par_dif_rad'],par2_only['wind_speed']), axis=1)

# weighting of wind sectors needed to give greater 'weight' to the wind speeds
# and directions that occur more frequently, and vice versa
# firstly calculate how frequently the total sector of L-R and R-L occurs
LR_freq = LR_only['fractional_occur'].sum()
RL_freq = RL_only['fractional_occur'].sum()

parallel = pd.concat([par1_only, par2_only])
#print(parallel)
par_freq = parallel['fractional_occur'].sum()
#print(par_freq)

def weighted_u(foccur,u,dir_freq):
    return (foccur*u)/dir_freq

LR_only["LR_u_weighted"] = LR_only.apply(lambda LR_only: weighted_u(LR_only['fractional_occur'], LR_only['LR_u'],LR_freq), axis=1)
RL_only["RL_u_weighted"] = RL_only.apply(lambda RL_only: weighted_u(RL_only['fractional_occur'], RL_only['RL_u'],RL_freq), axis=1)
parallel["par_u_weighted"] = parallel.apply(lambda parallel: weighted_u(parallel['fractional_occur'], parallel['par_u'],par_freq), axis=1)

# finally sum the fractional wind speed components
LR_u = LR_only['LR_u_weighted'].sum()
RL_u = RL_only['RL_u_weighted'].sum()
par_u = parallel['par_u_weighted'].sum()
#print(LR_u, RL_u, par_u)


# checking
#print(LR_only['wind_direction'], LR_only['street_dir_as_N'])
#print(LR_only['LR_difference'])
#print(LR_only['LR_dif_rad'])
#print(LR_only['wind_speed'])
print("WIND CONDITIONS")
print("Background wind (ubg) left to right:", LR_u)


#print(RL_only['wind_direction'], RL_only['street_dir_as_N'])
#print(RL_only['RL_difference'])
#print(RL_only['RL_dif_rad'])
#print(RL_only['wind_speed'])
print("Background wind (ubg) right to left:", RL_u)
print("Parallel wind (ubg_par)", par_u)

# _______________ assign values for use further in code _______________

# wind speed left to right (m/s) - climatological average
ubg_orig = LR_u

# wind speed right to left (m/s) - climatological average
ubg_mir = RL_u

ubg_parallel = par_u

LR_par_freq = par_freq/2
RL_par_freq = par_freq/2

print("L-->R Frequency:",LR_freq)
print("R-->L Frequency:", RL_freq)
print("Parallel Frequency:", par_freq)
print("Total time accounted for:", (LR_freq + RL_freq + par_freq))
print("")
#sum([LR_freq, RL_freq, LR_par_freq, RL_par_freq])





###############################################################################
# GEOMETRY: WIND ACROSS STREET LEFT --> RIGHT (original)
###############################################################################

# WHOLE SECTION IS MANUAL 

# ___________ total street width ___________

roadw= 41
# note: if you change total street width, will need to also manually adjust columns 

# ___________ set horizontal (row_original) defining values ___________

# row_original = storage container initialised with zeros
row_original = np.array([0,0,0,0,0,0,0,0,0,0], dtype = float)

# left hand side building height from Hood et al 2021
row_original[0]=23.5

# right hand side building height - keep the same as is treated in Hood et al model
row_original[1]=23.5


# NOTE: if you change building heights, will need to manually adjust rows

# ___________ set vertical defining values ___________

# zone_original = storage container
zone_original = np.array([0,0,0,0,0], dtype = float)


# upwind street boundary
# SOURCE: OS MEASURE TOOL
zone_original[0] = 7


# determine where emission zones start and end
# SOURCE: OS MEASURE TOOL
ez1_start = 13.5
ez1_finish = 22
ez2_start = 24.5
ez2_finish = 34.25
ez1_w = 8.5
ez2_w = 9.75

# determine number of emission zones
nez = 2
    
# determine upwind near edge and downwind far edge of emission zones 
zone_original[1] = 0   
zone_original[2] = 0

# if there's only 1 emission zone, use ez1 start and finish
# but if there are 2 emission zones, use ez1 start and ez2 finish
if nez == 1:
    zone_original[1] = ez1_start
    zone_original[2] = ez1_finish
elif nez == 2:
    zone_original[1] = ez1_start
    zone_original[2] = ez2_finish

# downwind street boundary   
zone_original[3] = 39.5

# downwind building
zone_original[4] = roadw 

#print(zone_original)

# barrier locations (m from left of street)
# bar_original = a strage container
bar_original = np.array([0,0,0,0], dtype = float)

# check presence of barriers in locations (1 = true, 0 = false)
check_original = np.array([0,0,0,0], dtype = float)

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


#print("Row: left to right dimensioning:", row_original)
#print("Zone: left to right dimensioning:", zone_original)
#print("Bar: left to right dimensioning:", bar_original)
#print("Check: left to right dimensioning:", check_original)
#print("Rec: left to right dimensioning:", rec_original)

###############################################################################
# GEOMETRY: WIND ACROSS STREET RIGHT --> LEFT (mirror)
###############################################################################
# row and column dimensioning are based on arrays that have stored important
# location data above, namely:
    # row, zone, bar, check, rec
# We now need to create a mirror image of these values to represent the wind
    # flowing right to left across the street

# initialise new empty containers
row_mirror = np.array([0,0,0,0,0,0,0,0,0,0], dtype = float)
zone_mirror = np.array([0,0,0,0,0], dtype = float)
bar_mirror = np.array([0,0,0,0], dtype = float)
check_mirror = np.array([0,0,0,0], dtype = float)
rec_mirror = np.array([0,0,0,0,0], dtype = float)

# ___________ reverse 'row' ___________
# row[0] = upwind building height, row[1] = downwind building height
# flip the upwind and downwind building heights
row_mirror[0] = row_original[1]
row_mirror[1] = row_original[0]

# all gi heights will remain the same
row_mirror[2] = row_original[2]
row_mirror[3] = row_original[3]
row_mirror[4] = row_original[4]
row_mirror[5] = row_original[5]

# existing barriers upwind/downwind flip over
row_mirror[8] = row_original[9]
row_mirror[9] = row_original[8]

# ___________ reverse 'zone' ___________
# upwind street boundary
zone_mirror[0] = roadw - zone_original[3]
# upwind edge of emission zone
zone_mirror[1] = roadw - zone_original[2]
# downwind edge of emission zone
zone_mirror[2] = roadw - zone_original[1]
# downwind street boundary
zone_mirror[3] = roadw - zone_original[0]
# downwind building
zone_mirror[4] = roadw


# ___________ reverse 'rec' ___________

recirc = 0

# upwind building recirc
# if there is building upwind the recirculation is calculated by 3H-3
# however if this is less than or equal to 0, 0.01 is assigned
# if there is no building upwind the recirc = 0 
# (done this way to avoid adding 0.01 as a recirc if there's no building)
if row_mirror[0] > 0:
    recirc = (row_mirror[0]*3)-3
    if recirc <= 0:
        recirc = 0.01    
    rec_mirror[0] = recirc
else:
    rec_mirror[0] = 0


#print("Row: right to left dimensioning:", row_mirror)
#print("Zone: right to left dimensioning:", zone_mirror)
#print("Bar: right to left dimensioning:", bar_mirror)
#print("Check: right to left dimensioning:", check_mirror)
#print("Rec: right to left dimensioning:", rec_mirror)
    
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
row_mirror = rounding_row(row_input=row_mirror)

def rounding_5(data):
    data[0] = round(data[0], 4)
    data[1] = round(data[1], 4)
    data[2] = round(data[2], 4)
    data[3] = round(data[3], 4)
    data[4] = round(data[4], 4)
    return data

zone_original = rounding_5(data=zone_original)
zone_mirror = rounding_5(data=zone_mirror)
rec_original = rounding_5(data=rec_original)
rec_mirror = rounding_5(data=rec_mirror)

def rounding_4(data):
    data[0] = round(data[0], 4)
    data[1] = round(data[1], 4)
    data[2] = round(data[2], 4)
    data[3] = round(data[3], 4)
    return data

bar_original = rounding_4(data=bar_original)
bar_mirror = rounding_4(data=bar_mirror)

#print(bar_original, bar_mirror)
    
    
###############################################################################
# ROW DIMENSIONING
###############################################################################
# MANUAL
h_original = np.array([0,0,0,0], dtype = float)
h_original[1] = 5
h_original[2] = 9.25
h_original[3] = 9.25

h_mirror = np.array([0,0,0,0], dtype = float)
h_mirror[1] = 5
h_mirror[2] = 9.25
h_mirror[3] = 9.25

print("DIMENSIONING")
print("Row widths (original):", h_original)
print("Row widths (mirror):", h_mirror)

###############################################################################
# COLUMN DIMENSIONING
###############################################################################

# MANUAL
# assign widths of each column (not cumulative point distance)
l_original = np.array([0,0,0,0,0,0], dtype = float)
l_original[1] = 8
l_original[2] = 5.5
l_original[3] = 9.75
l_original[4] = 9.75
l_original[5] = 8

l_mirror = np.array([0,0,0,0,0,0], dtype = float)
l_mirror[1] = 8
l_mirror[2] = 9.75
l_mirror[3] = 9.75
l_mirror[4] = 5.5
l_mirror[5] = 8

print("Column widths (original):", l_original)
print("Column widths (mirror):", l_mirror)
print("")

# check that all the columns add up to total road width
if round(sum(l_original),4) != roadw:
    error[7,1] = 1

if round(sum(l_mirror),4) != roadw:
    error[7,2] = 1

# ___________ format column and row output ___________

# convert to l and h to lists, otherwise you get the error: 
# Object of type 'ndarray' is not JSON serializable

# these provide a list of widths/heights of individual columns/rows
l_original_list = l_original.tolist()
h_original_list = h_original.tolist()

l_mirror_list = l_mirror.tolist()
h_mirror_list = h_mirror.tolist()


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


# calculate cumulative distance in m from left / bottom:
l_cumu_mirror = np.array([0,0,0,0,0,0], dtype = float)
l_cumu_mirror[1] = round(l_mirror[1], 4)
l_cumu_mirror[2] = round(l_mirror[1]+l_mirror[2], 4)
l_cumu_mirror[3] = round(l_mirror[1]+l_mirror[2]+l_mirror[3], 4)
l_cumu_mirror[4] = round(l_mirror[1]+l_mirror[2]+l_mirror[3]+l_mirror[4], 4)
l_cumu_mirror[5] = round(l_mirror[1]+l_mirror[2]+l_mirror[3]+l_mirror[4]+l_mirror[5], 4)
l_cumu_mirror_list = l_cumu_mirror.tolist()

h_cumu_mirror = np.array([0,0,0,0], dtype = float)
h_cumu_mirror[1] = round(h_mirror[1],4)
h_cumu_mirror[2] = round(h_mirror[1]+h_mirror[2],4)
h_cumu_mirror[3] = round(h_mirror[1]+h_mirror[2]+h_mirror[3],4)
h_cumu_mirror_list = h_cumu_mirror.tolist()

#print(l_cumu_original_list)
#print(h_cumu_original_list)
#print(l_cumu_mirror_list)
#print(h_cumu_mirror_list)

###############################################################################
# ADVECTION & DISPERSION PATTERNS
###############################################################################

# ___________ obstruction values for barriers ___________

# obstruction array
obs_original = np.array([0,0,0,0], dtype = float)

obs_mirror = np.array([0,0,0,0], dtype = float)


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
    
# if the upwind building is aligned with row 2:
if row_mirror[0] == h_cumu_mirror[2]:
    rec_nrow_mir = 2
    rec_ncol_mir = recirc_col(recirc = rec_mirror[0], l_cumu = l_cumu_mirror)
# if the upwind building is aligned with row 3:
elif row_mirror[0] == h_cumu_mirror[3]:
    # ft is reduced; flow within canyon is slower than background wind speed
    rec_nrow_mir = 3
    rec_ncol_mir = recirc_col(recirc = rec_mirror[0], l_cumu = l_cumu_mirror)

#print(rec_nrow_orig, rec_ncol_orig)
#print(rec_nrow_mir, rec_ncol_mir)

# ___________ dynamic assignment of wind speeds within canyon ___________

# H = upwind building height
H_orig = row_original[0]
H_mir = row_mirror[0]
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
    uh = ubg*(math.log(5000)/math.log(500))*(math.log(5*H-5*d)/math.log(500-5*d))
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
        

U3_orig = ws_average(row_min = h_cumu_original[2], row_max = h_cumu_original[3], ubg = ubg_orig, H = H_orig)
U2_orig = ws_average(row_min = h_cumu_original[1], row_max = h_cumu_original[2], ubg = ubg_orig, H = H_orig)
U1_orig = ws_average(row_min = 0, row_max = h_cumu_original[1], ubg = ubg_orig, H = H_orig)
Uh_orig = ws_point(ubg = ubg_orig, z = H_orig, H = H_orig, w = w)
#print("Uh_orig:", Uh_orig)
Ur_orig = (0.1*Uh_orig)*(H_orig/(2*h_original[rec_nrow_orig]))
#print("UR_orig:", Ur_orig)
Ut_orig = ws_point(ubg = ubg_orig,z=max(row_original[0],row_original[1]),H = H_orig, w = w)

U3_mir = ws_average(row_min = h_cumu_mirror[2], row_max = h_cumu_mirror[3], ubg = ubg_mir, H = H_mir)
U2_mir = ws_average(row_min = h_cumu_mirror[1], row_max = h_cumu_mirror[2], ubg = ubg_mir, H = H_mir)
U1_mir = ws_average(row_min = 0, row_max = h_cumu_mirror[1], ubg = ubg_mir, H = H_mir)
Uh_mir = ws_point(ubg = ubg_mir, z = H_mir, H = H_mir, w = w)
#print("Uh_mir:", Uh_mir)
Ur_mir = (0.1*Uh_mir)*(H_mir/(2*h_mirror[rec_nrow_mir]))
#print("UR_mir:", Ur_mir)
Ut_mir = ws_point(ubg = ubg_mir,z=max(row_mirror[0],row_mirror[1]),H = H_mir, w = w)

#print(Ut_orig, Uh_orig, Ur_orig, U3_orig, U2_orig, U1_orig)
#print(Ut_mir, Uh_mir, Ur_mir, U3_mir, U2_mir, U1_mir)


# ___________ prepare empty containers ___________

# base patterns: left to right wind
ua1_orig = np.zeros((5,6)) 	# horizontal advection velocities
wa1_orig = np.zeros((5,6))   # vertical advection velocities
ue1_orig = np.zeros((5,6))   # horizontal dispersion velocities
we1_orig = np.zeros((5,6))   # vertical dispersion velocities

ua1_mir = np.zeros((5,6)) 	# horizontal advection velocities
wa1_mir = np.zeros((5,6))   # vertical advection velocities
ue1_mir = np.zeros((5,6))   # horizontal dispersion velocities
we1_mir = np.zeros((5,6))   # vertical dispersion velocities


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

no_barriers_mir = no_barriers_pattern(row=row_mirror, h_cumu=h_cumu_mirror, 
                                rec_ncol=rec_ncol_mir, wa1=wa1_mir, we1=we1_mir, 
                                ua1=ua1_mir, ue1=ue1_mir, U1=U1_mir, U2=U2_mir, 
                                U3=U3_mir, Ut=Ut_mir, Uh=Uh_mir, Ur=Ur_mir, 
                                l=l_mirror, h=h_mirror)
ue1_mir = no_barriers_mir[0]
ua1_mir = no_barriers_mir[1]
we1_mir = no_barriers_mir[2]
wa1_mir = no_barriers_mir[3]

# print("ue1_mir")
# print(ue1_mir)
# print("ua1_mir")
# print(ua1_mir)
# print("we1_mir")
# print(we1_mir)
# print("wa1_mir")
# print(wa1_mir)


# ___________ Advection & Dispersion Assignment: FUNCTIONS for barriers ___________

# MANUAL: REMOVED FOR MARYLEBONE SLIM VERSION 

# ___________ Advection & Dispersion Assignment: EXISTING BARRIERS ___________

# copy existing patterns into new storage containers
    # MANUAL: with no barriers defined for Marylebone, can just assume same as above
ue2_orig = ue1_orig.copy()
ua2_orig = ua1_orig.copy()
we2_orig = we1_orig.copy()
wa2_orig = wa1_orig.copy()

ue2_mir = ue1_mir.copy()
ua2_mir = ua1_mir.copy()
we2_mir = we1_mir.copy()
wa2_mir = wa1_mir.copy()




# ___________ Parallel wind: dispersion only ___________

# we don't want any influence of recirculation zones; for these to occur wind 
# would need to be across the street

def ws_point_parallel(ubg,z,H):
    # firstly set d
    d = 0
    
    # calculate wind speed (u) at the height of shortest building (where log profile is still valid to)
    uh = ubg*(math.log(5000)/math.log(500))*(math.log(5*H-5*d)/math.log(500-5*d))
    
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


#test = ws_point_parallel(ubg = ubg_parallel, z=50,H=min(row_original[0],row_original[1]))
#print(test)

#Uh_parallel = ws_point_parallel(ubg = ubg_parallel, z = min(row_original[0],row_original[1]), H = min(row_original[0],row_original[1]))
# assumes that the rows don't change between original and mirror dimensioning
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

# apply linear reductions of ACH from roof top for both horizontal and vertical
ue2_par[1,1] = 0 #c11 to wall
ue2_par[1,2] = ue_row1*0.1 #c11 and c12
ue2_par[1,3] = ue_row1*0.1 #c12 and c13
ue2_par[1,4] = ue_row1*0.1 #c13 and c14
ue2_par[1,5] = ue_row1*0.1 #c14 and c15
ue2_par[2,1] = 0 #c21 to wall
ue2_par[2,2] = ue_row2*0.1 #c21 and c22
ue2_par[2,3] = ue_row2*0.1 #c22 and c23
ue2_par[2,4] = ue_row2*0.1 #c23 and c24
ue2_par[2,5] = ue_row2*0.1 #c24 and c25
ue2_par[3,1] = 0 #c31 to wall
ue2_par[3,2] = ue_row3*0.1 #c31 and c32
ue2_par[3,3] = ue_row3*0.1 #c32 and c33
ue2_par[3,4] = ue_row3*0.1 #c33 and c34
ue2_par[3,5] = ue_row3*0.1 #c34 and c35

#vertical dispersion co-efficients
#we[1,1] = 0 #ground to c11
#we[1,2] = 0 #ground to c12
#we[1,3] = 0 #ground to c13
#we[1,4] = 0 #ground to c14
#we[1,5] = 0 #ground to c15
we2_par[2,1] = we_row12*0.1 #c11 to c21
we2_par[2,2] = we_row12*0.1 #c12 and c22
we2_par[2,3] = we_row12*0.1 #c13 and c23
we2_par[2,4] = we_row12*0.1 #c13 and c24
we2_par[2,5] = we_row12*0.1 #c15 and c25
we2_par[3,1] = we_row23*0.1 #c21 and c31
we2_par[3,2] = we_row23*0.1 #c22 and c32
we2_par[3,3] = we_row23*0.1 #c23 and c33
we2_par[3,4] = we_row23*0.1 #c24 and c34
we2_par[3,5] = we_row23*0.1 #c25 and c35
we2_par[4,1] = we_row3b*0.1 #c31 and cb
we2_par[4,2] = we_row3b*0.1 #c32 and cb
we2_par[4,3] = we_row3b*0.1 #c33 and cb
we2_par[4,4] = we_row3b*0.1 #c24 and cb
we2_par[4,5] = we_row3b*0.1 #c35 and cb


# print("ue1_par")
# print(ue1_par)
# print("ua1_par")
# print(ua1_par)
# print("we1_par")
# print(we1_par)
# print("wa1_par")
# print(wa1_par)

# print("ue2_par")
# print(ue2_par)
# print("ua2_par")
# print(ua2_par)
# print("we2_par")
# print(we2_par)
# print("wa2_par")
# print(wa2_par)
    

###############################################################################
# EMISSIONS
###############################################################################

# MANUAL, source: LAEI 2016
#ez1_emis_no2 = 46.32
#ez2_emis_no2 = 37.66

# with all sources
#ez1_emis_pm25 = 17.59 
#ez2_emis_pm25 = 16.56
# just exhaust
#ez1_emis_pm25 = 6.17
#ez2_emis_pm25 = 5.17

# MANUAL, source: LAEI 2013
ez1_emis_no2 = 551.56*0.25
ez2_emis_no2 = 453.19*0.25

# with all sources
ez1_emis_pm25 = 20.189
ez2_emis_pm25 = 18.755

print("EMISSIONS")
print("NO2 emissions, EZ1:", ez1_emis_no2)
print("NO2 emissions, EZ2:", ez2_emis_no2)
print("PM2.5 emissions EZ1:", ez1_emis_pm25)
print("PM2.5 emissions EZ2:", ez2_emis_pm25)
print("")

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

# had to flip things around because function to partition emissions relies on
# EZ1 being first. Therefore in mirror version, EZ1 becomes EZ2 and vice versa
# is this hardwired into the platform? So if there's only 1 emissions zone it's EZ1,
# and if there are 2, EZ1 is upwind of EZ2?
ez2_finish_mir= round(roadw-ez1_start,4)
ez2_start_mir= round(roadw-ez1_finish,4)
ez1_finish_mir= round(roadw-ez2_start,4)
ez1_start_mir= round(roadw-ez2_finish,4)

#print(ez1_start_mir, ez1_finish_mir, ez2_start_mir, ez2_finish_mir)

emis_par_fun_mir = emission_partition(l_cumu=l_cumu_mirror, l=l_mirror, 
                                       ez1_start=ez1_start_mir, ez1_finish=ez1_finish_mir, ez2_start=ez2_start_mir, ez2_finish=ez2_finish_mir, 
                                       ez1_par=ez1_par_mir, ez2_par=ez2_par_mir, 
                                       ez1_w=ez2_w, ez2_w=ez1_w)

ez1_par_mir = emis_par_fun_mir[0]
ez2_par_mir = emis_par_fun_mir[1]

#print(ez1_par_mir)
#print(ez2_par_mir)

ez_tot_no2_orig = np.array([0,0,0,0,0,0], dtype = float)
ez_tot_no2_orig[1] = (ez1_par_orig[1]*float(ez1_emis_no2)) + (ez2_par_orig[1]*float(ez2_emis_no2))
ez_tot_no2_orig[2] = (ez1_par_orig[2]*float(ez1_emis_no2)) + (ez2_par_orig[2]*float(ez2_emis_no2))
ez_tot_no2_orig[3] = (ez1_par_orig[3]*float(ez1_emis_no2)) + (ez2_par_orig[3]*float(ez2_emis_no2))
ez_tot_no2_orig[4] = (ez1_par_orig[4]*float(ez1_emis_no2)) + (ez2_par_orig[4]*float(ez2_emis_no2))
ez_tot_no2_orig[5] = (ez1_par_orig[5]*float(ez1_emis_no2)) + (ez2_par_orig[5]*float(ez2_emis_no2))

ez_tot_pm25_orig = np.array([0,0,0,0,0,0], dtype = float)
ez_tot_pm25_orig[1] = (ez1_par_orig[1]*float(ez1_emis_pm25)) + (ez2_par_orig[1]*float(ez2_emis_pm25))
ez_tot_pm25_orig[2] = (ez1_par_orig[2]*float(ez1_emis_pm25)) + (ez2_par_orig[2]*float(ez2_emis_pm25))
ez_tot_pm25_orig[3] = (ez1_par_orig[3]*float(ez1_emis_pm25)) + (ez2_par_orig[3]*float(ez2_emis_pm25))
ez_tot_pm25_orig[4] = (ez1_par_orig[4]*float(ez1_emis_pm25)) + (ez2_par_orig[4]*float(ez2_emis_pm25))
ez_tot_pm25_orig[5] = (ez1_par_orig[5]*float(ez1_emis_pm25)) + (ez2_par_orig[5]*float(ez2_emis_pm25))

ez_tot_no2_mir = np.array([0,0,0,0,0,0], dtype = float)
ez_tot_no2_mir[1] = (ez1_par_mir[1]*float(ez2_emis_no2)) + (ez2_par_mir[1]*float(ez1_emis_no2))
ez_tot_no2_mir[2] = (ez1_par_mir[2]*float(ez2_emis_no2)) + (ez2_par_mir[2]*float(ez1_emis_no2))
ez_tot_no2_mir[3] = (ez1_par_mir[3]*float(ez2_emis_no2)) + (ez2_par_mir[3]*float(ez1_emis_no2))
ez_tot_no2_mir[4] = (ez1_par_mir[4]*float(ez2_emis_no2)) + (ez2_par_mir[4]*float(ez1_emis_no2))
ez_tot_no2_mir[5] = (ez1_par_mir[5]*float(ez2_emis_no2)) + (ez2_par_mir[5]*float(ez1_emis_no2))

ez_tot_pm25_mir = np.array([0,0,0,0,0,0], dtype = float)
ez_tot_pm25_mir[1] = (ez1_par_mir[1]*float(ez2_emis_pm25)) + (ez2_par_mir[1]*float(ez1_emis_pm25))
ez_tot_pm25_mir[2] = (ez1_par_mir[2]*float(ez2_emis_pm25)) + (ez2_par_mir[2]*float(ez1_emis_pm25))
ez_tot_pm25_mir[3] = (ez1_par_mir[3]*float(ez2_emis_pm25)) + (ez2_par_mir[3]*float(ez1_emis_pm25))
ez_tot_pm25_mir[4] = (ez1_par_mir[4]*float(ez2_emis_pm25)) + (ez2_par_mir[4]*float(ez1_emis_pm25))
ez_tot_pm25_mir[5] = (ez1_par_mir[5]*float(ez2_emis_pm25)) + (ez2_par_mir[5]*float(ez1_emis_pm25))

# check the sum of partitioned emissions equal the orginal
if round(sum(ez_tot_no2_orig),0) != round((float(ez1_emis_no2) + float(ez2_emis_no2))):
    error[8,1] = 1
    
if round(sum(ez_tot_no2_mir),0) != round((float(ez1_emis_no2) + float(ez2_emis_no2))):
    error[8,1] = 1
    
if round(sum(ez_tot_pm25_orig),0) != round((float(ez1_emis_pm25) + float(ez2_emis_pm25))):
    error[8,1] = 1
    
if round(sum(ez_tot_pm25_mir),0) != round((float(ez1_emis_pm25) + float(ez2_emis_pm25))):
    error[8,1] = 1

#print(error[8,1])
#print(ez_tot_no2_orig)
#print(ez_tot_pm25_orig)
#print(ez_tot_no2_mir)
#print(ez_tot_pm25_mir)


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

r2 = np.zeros((4,6))
r2[1,1] = h_mirror[1]/l_mirror[1] 
r2[1,2] = h_mirror[1]/l_mirror[2] 
r2[1,3] = h_mirror[1]/l_mirror[3]
r2[1,4] = h_mirror[1]/l_mirror[4] 
r2[1,5] = h_mirror[1]/l_mirror[5] 
r2[2,1] = h_mirror[2]/l_mirror[1] 
r2[2,2] = h_mirror[2]/l_mirror[2]
r2[2,3] = h_mirror[2]/l_mirror[3] 
r2[2,4] = h_mirror[2]/l_mirror[4]
r2[2,5] = h_mirror[2]/l_mirror[5]
r2[3,1] = h_mirror[3]/l_mirror[1]
r2[3,2] = h_mirror[3]/l_mirror[2] 
r2[3,3] = h_mirror[3]/l_mirror[3] 
r2[3,4] = h_mirror[3]/l_mirror[4]
r2[3,5] = h_mirror[3]/l_mirror[5] 


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

# wind right to left, before new barrier
a1_mir = a_matrix(r = r2, u = ue2_mir, w = we2_mir, U = ua2_mir, W = wa2_mir)

# important note: slightly different naming conventions used for parallel 
# conditions so don't compare directly with above
# wind parallel, before new barrier, dimensioning from wind left to right
a1_par_1 = a_matrix(r = r1, u = ue1_par, U = ua1_par, w = we1_par, W = wa1_par)

# wind parallel, before new barrier, dimensioning from wind right to left
a1_par_2 = a_matrix(r = r2, u = ue2_par, U = ua2_par, w = we2_par, W = wa2_par)


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

# d3_orig = inputs for existing conditions, PM2.5
d3_orig = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype = float)
d3_orig[0] = ez_tot_pm25_orig[1]/l_original[1] # averaging line emission source over area of 'ground' = (ug/m2/s)
d3_orig[1] = ez_tot_pm25_orig[2]/l_original[2]
d3_orig[2] = ez_tot_pm25_orig[3]/l_original[3]
d3_orig[3] = ez_tot_pm25_orig[4]/l_original[4]
d3_orig[4] = ez_tot_pm25_orig[5]/l_original[5]
d3_orig[5] = 0      # no inputs into middle boxes from either emissions or background
d3_orig[6] = 0
d3_orig[7] = 0
d3_orig[8] = 0
d3_orig[9] = 0
d3_orig[10] = (we2_orig[4,1] - beta(wa2_orig[4,1])*wa2_orig[4,1])*cB_pm25  # background inputs into top boxes
d3_orig[11] = (we2_orig[4,2] - beta(wa2_orig[4,2])*wa2_orig[4,2])*cB_pm25  # negative for beta because if true, the sign of W[4,x] would be negative
d3_orig[12] = (we2_orig[4,3] - beta(wa2_orig[4,3])*wa2_orig[4,3])*cB_pm25  # yet as here being treated as an input, it should be positive, hence signs cancel out
d3_orig[13] = (we2_orig[4,4] - beta(wa2_orig[4,4])*wa2_orig[4,4])*cB_pm25
d3_orig[14] = (we2_orig[4,5] - beta(wa2_orig[4,5])*wa2_orig[4,5])*cB_pm25

# d1_mir = inputs for existing conditions, NO2
d1_mir = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype = float)
d1_mir[0] = ez_tot_no2_mir[1]/l_mirror[1] # averaging line emission source over area of 'ground' = (ug/m2/s)
d1_mir[1] = ez_tot_no2_mir[2]/l_mirror[2]
d1_mir[2] = ez_tot_no2_mir[3]/l_mirror[3]
d1_mir[3] = ez_tot_no2_mir[4]/l_mirror[4]
d1_mir[4] = ez_tot_no2_mir[5]/l_mirror[5]
d1_mir[5] = 0      # no inputs into middle boxes from either emissions or background
d1_mir[6] = 0
d1_mir[7] = 0
d1_mir[8] = 0
d1_mir[9] = 0
d1_mir[10] = (we2_mir[4,1] - beta(wa2_mir[4,1])*wa2_mir[4,1])*cB_no2  # background inputs into top boxes
d1_mir[11] = (we2_mir[4,2] - beta(wa2_mir[4,2])*wa2_mir[4,2])*cB_no2  # negative for beta because if true, the sign of W[4,x] would be negative
d1_mir[12] = (we2_mir[4,3] - beta(wa2_mir[4,3])*wa2_mir[4,3])*cB_no2  # yet as here being treated as an input, it should be positive, hence signs cancel out
d1_mir[13] = (we2_mir[4,4] - beta(wa2_mir[4,4])*wa2_mir[4,4])*cB_no2
d1_mir[14] = (we2_mir[4,5] - beta(wa2_mir[4,5])*wa2_mir[4,5])*cB_no2

# d3_mir = inputs for existing conditions, PM2.5
d3_mir = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype = float)
d3_mir[0] = ez_tot_pm25_mir[1]/l_mirror[1] # averaging line emission source over area of 'ground' = (ug/m2/s)
d3_mir[1] = ez_tot_pm25_mir[2]/l_mirror[2]
d3_mir[2] = ez_tot_pm25_mir[3]/l_mirror[3]
d3_mir[3] = ez_tot_pm25_mir[4]/l_mirror[4]
d3_mir[4] = ez_tot_pm25_mir[5]/l_mirror[5]
d3_mir[5] = 0      # no inputs into middle boxes from either emissions or background
d3_mir[6] = 0
d3_mir[7] = 0
d3_mir[8] = 0
d3_mir[9] = 0
d3_mir[10] = (we2_mir[4,1] - beta(wa2_mir[4,1])*wa2_mir[4,1])*cB_pm25  # background inputs into top boxes
d3_mir[11] = (we2_mir[4,2] - beta(wa2_mir[4,2])*wa2_mir[4,2])*cB_pm25  # negative for beta because if true, the sign of W[4,x] would be negative
d3_mir[12] = (we2_mir[4,3] - beta(wa2_mir[4,3])*wa2_mir[4,3])*cB_pm25  # yet as here being treated as an input, it should be positive, hence signs cancel out
d3_mir[13] = (we2_mir[4,4] - beta(wa2_mir[4,4])*wa2_mir[4,4])*cB_pm25
d3_mir[14] = (we2_mir[4,5] - beta(wa2_mir[4,5])*wa2_mir[4,5])*cB_pm25


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

# d3_orig_par = inputs for existing conditions, PM2.5
d3_orig_par = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype = float)
d3_orig_par[0] = ez_tot_pm25_orig[1]/l_original[1] # averaging line emission source over area of 'ground' = (ug/m2/s)
d3_orig_par[1] = ez_tot_pm25_orig[2]/l_original[2]
d3_orig_par[2] = ez_tot_pm25_orig[3]/l_original[3]
d3_orig_par[3] = ez_tot_pm25_orig[4]/l_original[4]
d3_orig_par[4] = ez_tot_pm25_orig[5]/l_original[5]
d3_orig_par[5] = 0      # no inputs into middle boxes from either emissions or background
d3_orig_par[6] = 0
d3_orig_par[7] = 0
d3_orig_par[8] = 0
d3_orig_par[9] = 0
d3_orig_par[10] = (we1_par[4,1] - beta(wa1_par[4,1])*wa1_par[4,1])*cB_pm25  # background inputs into top boxes
d3_orig_par[11] = (we1_par[4,2] - beta(wa1_par[4,2])*wa1_par[4,2])*cB_pm25  # negative for beta because if true, the sign of W[4,x] would be negative
d3_orig_par[12] = (we1_par[4,3] - beta(wa1_par[4,3])*wa1_par[4,3])*cB_pm25  # yet as here being treated as an input, it should be positive, hence signs cancel out
d3_orig_par[13] = (we1_par[4,4] - beta(wa1_par[4,4])*wa1_par[4,4])*cB_pm25
d3_orig_par[14] = (we1_par[4,5] - beta(wa1_par[4,5])*wa1_par[4,5])*cB_pm25

# d1_mir_par = inputs for existing conditions, NO2
d1_mir_par = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype = float)
d1_mir_par[0] = ez_tot_no2_mir[1]/l_mirror[1] # averaging line emission source over area of 'ground' = (ug/m2/s)
d1_mir_par[1] = ez_tot_no2_mir[2]/l_mirror[2]
d1_mir_par[2] = ez_tot_no2_mir[3]/l_mirror[3]
d1_mir_par[3] = ez_tot_no2_mir[4]/l_mirror[4]
d1_mir_par[4] = ez_tot_no2_mir[5]/l_mirror[5]
d1_mir_par[5] = 0      # no inputs into middle boxes from either emissions or background
d1_mir_par[6] = 0
d1_mir_par[7] = 0
d1_mir_par[8] = 0
d1_mir_par[9] = 0
d1_mir_par[10] = (we2_par[4,1] - beta(wa2_par[4,1])*wa2_par[4,1])*cB_no2  # background inputs into top boxes
d1_mir_par[11] = (we2_par[4,2] - beta(wa2_par[4,2])*wa2_par[4,2])*cB_no2  # negative for beta because if true, the sign of W[4,x] would be negative
d1_mir_par[12] = (we2_par[4,3] - beta(wa2_par[4,3])*wa2_par[4,3])*cB_no2  # yet as here being treated as an input, it should be positive, hence signs cancel out
d1_mir_par[13] = (we2_par[4,4] - beta(wa2_par[4,4])*wa2_par[4,4])*cB_no2
d1_mir_par[14] = (we2_par[4,5] - beta(wa2_par[4,5])*wa2_par[4,5])*cB_no2

# d3_mir_par = inputs for existing conditions, PM2.5
d3_mir_par = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype = float)
d3_mir_par[0] = ez_tot_pm25_mir[1]/l_mirror[1] # averaging line emission source over area of 'ground' = (ug/m2/s)
d3_mir_par[1] = ez_tot_pm25_mir[2]/l_mirror[2]
d3_mir_par[2] = ez_tot_pm25_mir[3]/l_mirror[3]
d3_mir_par[3] = ez_tot_pm25_mir[4]/l_mirror[4]
d3_mir_par[4] = ez_tot_pm25_mir[5]/l_mirror[5]
d3_mir_par[5] = 0      # no inputs into middle boxes from either emissions or background
d3_mir_par[6] = 0
d3_mir_par[7] = 0
d3_mir_par[8] = 0
d3_mir_par[9] = 0
d3_mir_par[10] = (we2_par[4,1] - beta(wa2_par[4,1])*wa2_par[4,1])*cB_pm25  # background inputs into top boxes
d3_mir_par[11] = (we2_par[4,2] - beta(wa2_par[4,2])*wa2_par[4,2])*cB_pm25  # negative for beta because if true, the sign of W[4,x] would be negative
d3_mir_par[12] = (we2_par[4,3] - beta(wa2_par[4,3])*wa2_par[4,3])*cB_pm25  # yet as here being treated as an input, it should be positive, hence signs cancel out
d3_mir_par[13] = (we2_par[4,4] - beta(wa2_par[4,4])*wa2_par[4,4])*cB_pm25
d3_mir_par[14] = (we2_par[4,5] - beta(wa2_par[4,5])*wa2_par[4,5])*cB_pm25



###############################################################################
# CALCULATE & SEND DATA BACK
###############################################################################

# error check before sending data back
# if there are any error flags pass back the error message and an array of NaN
# otherwise, calculate the percentage change

empty_array = np.empty((3,5))
empty_array[:] = np.NaN
empty_array = empty_array.tolist()

#______________ define function to reverse mirror _____________________________
# create function that flips the concentrations back so they are referring to the 
# same street location
def flip_conc(C):
    C2 = C.copy()
    # row 1
    C2[0] = C[4]
    C2[1] = C[3]
    C2[2] = C[2]
    C2[3] = C[1]
    C2[4] = C[0]
    
    # row 2
    C2[5] = C[9]
    C2[6] = C[8]
    C2[7] = C[7]
    C2[8] = C[6]
    C2[9] = C[5]
    
    # row3
    C2[10] = C[14]
    C2[11] = C[13]
    C2[12] = C[12]
    C2[13] = C[11]
    C2[14] = C[10]
    
    return C2

#C = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
#C2 = flip_conc(C=C)
#print(C2)

def flip_column(l, l_cumu):
    # flip the mirrored columns back so they are refering to same street location
    l_cumu[1] = l[5]
    l_cumu[2] = l[5]+l[4]
    l_cumu[3] = l[5]+l[4]+l[3]
    l_cumu[4] = l[5]+l[4]+l[3]+l[2]
    l_cumu[5] = l[5]+l[4]+l[3]+l[2]+l[1]
    return l_cumu

#l_cumu_mirror = flip_column(l=l_mirror, l_cumu=l_cumu_mirror)
#print(l_cumu_mirror)

def weighting_concs(l_cumu_LR, l_cumu_RL, C_LR, C_LR_par, C_RL, C_RL_par, LR_freq, LR_par_freq, RL_freq, RL_par_freq):
    # column positions
    #l_cumu_LR = [0,5,7.5,15,21,25]
    #l_cumu_RL = [0,2.5,7,13,15,25]
    
    # concentration changes
    #C_LR = [5,2,-3,10,-2,]
    #C_LR_par = [2,1,0,-1,2,]
    #C_RL = [-3,-5,10,-2,-5]
    #C_RL_par = [-1,-3,5,-1,-10]
    
    # join the column positions together into 1 vector
    l_cumu_tot = np.concatenate((l_cumu_LR, l_cumu_RL))
    # order from smallest to largest
    l_cumu_tot = np.sort(l_cumu_tot)
    # take out any duplicates
    l_cumu_tot = np.unique(l_cumu_tot)
    #print(l_cumu_tot)
    
    # concatenate column positions for rows 2 and 3 so just in 1 linear vector
    #l_cumu_tot = np.concatenate((l_cumu_tot, l_cumu_tot, l_cumu_tot)) 
    #print(l_cumu_tot)
    
    # create empty containers for the weighting values, based on how many unique columns there are
    w_LR = l_cumu_tot.copy()
    w_LR[:] = 0
    w_LR = w_LR[0:(len(l_cumu_tot)-1)]
    w_LR_par = w_LR.copy()
    w_RL = w_LR.copy()
    w_RL_par = w_LR.copy()
    #print(w_LR, w_LR_par, w_RL, w_RL_par)
    
    weighted_row = w_LR.copy()
    
    
    # loop over each value in the combined column vector and assign concentration change value
    
    for i in range(len(l_cumu_tot)):
        if l_cumu_tot[i] <= l_cumu_LR[1]:
            w_LR[i-1] = C_LR[0]
            w_LR_par[i-1] = C_LR_par[0]
        elif l_cumu_tot[i] <= l_cumu_LR[2]:
            w_LR[i-1] = C_LR[1]
            w_LR_par[i-1] = C_LR_par[1]
        elif l_cumu_tot[i] <= l_cumu_LR[3]:
            w_LR[i-1] = C_LR[2]
            w_LR_par[i-1] = C_LR_par[2]
        elif l_cumu_tot[i] <= l_cumu_LR[4]:
            w_LR[i-1] = C_LR[3]
            w_LR_par[i-1] = C_LR_par[3]
        elif l_cumu_tot[i] <= l_cumu_LR[5]:
            w_LR[i-1] = C_LR[4]
            w_LR_par[i-1] = C_LR_par[4]
        
        if l_cumu_tot[i] <= l_cumu_RL[1]:
            w_RL[i-1] = C_RL[0]
            w_RL_par[i-1] = C_RL_par[0]
        elif l_cumu_tot[i] <= l_cumu_RL[2]:
            w_RL[i-1] = C_RL[1]
            w_RL_par[i-1] = C_RL_par[1]
        elif l_cumu_tot[i] <= l_cumu_RL[3]:
            w_RL[i-1] = C_RL[2]
            w_RL_par[i-1] = C_RL_par[2]
        elif l_cumu_tot[i] <= l_cumu_RL[4]:
            w_RL[i-1] = C_RL[3]
            w_RL_par[i-1] = C_RL_par[3]
        elif l_cumu_tot[i] <= l_cumu_RL[5]:
            w_RL[i-1] = C_RL[4]
            w_RL_par[i-1] = C_RL_par[4]
            
    #print(w_LR)
    #print(w_LR_par)
    #print(w_RL)
    #print(w_RL_par)
    
    #LR_freq = 0.7
    #LR_par_freq = 0.1
    #RL_freq = 0.1
    #RL_par_freq = 0.1
    
    for i in range(len(weighted_row)):
        weighted_row[i] = (w_LR[i]*LR_freq)+(w_RL[i]*RL_freq)+(w_LR_par[i]*LR_par_freq)+(w_RL_par[i]*RL_par_freq)
    return(weighted_row, l_cumu_tot)

#______________________________________________________________________________


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
    print("********************************************************")
    print("IMPORTANT NOTE: Concentrations are listed in order from bottom left box to top right box. Therefore the box of R1C2 where the measurement station lies is the 2nd box concentration listed")
    print("********************************************************")
    print("")
    # _________________________________________________________________________
    # WIND: LEFT TO RIGHT
    # NO2
    C1_orig = solve(a1_orig, d1_orig)
    #C2_orig = solve(a2_orig, d2_orig)
    
    # PM2.5
    C3_orig = solve(a1_orig, d3_orig)
    #C4_orig = solve(a2_orig, d4_orig)
    
    print("CONCENTRATIONS: L-->R WIND CONDITIONS")
    print("NO2:", C1_orig, sep = '\n')
    print("PM2.5:", C3_orig, sep = '\n')
    print("")
    
    # calculate the percentage changes in concentrations before and after
    # NO2
    #per_change_no2_orig = ((C2_orig - C1_orig)/C1_orig)*100
    #print("NO2 % Change (wind L->R)", per_change_no2_orig, sep = '\n')
    
    # PM2.5
    #per_change_pm25_orig = ((C4_orig - C3_orig)/C3_orig)*100
    #print("PM 2.5 % Change (wind L->R)", per_change_pm25_orig, sep = '\n')
    
    # _________________________________________________________________________
    # WIND: RIGHT TO LEFT
    # NO2
    C1_mir = solve(a1_mir, d1_mir)
    #C2_mir = solve(a2_mir, d2_mir)
    
    # PM2.5
    C3_mir = solve(a1_mir, d3_mir)
    #C4_mir = solve(a2_mir, d4_mir)
    
    
    # calculate the percentage changes in concentrations before and after
    # NO2
    #per_change_no2_mir = ((C2_mir - C1_mir)/C1_mir)*100
    
    # PM2.5
    #per_change_pm25_mir = ((C4_mir - C3_mir)/C3_mir)*100
    
    # flip back so each box corresponds to the same space in the street
    C1_mir_flipped = flip_conc(C=C1_mir)
    C3_mir_flipped = flip_conc(C=C3_mir)
    #per_change_no2_mir_flipped = flip_conc(C=per_change_no2_mir)
    #per_change_pm25_mir_flipped = flip_conc(C=per_change_pm25_mir)
    l_cumu_flipped = flip_column(l=l_mirror, l_cumu=l_cumu_mirror)
    
    print("CONCENTRATIONS: R-->L WIND CONDITIONS (flipped back so correspond to same boxes as above)")
    print("NO2:", C1_mir_flipped, sep = '\n')
    print("PM2.5:", C3_mir_flipped, sep = '\n')
    print("")
    
    #print("NO2 % Change (wind R->L)", per_change_no2_mir_flipped, sep = "\n")
    #print("PM 2.5 % Change (wind R->L)",per_change_pm25_mir_flipped, sep = '\n')
    
    # _________________________________________________________________________
    # WIND: PARALLEL
    # NO2
    # dimensioning from left to right
    C1_par_1 = solve(a1_par_1, d1_orig_par)
    #C2_par_1 = solve(a2_par_1, d2_orig_par)
    
    # dimensioning from right to left
    C1_par_2 = solve(a1_par_2, d1_mir_par)
    #C2_par_2 = solve(a2_par_2, d2_mir_par)
    
    # % change of NO2 based on dispersion only (parallel wind) with dimensioning from wind left-to-right
    #per_change_no2_par_1 = ((C2_par_1 - C1_par_1)/C1_par_1)*100
    # % change of NO2 based on dispersion only (parallel wind) with dimensioning from wind right-to-left
    #per_change_no2_par_2 = ((C2_par_2 - C1_par_2)/C1_par_2)*100
    
    # PM2.5
    # dimensioning from left to right
    C3_par_1 = solve(a1_par_1, d3_orig_par)
    #C4_par_1 = solve(a2_par_1, d4_orig_par)
    
    # dimensioning from right to left
    C3_par_2 = solve(a1_par_2, d3_mir_par)
    #C4_par_2 = solve(a2_par_2, d4_mir_par)
    
    # % change of PM2.5 based on dispersion only (parallel wind) with dimensioning from wind left-to-right
    #per_change_pm25_par_1 = ((C4_par_1 - C3_par_1)/C3_par_1)*100
    # % change of PM2.5 based on dispersion only (parallel wind) with dimensioning from wind right-to-left
    #per_change_pm25_par_2 = ((C4_par_2 - C3_par_2)/C3_par_2)*100
    
    # flip back those calculated with dimensioning from right-to-left
    C1_par_2_flipped = flip_conc(C=C1_par_2)
    C3_par_2_flipped = flip_conc(C=C3_par_2)
    #per_change_no2_par_flipped = flip_conc(C=per_change_no2_par_2)
    #per_change_pm25_par_flipped = flip_conc(C=per_change_pm25_par_2)
    
    print("CONCENTRATIONS: PARALLEL WIND CONDITIONS")
    print("NO2")
    print("Dimensioning L-->R:", C1_par_1, sep = '\n')
    print("Dimensioning R-->L (flipped back):", C1_par_2_flipped, sep = '\n')
    print("")
    print("PM2.5")
    print("Dimensioning L-->R:", C3_par_1, sep = '\n')
    print("Dimensioning R-->L (flipped back):", C3_par_2_flipped, sep = '\n')
    print("")

    
    #print("NO2 % Change (Parallel wind, dimensioning: L->R)", per_change_no2_par_1, sep = '\n')
    #print("NO2 % Change (Parallel wind, dimensioning: R->L - flipped back)",per_change_no2_par_flipped, sep = '\n')
    #print("PM2.5 % Change (Parallel wind, dimensioning: L->R)",per_change_pm25_par_1, sep = '\n')
    #print("PM2.5 % Change (Parallel wind, dimensioning: R->L - flipped back)",per_change_pm25_par_flipped, sep = '\n')
    
    #print("NO2", "L-->R:", C1_orig, "R-->L:", C1_mir_flipped, "Parallel (LR dimensioning):", C1_par_1, "Parallel (RL dimensioning):", C1_par_2_flipped, sep = "\n")
    #print("PM2.5", "L-->R:", C3_orig, "R-->L:", C3_mir_flipped, "Parallel (LR dimensioning):", C3_par_1, "Parallel (RL dimensioning):", C3_par_2_flipped, sep = "\n")
    #__________________________________________________________________________
    # WEIGHT RESULTS BASED ON FREQUENCY OF CLIMATOLOGY
    # NO2 values
    # weighted_row1_no2 = weighting_concs(l_cumu_LR=l_cumu_original, l_cumu_RL=l_cumu_flipped, 
    #                             C_LR=per_change_no2_orig[0:5], C_LR_par=per_change_no2_par_1[0:5], 
    #                             C_RL=per_change_no2_mir_flipped[0:5], C_RL_par=per_change_no2_par_flipped[0:5], 
    #                             LR_freq=LR_freq, LR_par_freq=LR_par_freq, RL_freq=RL_freq, RL_par_freq=RL_par_freq)
    
    # l_cumu_total = weighted_row1_no2[1]
    # weighted_row1_no2 = weighted_row1_no2[0]
    # #print(weighted_row1_no2)
    
    weighted_row1_no2 = weighting_concs(l_cumu_LR=l_cumu_original, l_cumu_RL=l_cumu_flipped, 
                                C_LR=C1_orig[0:5], C_LR_par=C1_par_1[0:5], 
                                C_RL=C1_mir_flipped[0:5], C_RL_par=C1_par_2_flipped[0:5], 
                                LR_freq=LR_freq, LR_par_freq=LR_par_freq, RL_freq=RL_freq, RL_par_freq=RL_par_freq)
    
    #l_cumu_total = weighted_row1_no2[1]
    weighted_row1_no2 = weighted_row1_no2[0]
    #print(weighted_row1_no2)
    
    ############################
    
    # weighted_row2_no2 = weighting_concs(l_cumu_LR=l_cumu_original, l_cumu_RL=l_cumu_flipped, 
    #                             C_LR=per_change_no2_orig[5:10], C_LR_par=per_change_no2_par_1[5:10], 
    #                             C_RL=per_change_no2_mir_flipped[5:10], C_RL_par=per_change_no2_par_flipped[5:10], 
    #                             LR_freq=LR_freq, LR_par_freq=LR_par_freq, RL_freq=RL_freq, RL_par_freq=RL_par_freq)
    # weighted_row2_no2 = weighted_row2_no2[0]
    # #print(weighted_row2_no2)
    
    weighted_row2_no2 = weighting_concs(l_cumu_LR=l_cumu_original, l_cumu_RL=l_cumu_flipped, 
                                C_LR=C1_orig[5:10], C_LR_par=C1_par_1[5:10], 
                                C_RL=C1_mir_flipped[5:10], C_RL_par=C1_par_2_flipped[5:10], 
                                LR_freq=LR_freq, LR_par_freq=LR_par_freq, RL_freq=RL_freq, RL_par_freq=RL_par_freq)
    
    #l_cumu_total = weighted_row1_no2[1]
    weighted_row2_no2 = weighted_row2_no2[0]
    #print(weighted_row1_no2)
    
    ############################
    
    # weighted_row3_no2 = weighting_concs(l_cumu_LR=l_cumu_original, l_cumu_RL=l_cumu_flipped, 
    #                             C_LR=per_change_no2_orig[10:15], C_LR_par=per_change_no2_par_1[10:15], 
    #                             C_RL=per_change_no2_mir_flipped[10:15], C_RL_par=per_change_no2_par_flipped[10:15], 
    #                             LR_freq=LR_freq, LR_par_freq=LR_par_freq, RL_freq=RL_freq, RL_par_freq=RL_par_freq)
    # weighted_row3_no2 = weighted_row3_no2[0]
    # #print(weighted_row3_no2)
    
    weighted_row3_no2 = weighting_concs(l_cumu_LR=l_cumu_original, l_cumu_RL=l_cumu_flipped, 
                                C_LR=C1_orig[10:15], C_LR_par=C1_par_1[10:15], 
                                C_RL=C1_mir_flipped[10:15], C_RL_par=C1_par_2_flipped[10:15], 
                                LR_freq=LR_freq, LR_par_freq=LR_par_freq, RL_freq=RL_freq, RL_par_freq=RL_par_freq)
    
    #l_cumu_total = weighted_row1_no2[1]
    weighted_row3_no2 = weighted_row3_no2[0]
    #print(weighted_row1_no2)
    
    # per_change_no2 = np.concatenate((weighted_row1_no2, weighted_row2_no2, weighted_row3_no2))
    # #print("Weighted NO2 % Change", per_change_no2, sep = '\n')
    
    weighted_NO2 = np.concatenate((weighted_row1_no2, weighted_row2_no2, weighted_row3_no2))
    print("WEIGHTED RESULTS")
    print("NO2", weighted_NO2, sep = '\n')
    
    ##################################
    
    # # pm25 values
    # weighted_row1_pm25 = weighting_concs(l_cumu_LR=l_cumu_original, l_cumu_RL=l_cumu_flipped, 
    #                             C_LR=per_change_pm25_orig[0:5], C_LR_par=per_change_pm25_par_1[0:5], 
    #                             C_RL=per_change_pm25_mir_flipped[0:5], C_RL_par=per_change_pm25_par_flipped[0:5], 
    #                             LR_freq=LR_freq, LR_par_freq=LR_par_freq, RL_freq=RL_freq, RL_par_freq=RL_par_freq)
    # weighted_row1_pm25 = weighted_row1_pm25[0]
    # #print(weighted_row1_pm25)
    
    weighted_row1_pm25 = weighting_concs(l_cumu_LR=l_cumu_original, l_cumu_RL=l_cumu_flipped, 
                                C_LR=C3_orig[0:5], C_LR_par=C3_par_1[0:5], 
                                C_RL=C3_mir_flipped[0:5], C_RL_par=C3_par_2_flipped[0:5], 
                                LR_freq=LR_freq, LR_par_freq=LR_par_freq, RL_freq=RL_freq, RL_par_freq=RL_par_freq)
    
    #l_cumu_total = weighted_row1_no2[1]
    weighted_row1_pm25 = weighted_row1_pm25[0]
    #print(weighted_row1_no2)
    
    #####################################
    
    # weighted_row2_pm25 = weighting_concs(l_cumu_LR=l_cumu_original, l_cumu_RL=l_cumu_flipped, 
    #                             C_LR=per_change_pm25_orig[5:10], C_LR_par=per_change_pm25_par_1[5:10], 
    #                             C_RL=per_change_pm25_mir_flipped[5:10], C_RL_par=per_change_pm25_par_flipped[5:10], 
    #                             LR_freq=LR_freq, LR_par_freq=LR_par_freq, RL_freq=RL_freq, RL_par_freq=RL_par_freq)
    # weighted_row2_pm25 = weighted_row2_pm25[0]
    # #print(weighted_row2_pm25)
    
    weighted_row2_pm25 = weighting_concs(l_cumu_LR=l_cumu_original, l_cumu_RL=l_cumu_flipped, 
                                C_LR=C3_orig[5:10], C_LR_par=C3_par_1[5:10], 
                                C_RL=C3_mir_flipped[5:10], C_RL_par=C3_par_2_flipped[5:10], 
                                LR_freq=LR_freq, LR_par_freq=LR_par_freq, RL_freq=RL_freq, RL_par_freq=RL_par_freq)
    
    #l_cumu_total = weighted_row1_no2[1]
    weighted_row2_pm25 = weighted_row2_pm25[0]
    #print(weighted_row1_no2)
    
    ######################################
    
    # weighted_row3_pm25 = weighting_concs(l_cumu_LR=l_cumu_original, l_cumu_RL=l_cumu_flipped, 
    #                             C_LR=per_change_pm25_orig[10:15], C_LR_par=per_change_pm25_par_1[10:15], 
    #                             C_RL=per_change_pm25_mir_flipped[10:15], C_RL_par=per_change_pm25_par_flipped[10:15], 
    #                             LR_freq=LR_freq, LR_par_freq=LR_par_freq, RL_freq=RL_freq, RL_par_freq=RL_par_freq)
    # weighted_row3_pm25 = weighted_row3_pm25[0]
    # #print(weighted_row3_pm25)
    
    
    weighted_row3_pm25 = weighting_concs(l_cumu_LR=l_cumu_original, l_cumu_RL=l_cumu_flipped, 
                                C_LR=C3_orig[10:15], C_LR_par=C3_par_1[10:15], 
                                C_RL=C3_mir_flipped[10:15], C_RL_par=C3_par_2_flipped[10:15], 
                                LR_freq=LR_freq, LR_par_freq=LR_par_freq, RL_freq=RL_freq, RL_par_freq=RL_par_freq)
    
    #l_cumu_total = weighted_row1_no2[1]
    weighted_row3_pm25 = weighted_row3_pm25[0]
    #print(weighted_row1_no2)
    
    # per_change_pm25 = np.concatenate((weighted_row1_pm25, weighted_row2_pm25, weighted_row3_pm25))
    # #print("Weighted PM2.5 % Change", per_change_pm25, sep = '\n')
    
    weighted_PM25 = np.concatenate((weighted_row1_pm25, weighted_row2_pm25, weighted_row3_pm25))
    print("PM2.5", weighted_PM25, sep = '\n')
    
    # # format to correct layout for feeding back to UI
    # street = {}
    # street["columns"] = l_cumu_total.tolist()
    # street["rows"] = h_cumu_original_list
    # street["per_change_no2"] = per_change_no2.tolist()
    # street["per_change_pm25"] = per_change_pm25.tolist()
    
    # # send data back
    # json_data = json.dumps(street)
    # print(json_data, sep = '\n')
    


###############################################################################
# USEFUL PARAMETERS TO PRINT TO THE DISPLAY TO CHECK THAT THE CODE RUNS CORRECTLY
    # - UNCOMMENT AS NECESSARY
    
#print(gi)
#print("Row:", row)
#print("Zone:", zone)
#print("Bar:", bar)
#print("Check", check)
#print("Recirc:", rec)
#print("Columns:",l_cumu)
#print("Rows:", h_cumu)
    
#print("Rows covered by recirculation of upwind building:",rec_nrow) 
#print("Columns fully covered by recirculation of upwind building:",rec_ncol)

#print("U1:", U1)
#print("U2:", U2)
#print("U3:", U3)
#print("Uh:", Uh)
#print("Ur:", Ur)
#print("Ut:", Ut)

#print("Horizontal advection NO BARRIERS:", np.around(ua1,3), sep = '\n')
#print("Vertical advection NO BARRIERS:", np.around(wa1,3), sep = '\n')
#print("Horizontal dispersion NO BARRIERS:", np.around(ue1,3), sep = '\n')
#print("Vertical dispersion NO BARRIERS:", np.around(we1,3), sep = '\n')

#print("Horizontal advection after existing barriers:", np.around(ua2,3), sep = '\n')
#print("Vertical advection after existing barriers:", np.around(wa2,3), sep = '\n')
#print("Horizontal dispersion after existing barriers:", np.around(ue2,3), sep = '\n')
#print("Vertical dispersion after existing barriers:", np.around(we2,3), sep = '\n')

#print("Horizontal advection after GI:", np.around(ua3,3),sep = '\n')
#print("Vertical advection after GI:", np.around(wa3,3),sep = '\n')
#print("Horizontal dispersion after GI:", np.around(ue3,3),sep = '\n')
#print("Vertical dispersion after GI:", np.around(we3,3),sep = '\n')
