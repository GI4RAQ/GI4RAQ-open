#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:04:01 2019

@author: HelenPearce
"""
###############################################################################
# PHASE 3: AUGUST 2020: WIND CLIMATOLOGY WEIGHTING AND PM2.5

###############################################################################
# HOUSE-KEEPING
###############################################################################

# import necessary packages
import numpy as np
import base64
import json
from scipy.linalg import solve
import math as math
from statistics import mean

# note: there will be a warning symbol for 'import sys' if you are using a static
# string below to obtain user info, but sys is needed for dynamic base64 
# loading. Warning can be ignored, code will still run without issue
import sys

# for met data
from shapely.geometry import Point, MultiPoint
from shapely.ops import nearest_points
import pandas as pd

# set container for error flags - this will be checked before final calculations
error = np.zeros((10,10))

###############################################################################
# DATA FROM USER INTERFACE CONTAINING INFORMATION ON STREET LAYOUT
###############################################################################

# ___________ dynamic info from user interface ___________
#if using a dynamic base64 string (fed from the user interface) use line below 
content = json.loads(base64.b64decode(sys.argv[1]).decode('utf8'))

# ___________ static info for testing purposes ___________
# if using your own generate base64 string, assign as a string to base64_test
# and then uncomment line 55 (decode the string and store as content)

# mirror test completed on this layout:
#base64_test = "eyJpZCI6ODMsInByb2plY3RzX2lkIjozOCwiYXV0aG9yX2lkIjozMCwibmFtZSI6IkV4YW1wbGUgQW5hbHlzaXMiLCJ0YWciOiJleGFtcGxlLWFuYWx5c2lzIiwid2luZCI6ImxlZnQiLCJvYmplY3RzIjoiW3tcInR5cGVcIjpcImJ1aWxkaW5nXCIsXCJuYW1lXCI6XCJCTDFcIixcIm9yZGVyXCI6MCxcImxvY2tlZFwiOnRydWUsXCJoZWlnaHRcIjpcIjI0LjNcIixcIndpZHRoXCI6XCIyXCIsXCJ3aGVyZVwiOlwic3RyZWV0dG9lbmRcIn0se1widHlwZVwiOlwicmN6XCIsXCJuYW1lXCI6XCJSWkJMMVwiLFwib3JkZXJcIjoxLFwibG9ja2VkXCI6dHJ1ZSxcIndpZHRoXCI6XCIxMC4xXCIsXCJnYlwiOlt7XCJuYW1lXCI6XCJFQjFcIixcInR5cGVcIjpcImdyZXktYmFycmllclwiLFwibWV0ZXJzaW56b25lXCI6XCIxMFwiLFwiaGVpZ2h0XCI6XCIxLjZcIixcIndpZHRoXCI6XCIwLjJcIixcImxhZFwiOlwiMjBcIixcInNwZWNpZXNcIjpudWxsLFwiYWFwXCI6bnVsbCxcImFhbVwiOm51bGwsXCJleHBlY3RlZGxpZmV0aW1lXCI6bnVsbCxcInRhYXBcIjpudWxsLFwidGN0aFwiOm51bGwsXCJ0Y2JoXCI6bnVsbCxcInRhYW1cIjpudWxsLFwidGZvcm1cIjpudWxsLFwidHNwZWNpZXNcIjpudWxsLFwidGVsXCI6bnVsbCxcInRzcFwiOm51bGwsXCJ0bGFkXCI6XCIyMFwiLFwidHN3XCI6bnVsbCxcIm9yZGVyXCI6MH1dLFwid2hlcmVcIjpcInN0cmVldHRvZW5kXCIsXCJnaVwiOltdfSx7XCJ0eXBlXCI6XCJtYXJrZXJcIixcIm5hbWVcIjpcIlN0cmVldCBCb3VuZGFyeSAxXCIsXCJvcmRlclwiOjIsXCJsb2NrZWRcIjp0cnVlLFwic3VidHlwZVwiOlwiZWRnZVwifSx7XCJvcmRlclwiOjAsXCJ0eXBlXCI6XCJyY3pcIixcIm5hbWVcIjpcIlJaMVwiLFwiaGVpZ2h0XCI6MCxcIndpZHRoXCI6XCI0LjNcIixcImN1cnJlbnRfZXhwb3N1cmVfbm8yXCI6XCIyXCIsXCJjdXJyZW50X2V4cG9zdXJlX1BNMl81XCI6XCIzXCIsXCJ2bW92ZW1lbnRcIjpudWxsLFwiZ2lcIjpbXSxcIndoZXJlXCI6XCJzdHJlZXRva2VyYlwifSx7XCJ0eXBlXCI6XCJuZXV0cmFsXCIsXCJuYW1lXCI6XCJOWjFcIixcImhlaWdodFwiOjAsXCJ3aWR0aFwiOlwiMlwiLFwiY3VycmVudF9leHBvc3VyZV9ubzJcIjpudWxsLFwiY3VycmVudF9leHBvc3VyZV9QTTJfNVwiOm51bGwsXCJ3aGVyZVwiOlwic3RyZWV0b2tlcmJcIixcImdpXCI6W119LHtcInR5cGVcIjpcIm1hcmtlclwiLFwibmFtZVwiOlwiS2VyYiAxXCIsXCJvcmRlclwiOjIsXCJsb2NrZWRcIjpmYWxzZSxcInN1YnR5cGVcIjpcImtlcmJcIn0se1widHlwZVwiOlwiZW16XCIsXCJuYW1lXCI6XCJFWjFcIixcIm9yZGVyXCI6MyxcImxvY2tlZFwiOmZhbHNlLFwid2lkdGhcIjpcIjEyLjRcIixcImN1cnJlbnRfZXhwb3N1cmVfbm8yXCI6XCIzXCIsXCJjdXJyZW50X2V4cG9zdXJlX1BNMl81XCI6XCIzXCIsXCJ3aGVyZVwiOlwia2VyYnRva2VyYlwifSx7XCJvcmRlclwiOjAsXCJ0eXBlXCI6XCJuZXV0cmFsXCIsXCJuYW1lXCI6XCJOWjJcIixcImhlaWdodFwiOjAsXCJ3aWR0aFwiOlwiMi42XCIsXCJjdXJyZW50X2V4cG9zdXJlX25vMlwiOm51bGwsXCJjdXJyZW50X2V4cG9zdXJlX1BNMl81XCI6bnVsbCxcInZtb3ZlbWVudFwiOm51bGwsXCJ3aGVyZVwiOlwia2VyYnRva2VyYlwiLFwiZ2lcIjpbXX0se1widHlwZVwiOlwiZW16XCIsXCJuYW1lXCI6XCJFWjJcIixcImhlaWdodFwiOjAsXCJ3aWR0aFwiOlwiMTAuNFwiLFwiY3VycmVudF9leHBvc3VyZV9ubzJcIjpcIjJcIixcImN1cnJlbnRfZXhwb3N1cmVfUE0yXzVcIjpcIjJcIixcInZtb3ZlbWVudFwiOlwiMTBcIixcIndoZXJlXCI6XCJrZXJidG9rZXJiXCJ9LHtcInR5cGVcIjpcIm5ldXRyYWxcIixcIm5hbWVcIjpcIk5aM1wiLFwiaGVpZ2h0XCI6MCxcIndpZHRoXCI6XCIyXCIsXCJjdXJyZW50X2V4cG9zdXJlX25vMlwiOm51bGwsXCJjdXJyZW50X2V4cG9zdXJlX1BNMl81XCI6bnVsbCxcIndoZXJlXCI6XCJrZXJidG9rZXJiXCIsXCJnaVwiOltdfSx7XCJ0eXBlXCI6XCJtYXJrZXJcIixcIm5hbWVcIjpcIktlcmIgMlwiLFwib3JkZXJcIjoyLFwibG9ja2VkXCI6ZmFsc2UsXCJzdWJ0eXBlXCI6XCJrZXJiXCJ9LHtcInR5cGVcIjpcInJjelwiLFwibmFtZVwiOlwiUloyXCIsXCJoZWlnaHRcIjowLFwid2lkdGhcIjpcIjEwLjhcIixcImN1cnJlbnRfZXhwb3N1cmVfbm8yXCI6XCIyXCIsXCJjdXJyZW50X2V4cG9zdXJlX1BNMl81XCI6XCIzXCIsXCJnaVwiOlt7XCJuYW1lXCI6XCJUcmVlbkxpbmVcIixcInR5cGVcIjpcImdyZWVuLWJhcnJpZXItdHJlZXNcIixcIm1ldGVyc2luem9uZVwiOlwiMi40XCIsXCJoZWlnaHRcIjpcIjIuNlwiLFwid2lkdGhcIjpcIjAuOVwiLFwibGFkXCI6XCIyMFwiLFwic3BlY2llc1wiOlwiZXZlcmdyZWVuXCIsXCJhYXBcIjpcIjFcIixcImFhbVwiOlwiMlwiLFwiZXhwZWN0ZWRsaWZldGltZVwiOlwiM1wiLFwidGFhcFwiOlwiMVwiLFwidGN0aFwiOlwiMTBcIixcInRjYmhcIjpcIjVcIixcInRhYW1cIjpcIjVcIixcInRmb3JtXCI6XCJ1bmtub3duXCIsXCJ0c3BlY2llc1wiOlwiZGVjaWR1b3VzXCIsXCJ0ZWxcIjpcIjJcIixcInRzcFwiOlwiOFwiLFwidGxhZFwiOlwiMjBcIixcInRzd1wiOm51bGwsXCJvcmRlclwiOjAsXCJ0Y3dcIjpcIjZcIn1dLFwiZ2JcIjpbXSxcIndoZXJlXCI6XCJzdHJlZXRva2VyYlwifSx7XCJ0eXBlXCI6XCJtYXJrZXJcIixcIm5hbWVcIjpcIlN0cmVldCBCb3VuZGFyeSAyXCIsXCJvcmRlclwiOjQsXCJsb2NrZWRcIjp0cnVlLFwic3VidHlwZVwiOlwiZWRnZVwifSx7XCJ0eXBlXCI6XCJyY3pcIixcIm5hbWVcIjpcIlJaQkwyXCIsXCJvcmRlclwiOjUsXCJsb2NrZWRcIjp0cnVlLFwid2lkdGhcIjpcIjUuMVwiLFwiZ2JcIjpbe1wibmFtZVwiOlwiRUIyXCIsXCJ0eXBlXCI6XCJncmV5LWJhcnJpZXJcIixcIm1ldGVyc2luem9uZVwiOjAsXCJoZWlnaHRcIjpcIjFcIixcIndpZHRoXCI6XCIwLjRcIixcImxhZFwiOlwiMzBcIixcInNwZWNpZXNcIjpudWxsLFwiYWFwXCI6bnVsbCxcImFhbVwiOm51bGwsXCJleHBlY3RlZGxpZmV0aW1lXCI6bnVsbCxcInRhYXBcIjpudWxsLFwidGN0aFwiOm51bGwsXCJ0Y2JoXCI6bnVsbCxcInRhYW1cIjpudWxsLFwidGZvcm1cIjpudWxsLFwidHNwZWNpZXNcIjpudWxsLFwidGVsXCI6bnVsbCxcInRzcFwiOm51bGwsXCJ0bGFkXCI6XCI0MFwiLFwidHN3XCI6bnVsbCxcIm9yZGVyXCI6MH1dLFwid2hlcmVcIjpcInN0cmVldHRvZW5kXCIsXCJnaVwiOltdfSx7XCJ0eXBlXCI6XCJidWlsZGluZ1wiLFwibmFtZVwiOlwiQkwyXCIsXCJvcmRlclwiOjYsXCJsb2NrZWRcIjp0cnVlLFwiaGVpZ2h0XCI6XCIyMS4yXCIsXCJ3aWR0aFwiOlwiMlwiLFwid2hlcmVcIjpcInN0cmVldHRvZW5kXCJ9XSIsImxvY2tlZCI6MSwiUE0yNSI6NDUsIk5PMiI6NDUsImNyZWF0ZWRfYXQiOiIyMDE5LTEyLTAzIDEzOjU4OjQ5IiwidXBkYXRlZF9hdCI6IjIwMTktMTItMDMgMTM6NTg6NDkiLCJkZWxldGVkX2F0IjpudWxsfQ=="

# barrier position 0, vmovement specified
#base64_test = "eyJpZCI6MjgwLCJwcm9qZWN0c19pZCI6ODMsImF1dGhvcl9pZCI6MjMsImxhdCI6IjU4IiwibG5nIjoiMCIsIm5hbWUiOiJUZXN0IDIiLCJ0YWciOiJ0ZXN0LTItMSIsIndpbmQiOiJFU0UiLCJvYmplY3RzIjoiW3tcInR5cGVcIjpcImJ1aWxkaW5nXCIsXCJuYW1lXCI6XCJCTDFcIixcIm9yZGVyXCI6MCxcImxvY2tlZFwiOnRydWUsXCJoZWlnaHRcIjpcIjhcIixcIndpZHRoXCI6MixcInNpZGVcIjpcImxlZnRcIixcIndoZXJlXCI6XCJzdHJlZXR0b2VuZFwifSx7XCJ0eXBlXCI6XCJyZWNlcHRvcl96b25lXCIsXCJuYW1lXCI6XCJSWkJMMVwiLFwib3JkZXJcIjoxLFwibG9ja2VkXCI6dHJ1ZSxcImF0dGFjaGVkVG9CdWlsZGluZ1wiOnRydWUsXCJ3aWR0aFwiOlwiM1wiLFwic2lkZVwiOlwibGVmdFwiLFwid2hlcmVcIjpcInN0cmVldHRvZW5kXCIsXCJleGlzdGluZ19iYXJyaWVyXCI6W3tcIm5hbWVcIjpcIldhbGxcIixcInR5cGVcIjpcImdyZXktYmFycmllclwiLFwid2hlcmVfaW5fem9uZVwiOlwiM1wiLFwiaGVpZ2h0XCI6XCIyXCIsXCJ3aWR0aFwiOlwiMC41XCIsXCJvYnN0XCI6XCIxMDBcIixcInNlYXNcIjpudWxsLFwic2Vhc29uYWxpdHlcIjpudWxsLFwiYWFwXCI6bnVsbCxcImFhbVwiOm51bGwsXCJlbGlmZVwiOm51bGwsXCJ0YWFwXCI6bnVsbCxcInRjdGhcIjpudWxsLFwidGNiaFwiOm51bGwsXCJ0Y3dcIjpudWxsLFwidGFhbVwiOm51bGwsXCJ0Zm9ybVwiOm51bGwsXCJ0c2Vhc1wiOm51bGwsXCJ0c3BlY2llc1wiOm51bGwsXCJ0ZWxpZmVcIjpudWxsLFwidHNwXCI6bnVsbCxcInRvYnN0XCI6bnVsbCxcIm9yZGVyXCI6MH1dfSx7XCJ0eXBlXCI6XCJtYXJrZXJcIixcIm5hbWVcIjpcIlN0cmVldCBCb3VuZGFyeSAxXCIsXCJzdWJ0eXBlXCI6XCJlZGdlXCIsXCJvcmRlclwiOjIsXCJsb2NrZWRcIjp0cnVlLFwic2lkZVwiOlwibGVmdFwifSx7XCJvcmRlclwiOjAsXCJ0eXBlXCI6XCJyZWNlcHRvcl96b25lXCIsXCJuYW1lXCI6XCJSWjFcIixcImhlaWdodFwiOjAsXCJ3aWR0aFwiOlwiMlwiLFwiTk8yXCI6bnVsbCxcImhhc0VtaXNzaW9uc1wiOnRydWUsXCJ0cmFmZmljXCI6bnVsbCxcIlBNMjVcIjpudWxsLFwiZ2k0cmFxX2JhcnJpZXJcIjpbXSxcInNpZGVcIjpcImxlZnRcIixcIndoZXJlXCI6XCJzdHJlZXR0b2tlcmJcIn0se1widHlwZVwiOlwibWFya2VyXCIsXCJuYW1lXCI6XCJLZXJiIDFcIixcInN1YnR5cGVcIjpcImtlcmJcIixcIm9yZGVyXCI6MixcImxvY2tlZFwiOmZhbHNlLFwic2lkZVwiOlwibGVmdFwifSx7XCJ0eXBlXCI6XCJlbWlzc2lvbnNfem9uZVwiLFwibmFtZVwiOlwiRVoxXCIsXCJvcmRlclwiOjMsXCJsb2NrZWRcIjpmYWxzZSxcIndpZHRoXCI6MTAsXCJjdXJyZW50X2V4cG9zdXJlX25vMlwiOm51bGwsXCJjdXJyZW50X2V4cG9zdXJlX1BNMl81XCI6bnVsbCxcInRyYWZmaWNcIjpcIjEyMDBcIixcInNpZGVcIjpcInJpZ2h0XCIsXCJ3aGVyZVwiOlwia2VyYnRva2VyYlwiLFwiaGFzRW1pc3Npb25zXCI6ZmFsc2UsXCJOTzJcIjpudWxsLFwiUE0yNVwiOm51bGwsXCJ2bW92ZW1lbnRcIjpcIjEyMDBcIixcImN1cnJlbnRfZW1pc3Npb25zX1BNMl81XCI6bnVsbCxcImN1cnJlbnRfZW1pc3Npb25zX25vMlwiOm51bGx9LHtcInR5cGVcIjpcIm1hcmtlclwiLFwibmFtZVwiOlwiS2VyYiAyXCIsXCJzdWJ0eXBlXCI6XCJrZXJiXCIsXCJvcmRlclwiOjIsXCJsb2NrZWRcIjpmYWxzZSxcInNpZGVcIjpcInJpZ2h0XCJ9LHtcInR5cGVcIjpcInJlY2VwdG9yX3pvbmVcIixcIm5hbWVcIjpcIlJaMlwiLFwib3JkZXJcIjoxLFwibG9ja2VkXCI6ZmFsc2UsXCJ3aWR0aFwiOjIsXCJzaWRlXCI6XCJyaWdodFwiLFwid2hlcmVcIjpcInN0cmVldHRva2VyYlwiLFwiZ2k0cmFxX2JhcnJpZXJcIjpbe1wibmFtZVwiOlwiR0k0UkFRXCIsXCJ0eXBlXCI6XCJncmVlbi1iYXJyaWVyXCIsXCJ3aGVyZV9pbl96b25lXCI6XCIwXCIsXCJoZWlnaHRcIjpcIjJcIixcIndpZHRoXCI6XCIxXCIsXCJvYnN0XCI6XCI3NVwiLFwic2Vhc1wiOm51bGwsXCJzZWFzb25hbGl0eVwiOlwiZXZlcmdlZW5cIixcImFhcFwiOm51bGwsXCJhYW1cIjpudWxsLFwiZWxpZmVcIjpudWxsLFwidGFhcFwiOm51bGwsXCJ0Y3RoXCI6XCI1LjVcIixcInRjYmhcIjpcIjIuNVwiLFwidGN3XCI6XCI0XCIsXCJ0YWFtXCI6bnVsbCxcInRmb3JtXCI6bnVsbCxcInRzZWFzXCI6XCJldmVyZ2VlblwiLFwidHNwZWNpZXNcIjpudWxsLFwidGVsaWZlXCI6bnVsbCxcInRzcFwiOlwiNVwiLFwidG9ic3RcIjpcIjc1XCIsXCJvcmRlclwiOjB9XX0se1widHlwZVwiOlwibWFya2VyXCIsXCJuYW1lXCI6XCJTdHJlZXQgQm91bmRhcnkgMlwiLFwic3VidHlwZVwiOlwiZWRnZVwiLFwib3JkZXJcIjo0LFwibG9ja2VkXCI6dHJ1ZSxcInNpZGVcIjpcInJpZ2h0XCJ9LHtcInR5cGVcIjpcInJlY2VwdG9yX3pvbmVcIixcIm5hbWVcIjpcIlJaQkwyXCIsXCJvcmRlclwiOjUsXCJsb2NrZWRcIjp0cnVlLFwiYXR0YWNoZWRUb0J1aWxkaW5nXCI6dHJ1ZSxcIndpZHRoXCI6XCI0XCIsXCJzaWRlXCI6XCJyaWdodFwiLFwid2hlcmVcIjpcInN0cmVldHRvZW5kXCIsXCJleGlzdGluZ19iYXJyaWVyXCI6W119LHtcInR5cGVcIjpcImJ1aWxkaW5nXCIsXCJuYW1lXCI6XCJCTDJcIixcIm9yZGVyXCI6NixcImxvY2tlZFwiOnRydWUsXCJoZWlnaHRcIjpcIjEwXCIsXCJ3aWR0aFwiOjIsXCJzaWRlXCI6XCJyaWdodFwiLFwid2hlcmVcIjpcInN0cmVldHRvZW5kXCJ9XSIsImxvY2tlZCI6bnVsbCwiUE0yNSI6MTYsIk5PMiI6NDAsImNyZWF0ZWRfYXQiOiIyMDIwLTA4LTE0IDEyOjU0OjU5IiwidXBkYXRlZF9hdCI6IjIwMjAtMDgtMTcgMTY6MjY6MDkiLCJkZWxldGVkX2F0IjpudWxsLCJubzJfYmdfY29uY2VudHJhdGlvbiI6NDAsInBtMnA1X2JnX2NvbmNlbnRyYXRpb24iOjE2LCJwcm9qZWN0Ijp7ImlkIjo4MywiYXV0aG9yX2lkIjoyMywibmFtZSI6IjEwdGggQXVnIDIwMjAiLCJ0YWciOiIxMHRoLWF1Zy0yMDIwIiwiY3JlYXRlZF9hdCI6IjIwMjAtMDgtMTAgMTQ6MDg6NDciLCJ1cGRhdGVkX2F0IjoiMjAyMC0wOC0xMCAxNDowODo0NyIsImRlbGV0ZWRfYXQiOm51bGx9fQ=="

# decode the base 64 string into JSON format
#content = json.loads(base64.b64decode(base64_test))

# ___________ same from now on ___________

# display ALL the information from user interface (uncomment to see)
#print(json.dumps(content, indent = 4))

# extract the "objects" field 
objects = content["objects"]

# and convert to a list of dictionaries:
objects = json.loads(objects)

# for visual inspection (uncomment to see):
#print(json.dumps(objects, indent = 4))



# ___________ extract background pollution concentrations ___________

cB_no2 = content["no2_bg_concentration"] # UNIT: ug/m3 

cB_pm25 = content["pm2p5_bg_concentration"] # UNIT: ug/m3 


# ___________ climatological wind characteristics ___________

# ___________ choose closest met station ______________

# read in the met station sites from the github repository
# (to get the path go on the file, click 'raw', and copy the URL)

url = 'https://raw.githubusercontent.com/pearce-helen/GI4RAQ-open/master/meteorology/locations.csv'
sites_df = pd.read_csv(url)

# visual inspection
#print(sites_df)

# combine all possible met station locations into 1 vector
points = []
for i in range(len(sites_df)):
    points.append([float(sites_df['longitude'][i]), float(sites_df['latitude'][i])])

# convert vector into Multipoint
destinations = MultiPoint(points)
#print(destinations)

# set street location
# extract the location of the street, type float to ensure correct format
latitude = float(content["lat"])
longitude = float(content["lng"])

# combine into one point value
orig = (longitude, latitude)
orig = Point(orig)
#print(orig)

# find the nearest met station to the street
# website for reference: https://automating-gis-processes.github.io/2017/lessons/L3/nearest-neighbour.html
nearest_site = nearest_points(orig, destinations)
# nearest_site[0] is the origin (street) while nearest_site[1] is the closest destination
nearest_met = nearest_site[1] 
# extract the x y (lon, lat) for the closest station
nearest_met_lat = nearest_met.y
nearest_met_lon = nearest_met.x
# select the site ID from the sites_df
station_id = sites_df.loc[(sites_df['latitude'] == nearest_met_lat) & (sites_df['longitude'] == nearest_met_lon), 'station'].item()
# visual inspection
#print(orig, station_id)

station_id = int(station_id)
# the station ID will be used to pull in the correct wind data file from github

# _______________ calculate weighted L-R and R-L cross-canyon wind speed (u) _______________

# convert street direction to degrees
street_dir_chr = content["wind"]
street_dir = 0

if street_dir_chr == "N":
    street_dir = 360
elif street_dir_chr == "NNE":
    street_dir = 22.5
elif street_dir_chr == "NE":
    street_dir = 45
elif street_dir_chr == "ENE":
    street_dir = 67.5
elif street_dir_chr == "E":
    street_dir = 90
elif street_dir_chr == "ESE":
    street_dir = 112.5
elif street_dir_chr == "SE":
    street_dir = 135
elif street_dir_chr == "SSE":
    street_dir = 157.5
elif street_dir_chr == "S":
    street_dir = 180
elif street_dir_chr == "SSW":
    street_dir = 202.5
elif street_dir_chr == "SW":
    street_dir = 225
elif street_dir_chr == "WSW":
    street_dir = 247.5
elif street_dir_chr == "W":
    street_dir = 270
elif street_dir_chr == "WNW":
    street_dir = 292.5
elif street_dir_chr == "NW":
    street_dir = 315
elif street_dir_chr == "NNW":
    street_dir = 337.5

# visual inspection
#print(street_dir)

# select the station data based on the above:
root = "https://raw.githubusercontent.com/pearce-helen/GI4RAQ-open/master/meteorology/"
# combine the root and the station info into correct URL
#url_station = (f"{root}station{station_id}.csv")
url_station = root + "station" + str(station_id) + ".csv"
# visual inspection
#print(url_station)

# read in wind data from github
wind = pd.read_csv(url_station)

# visual inspection
#print(wind)

# wind stored in a pandas dataframe
# developing with a pandas dataframe is different to in R, safest way to 
# calculate things is to define a function first and then apply it using 
# the lambda method. 
# Also remember .copy() if making a new dataframe - ensures not just 'viewing'
# a dataframe but actually making alterations

# remove wind direction = 0 as per the PPT 0 should always be treated as parallel
# we can drop this here as we'll only ever work out the L-R occurence and R-L occurence directly,
# paralled occurrence will then be the remainder
wind = wind[wind.wind_direction != 0].copy()

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

# secondly, calculate theta: the difference between the perpendicular angle 
# (relative to the street) and the relative wind directions for both the L-R and R-L cases
# relative perpendicular wind at 270 degrees for L-R, and 90 degrees for R-L
def dif_to_perp(perp,wind_d):
    return abs(perp-wind_d)

wind['LR_difference'] = wind.apply(lambda wind: dif_to_perp(270, wind['street_dir_as_N']), axis=1)
wind['RL_difference'] = wind.apply(lambda wind: dif_to_perp(90, wind['street_dir_as_N']), axis=1)


# create subsets that only include data that is relevant for each sector 
# i.e. 45 +/- from the perpendicular
LR_only = wind.loc[wind.LR_difference <= 45].copy()
RL_only = wind.loc[wind.RL_difference <= 45].copy()


# convert theta into radians as this is what math.cos assumes
def calculate_radians(degrees):
    return math.radians(degrees)

LR_only["LR_dif_rad"] = LR_only.apply(lambda LR_only: calculate_radians(LR_only['LR_difference']), axis=1)
RL_only["RL_dif_rad"] = RL_only.apply(lambda RL_only: calculate_radians(RL_only['RL_difference']), axis=1)


# calculate the perpendicular component (across canyon) of each of the wind vectors

# IMPORTANT: the angles calculated only need to be relative to the perpendicular
# but need to still use the original wind speed values from the original wind directions
# that they represent

# cosine used as the hypotenuse of a right-angled triangle is the original wind speed
# the acute angle has been calculated (theta - relative to the perpendicular)
# and the adjacent side of the right-angled triangle represents the cross-street
# component of the wind vector
def calculate_u(theta, wind_speed):
    return math.cos(theta)*wind_speed

LR_only["LR_u"] = LR_only.apply(lambda LR_only: calculate_u(LR_only['LR_dif_rad'],LR_only['wind_speed']), axis=1)
RL_only["RL_u"] = RL_only.apply(lambda RL_only: calculate_u(RL_only['RL_dif_rad'],RL_only['wind_speed']), axis=1)


# weighting of wind sectors needed to give greater 'weight' to the wind speeds
# and directions that occur more frequently, and vice versa
# firstly calculate how frequently the total sector of L-R and R-L occurs
LR_freq = LR_only['fractional_occur'].sum()
RL_freq = RL_only['fractional_occur'].sum()

def weighted_u(foccur,u,dir_freq):
    return (foccur*u)/dir_freq

LR_only["LR_u_weighted"] = LR_only.apply(lambda LR_only: weighted_u(LR_only['fractional_occur'], LR_only['LR_u'],LR_freq), axis=1)
RL_only["RL_u_weighted"] = RL_only.apply(lambda RL_only: weighted_u(RL_only['fractional_occur'], RL_only['RL_u'],RL_freq), axis=1)


# finally sum the fractional wind speed components
LR_u = LR_only['LR_u_weighted'].sum()
RL_u = RL_only['RL_u_weighted'].sum()


# checking
#print(LR_only['wind_direction'], LR_only['street_dir_as_N'])
#print(LR_only['LR_difference'])
#print(LR_only['LR_dif_rad'])
#print(LR_only['wind_speed'])
#print(LR_only['LR_u'])


#print(RL_only['wind_direction'], RL_only['street_dir_as_N'])
#print(RL_only['RL_difference'])
#print(RL_only['RL_dif_rad'])
#print(RL_only['wind_speed'])
#print(RL_only['RL_u'])

# _______________ assign values for use further in code _______________

# wind speed left to right (m/s) - climatological average
ubg_orig = LR_u

# wind speed right to left (m/s) - climatological average
ubg_mir = RL_u

# frequency of wind condition occurence (i.e. how often the wind is blowing L->R)
LR_freq = LR_freq
RL_freq = RL_freq
LR_par_freq = (1-LR_freq-RL_freq)/2
RL_par_freq = (1-LR_freq-RL_freq)/2

#print(LR_freq, RL_freq, LR_par_freq, RL_par_freq)
#sum([LR_freq, RL_freq, LR_par_freq, RL_par_freq])


# ___________ extract GI intervention information ___________

# NOTE:
# dictionaries within a list must first be searched by index of list, 
# then can use key:value pairs, otherwise keys are not 'seen' within higher 
# level of list

#extract information for specific variables if names are known by searching:
#next((item.get('height') for item in objects if item["name"] == "BL1"), False)

# if objects[index][key] != []
# means only where GI fields are complete is the information retained

# extract the information on the GI by storing in another container
gi = []           
for index in range(len(objects)):
    for key in objects[index]:
        if key == "gi4raq_barrier":
            if objects[index][key] != []:
                gi = objects[index][key]

#print(gi)

# obtain the zone in which the GI is placed
gi_zone = 0     # initialising value
for index in range(len(objects)):
    for key in objects[index]:
        if key == "gi4raq_barrier":
            if objects[index][key] != []:
                gi_zone = objects[index]["name"]
#print(gi_zone)
gi_zone_w = next((item.get('width') for item in objects if item["name"]==gi_zone), 0)

# determine exactly where the GI barrier is (m from left to middle of barrier)
gi_loc = 0  # initialising value
a = 0     # just a counter

# firstly determine distance from left edge of street to the zone in which the
# GI is located 
for item in objects:
    if item["name"] == gi_zone:
        break
    if item["type"] != "marker":
        if item["type"] != "building":
            a = float(item.get("width"))
            gi_loc = gi_loc + a
            gi_loc = round(gi_loc, 4)

#print(gi_loc)

# extract distance from lhs of zone to middle of barrier
gi_loc2 = next((item.get('where_in_zone') for item in gi), 0)
#print(gi_loc2)

if gi_loc2 == 0:
    gi_loc2 = gi_loc2 + 0.1
elif gi_loc2 == gi_zone_w:
    gi_loc2 = gi_loc2 - 0.1

# combine location of zone and location of object within the zone to get
# the location of the barrier in meters from left edge of street
gi_loc = round(float(gi_loc) + float(gi_loc2), 4)

#print(gi_loc)

###############################################################################
# GEOMETRY: WIND ACROSS STREET LEFT --> RIGHT (original)
###############################################################################

# ___________ total street width ___________
a = 0 # just a counter
roadw= 0  # initialising value

# calculate total street width by adding zone widths together
# (excluding marker & building widths)        
for item in objects:
    if item["type"] != "marker":
        if item["type"] != "building":
            a = float(item.get("width"))
            roadw = roadw + a
            roadw = round(roadw, 4)

#print(roadw)            

# ___________ set horizontal (row_original) defining values ___________

# row_original = storage container initialised with zeros
row_original = np.array([0,0,0,0,0,0,0,0,0,0], dtype = float)

# left hand side building height
row_original[0]=next((item.get('height') for item in objects if item["name"]=="BL1"), 0)

# right hand side building height
row_original[1]=next((item.get('height') for item in objects if item["name"]=="BL2"), 0)

# hedge height
row_original[2] = 0
for item in gi:
    if item["type"] != "green-barrier":
        row_original[2] = 0
    else:
        row_original[2] = item.get("height")

# proposed grey infra height
row_original[3] = 0
for item in gi:
    if item["type"] != "grey-barrier":
        row_original[3] = 0
    else:
        row_original[3] = item.get("height")
        
# tree base height at maturity
row_original[4] = 0
for item in gi:
    if item["type"] == "green-barrier-trees":
        row_original[4] = item.get("tcbh")
    elif item["type"] == "grey-barrier-trees":
        row_original[4] = item.get("tcbh")
    else:
        row_original[4] = 0

# tree top height at maturity
row_original[5] = 0
for item in gi:
    if item["type"] == "green-barrier-trees":
        row_original[5] = item.get("tcth")
    elif item["type"] == "grey-barrier-trees":
        row_original[5] = item.get("tcth")
    else:
        row_original[5] = 0

# existing barrier upwind height
# extract all information on upwind existing barrier and store in eb_up
eb_up = []
for item in objects:
    if item["name"] == "RZBL1":
        eb_up = item.get("existing_barrier")           

# if eb_up contains information, proceed to extract existing barrier height
if eb_up != []:
    row_original[8] = next((item.get('height') for item in eb_up), False)
else:
    row_original[8] = 0
    

# existing barrier downwind height
# extract all information on downwind existing barrier and store in eb_down
eb_down = []
for item in objects:
    if item["name"] == "RZBL2":
        eb_down = item.get("existing_barrier")           

# if eb_down contains information, proceed to extract existing barrier height
if eb_down != []:
    row_original[9] = next((item.get('height') for item in eb_down), False)
else:
    row_original[9] = 0

#print(row_original)

# ___________ set vertical defining values ___________

# zone_original = storage container
zone_original = np.array([0,0,0,0,0], dtype = float)

# for each, add up the widths of objects until you hit the marker
# upwind street boundary
a = 0  
zone_original[0] = 0
for item in objects:
    if item["type"] != "marker":
        if item["type"] != "building":
            a = float(item.get("width"))
            zone_original[0] = zone_original[0] + a
            zone_original[0] = round(zone_original[0], 4)
            if zone_original[0] == 0:
                zone_original[0] = 0.01
    elif item["name"] == "Street Boundary 1":
        break

# determine where emission zones start and end
# first, initialise values
ez1_start = 0
ez1_finish = 0
ez2_start = 0
ez2_finish = 0

# ez1 finish
a = 0    
for item in objects:
    if item["type"] != "marker":
        if item["type"] != "building":
            a = float(item.get("width"))
            ez1_finish = ez1_finish + a
            ez1_finish = round(ez1_finish, 4)
    if item["name"] == "EZ1":
        break

# ez1 start
ez1_w = 0
for item in objects:
    if item["name"] == "EZ1":
        ez1_w = float(item.get("width"))
        
ez1_start = ez1_finish - ez1_w

# ez2 finish
a = 0    
for item in objects:
    if item["type"] != "marker":
        if item["type"] != "building":
            a = float(item.get("width"))
            ez2_finish = ez2_finish + a
            ez2_finish = round(ez2_finish, 4)
    if item["name"] == "EZ2":
        break
 
# ez2 start   
ez2_w = 0
for item in objects:
    if item["name"] == "EZ2":
        ez2_w = float(item.get("width"))
     
ez2_start = ez2_finish - ez2_w  

# determine number of emission zones
nez = 0
if ez2_start == ez2_finish:
    nez = 1
else:
    nez = 2    
    
#print(nez)
    
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
a = 0    
zone_original[3] = 0
for item in objects:
    if item["type"] != "marker":
        if item["type"] != "building":
            a = float(item.get("width"))
            zone_original[3] = zone_original[3] + a
            zone_original[3] = round(zone_original[3], 4)
            if zone_original[3] == roadw:
                zone_original[3] = roadw - 0.01
    if item["name"] == "Street Boundary 2":
        break

# downwind building
zone_original[4] = roadw 

#print(zone_original)

# barrier locations (m from left of street)
# bar_original = a strage container
bar_original = np.array([0,0,0,0], dtype = float)

# existing barrier upwind
if eb_up == []:
    bar_original[0] = 0
else:
    # if there is an existing barrier upwind it is defaulted to SB:
    bar_original[0] = zone_original[0]
    
    # alternative methodology to not default to SB:
    #bar_original[0] = next((item.get('metersinzone_original') for item in eb_up), 0)


# existing barrier downwind
if eb_down == []:
    bar_original[2] = 0
else:
    # make the position default to only being on the street boundary:
    bar_original[2] = zone_original[3]
    
    # alternative methodology to not default to SB:
    #bar_original[2] = zone_original[3] + next((item.get('metersinzone_original') for item in eb_down), 0)
    
    
# new barriers upwind and downwind done together
if gi_loc <= zone_original[1]:
    bar_original[1] = gi_loc
    bar_original[3] = 0
elif gi_loc >= zone_original[2]:
    bar_original[1] = 0
    bar_original[3] = gi_loc

#print(bar_original)

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

#print(check_original)    

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

# ___________ reverse 'check' ___________
# existing barrier upwind
check_mirror[0] = check_original[2]
# new barrier upwind
check_mirror[1] = check_original[3]
#existing barrier downwind
check_mirror[2] = check_original[0]
# new barrier downwind
check_mirror[3] = check_original[1]


# ___________ reverse 'bar' ___________

if check_mirror[0] == 1:
    # existing barrier upwind position (not height)
    bar_mirror[0] = roadw - bar_original[2]
else:
    bar_mirror[0] = 0
    
if check_mirror[1] == 1:
    # new barrier upwind position (not height)
    bar_mirror[1] = roadw - bar_original[3]
else:
    bar_mirror[1] = 0
    
if check_mirror[2] == 1:
    # existing barrier downwind position (not height)
    bar_mirror[2] = roadw - bar_original[0]
else:
    bar_mirror[2] = 0

if check_mirror[3] == 1:
    # new barrier downwind position (not height)
    bar_mirror[3] = roadw - bar_original[1]
else:
    bar_mirror[3] = 0


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

# upwind existing barrier recirc
# reassign recirc to not carry over anything calculated previously
recirc = 0
# if there is an existing barrier upwind
if check_mirror[0] == 1:
    # the recirculation = 3H-3
    recirc = (row_mirror[8]*3)-3
    # if the above calculation results <= 0 a default of 0.01 is applied
    if recirc <= 0:
        recirc = 0.01
    # finally, vertical reference of recirculation touch-down is: 
    # the width of recirculation + the location of the barrier
    rec_mirror[1] = recirc+bar_mirror[0]
else:
    rec_mirror[1] = 0
    

# upwind new barrier recirc 
# reassign recirc to not carry over anything calculated previously
recirc = 0
# if there is a new barrier upwind
if check_mirror[1] == 1:
    # the recirculation = 3H-3
    recirc = (max(row_mirror[2:6])*3)-3
    # if the above calculation results <= 0 default of 0.01 is applied
    if recirc <= 0:
        recirc = 0.01
    # finally, vertical reference of recirculation touch-down is: 
    # the width of recirculation + the location of the barrier
    rec_mirror[2] = recirc+bar_mirror[1]
else:
    rec_mirror[2] = 0

# downwind existing barrier recirc
# reassign recirc to not carry over anything calculated previously
recirc = 0
# if there is an existing barrier downwind
if check_mirror[2] == 1:
    # the recirculation = 3H-3
    recirc = (row_mirror[9]*3)-3
    # if the above calculation results <= 0 default of 0.01 is applied
    if recirc <= 0:
        recirc = 0.01
    # finally, vertical reference of recirculation touch-down is: 
    # the width of recirculation + the location of the barrier
    rec_mirror[3] = recirc+bar_mirror[2]
else:
    rec_mirror[3] = 0

# downwind new barrier recirc 
# reassign recirc to not carry over anything calculated previously
recirc = 0
# if there is a new barrier downwind
if check_mirror[3] == 1:
    # the recirculation = 3H-3
    recirc = (max(row_mirror[2:6])*3)-3
    # if the above calculation resuls <= 0 default of 0.01 is applied
    if recirc <= 0:
        recirc = 0.01
    # finally, vertical reference of recirculation touch-down is: 
    # the width of recirculation + the location of the barrier
    rec_mirror[4] = recirc+bar_mirror[3]
else:
    rec_mirror[4] = 0

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
h_mirror = row_dimensioning(row = row_mirror)

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
                    
    elif rec[0] >= zone[0] and rec[0] < zone[1]:
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
        
    elif rec[0] >= zone[1] and rec[0] < zone[2]:
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
            
    elif rec[0] >= zone[2] and rec[0] < zone[3]:
        # CASE 4
        if rec[0] == zone[2]:
            rec[0] = rec[0] - 0.1
        
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
                if bar[3] < rec[0]:
                    l[2] = bar[3] - l[1]
                    l[3] = rec[0] - l[1] - l[2]
                    l[4] = zone[3] - l[1] - l[2] - l[3]
                elif bar[3] == rec[0]:
                    l[2] = zone[2] - l[1]
                    l[3] = bar[3] - l[1] - l[2]
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
    
    elif rec[0] >= zone[3] and rec[0] < zone[4]:
        # CASE 5
        if rec[0] == zone[3]:
            rec[0] = rec[0]+0.1
        
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
l_mirror = column_dimensioning(rec=rec_mirror, zone=zone_mirror, 
                               check=check_mirror, bar=bar_mirror, 
                               row = row_mirror)
    
#print(l_original)
#print(l_mirror)

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

# defined in terms of HOW MUCH THE AIR IS STOPPED 
# e.g. 80% obstruction = only 20% of air can get through barrier
 
# upwind existing
obs_original[0] = next((item.get('obst') for item in eb_up), 0)
obs_original[0] = obs_original[0]/100

# downwind existing
obs_original[2] = next((item.get('obst') for item in eb_down), 0)
obs_original[2] = obs_original[2]/100

# upwind new and downwind new done together as there can only be 1

# account for a dispropotionate decrease in blocking ability by a barrier
# if there's a gap between the barrier and tree line

for item in gi:
    # if it's a hedge + trees:
    if item["type"] == "green-barrier-trees":
        # obstruction of barrier value
        obar = float(item.get("obst"))
        obar = obar/100
        #obstruction of tree value
        otree = float(item.get("tobst"))
        otree = otree/100
        # obstruction of gap assumed to be 0.1
        ogap = 0.1
        
        # heights
        # total barrier height
        htotal = float(item.get("tcth"))
        # just tree crown
        htree = float(item.get("tcth"))-float(item.get("tcbh"))
        # calculate gap between hedge and tree
        hgap = float(item.get("tcbh"))-float(item.get("height"))
        # just hedge height
        hbar = float(item.get("height"))
        
        # work out gi obstruction based on disproportionate influence of gap
        ototal = (((1/obar)*(hbar/htotal))+((1/ogap)*(hgap/htotal))+((1/otree)*(htree/htotal)))**-1
        gi_obs = round(ototal, 2)
    
    # if it's a fence/wall + trees use same methodology as above: 
    elif item["type"] == "grey-barrier-trees":
        # obstruction of barrier value
        obar = float(item.get("obst"))
        obar = obar/100
        #obstruction of tree value
        otree = float(item.get("tobst"))
        otree = otree/100
        # obstruction of gap assumed to be 0.1
        ogap = 0.1
        
        # heights
        # total barrier height
        htotal = float(item.get("tcth"))
        # just tree crown
        htree = float(item.get("tcth"))-float(item.get("tcbh"))
        # calculate gap between hedge and tree
        hgap = float(item.get("tcbh"))-float(item.get("height"))
        # just fence/wall height
        hbar = float(item.get("height"))
        
        # work out gi obstruction based on disproportionate influence of gap
        ototal = (((1/obar)*(hbar/htotal))+((1/ogap)*(hgap/htotal))+((1/otree)*(htree/htotal)))**-1
        gi_obs = round(ototal, 2)
    
    # if it's just a barrier and no trees:
    else:
        gi_obs = float(item.get("obst"))/100

        
# assign to the correct location (upwind or downwind)
if gi_loc <= zone_original[1]:
    obs_original[1] = gi_obs
    obs_original[3] = 0
elif gi_loc >= zone_original[2]:
    obs_original[1] = 0
    obs_original[3] = gi_obs

#print(obs_original)

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

# ___________ flip for mirror image ___________

obs_mirror = np.array([0,0,0,0], dtype = float)
#existing upwind
obs_mirror[0] = obs_original[2]
#new upwind
obs_mirror[1] = obs_original[3]
#existing downwind
obs_mirror[2] = obs_original[0]
#new downwind
obs_mirror[3] = obs_original[1]


# if any barriers have 100% coverage, change obstruction to 0.99 otherwise solver 
# issues occur because of knock-on effect on advection and dispersion patterns;
# now dispersion is calculated from advection values (adv*0.1)
# but if obs=1, advection will = 0 in some places and throw out the calculations

for i in range(len(obs_original)):
    if obs_original[i]==1:
        obs_original[i]=0.99
    else:
        obs_original[i]=obs_original[i]
        
for i in range(len(obs_mirror)):
    if obs_mirror[i]==1:
        obs_mirror[i]=0.99
    else:
        obs_mirror[i]=obs_mirror[i]
        
#print(obs_original)
#print(obs_mirror)

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
    if w > (3*H) and w <= (5*H):
        d = (1.75*H)-(0.35*w)
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
Ur_orig = (0.1*Uh_orig)
Ut_orig = ws_point(ubg = ubg_orig,z=max(row_original[0],row_original[1]),H = H_orig, w = w)

U3_mir = ws_average(row_min = h_cumu_mirror[2], row_max = h_cumu_mirror[3], ubg = ubg_mir, H = H_mir)
U2_mir = ws_average(row_min = h_cumu_mirror[1], row_max = h_cumu_mirror[2], ubg = ubg_mir, H = H_mir)
U1_mir = ws_average(row_min = 0, row_max = h_cumu_mirror[1], ubg = ubg_mir, H = H_mir)
Uh_mir = ws_point(ubg = ubg_mir, z = H_mir, H = H_mir, w = w)
Ur_mir = (0.1*Uh_mir)
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
    dis = 0.01*Uh

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

# 1) ----- define functions -----
# 1a) flow effected by barrier placed outside the recirculation zone

def bar_outside(obs, bar_col, bar_recirc, ue, ua, we, wa, h, l, l_cumu):
    # dispersion at barrier: reduce dispersion that it currently is by barrier effect
    # ue1[1,(bar_col+1)] = (1-obs)*ue1[1,(bar_col+1)]
    # this is now altered based on 10% of the slowed advection term (U1) below
    
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
    rec_ncol_bar = recirc_col(recirc = bar_recirc, l_cumu=l_cumu)
    
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
        bar_outside(obs=obs, bar_col=1, bar_recirc=rec, ue=ue, ua=ua, we=we, wa=wa, h=h, l=l, l_cumu=l_cumu)
    if bar == l_cumu[2]:
        bar_outside(obs=obs, bar_col=2, bar_recirc=rec, ue=ue, ua=ua, we=we, wa=wa, h=h, l=l, l_cumu=l_cumu)
    if bar == l_cumu[3]:
        bar_outside(obs=obs, bar_col=3, bar_recirc=rec, ue=ue, ua=ua, we=we, wa=wa, h=h, l=l, l_cumu=l_cumu)
    if bar == l_cumu[4]:
        bar_outside(obs=obs, bar_col=4, bar_recirc=rec, ue=ue, ua=ua, we=we, wa=wa, h=h, l=l, l_cumu=l_cumu)
        
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

ue2_mir = ue1_mir.copy()
ua2_mir = ua1_mir.copy()
we2_mir = we1_mir.copy()
wa2_mir = wa1_mir.copy()

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
            # the recirc, take the largest to account for advection changes (& disp included), 
            # but include the effect on dispersion from the smaller barrier
    
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
                # decrease advection (& disp included) based on upwind existing
                bar_inside_check(bar=bar[0], obs=obs[0], ue=ue2, ua=ua2, we=we2, wa=wa2, h=h, l=l, rec_ncol=rec_ncol, rec_nrow=rec_nrow, l_cumu=l_cumu)
                
                # decrease dispersion based on downwind existing
                bar_inside_disp(bar=bar[2], ue=ue2, obs=obs[2], l_cumu=l_cumu)
    
                
            # if existing downwind is largest:
            if row[9] >= row[8]:
                # decrease advection (& disp included) based on downwind existing
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

existing_barriers_mir = existing_barrier_pattern(check=check_mirror, l_cumu=l_cumu_mirror, 
                                                  rec_ncol=rec_ncol_mir, rec_nrow=rec_nrow_mir, 
                                                  bar=bar_mirror, obs=obs_mirror, rec=rec_mirror, 
                                                  ue2=ue2_mir, ua2=ua2_mir, we2=we2_mir, wa2=wa2_mir, 
                                                  h=h_mirror, l=l_mirror, row=row_mirror)

ue2_mir = existing_barriers_mir[0]
ua2_mir = existing_barriers_mir[1]
we2_mir = existing_barriers_mir[2]
wa2_mir = existing_barriers_mir[3]

# print("ue2_mir")
# print(ue2_mir)
# print("ua2_mir")
# print(ua2_mir)
# print("we2_mir")
# print(we2_mir)
# print("wa2_mir")
# print(wa2_mir)

# ___________ Advection & Dispersion Assignment: NEW GI BARRIER ___________

# make a fresh copy of base advection patterns
ue3_orig = ue1_orig.copy()
ua3_orig = ua1_orig.copy()
we3_orig = we1_orig.copy()
wa3_orig = wa1_orig.copy()

ue3_mir = ue1_mir.copy()
ua3_mir = ua1_mir.copy()
we3_mir = we1_mir.copy()
wa3_mir = wa1_mir.copy()

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
            # apply advection changes (disp included) based on GI
            # if the barrier is upwind
            if check[1] == 1:
                # apply advection changes based on new barrier upwind
                bar_inside_check(bar=bar[1], obs=obs[1], ue=ue3, ua=ua3, we=we3, wa=wa3, l_cumu=l_cumu, h=h, l=l, rec_ncol=rec_ncol, rec_nrow=rec_nrow)
                
                # apply dispersion to the existing barriers
                bar_inside_disp(bar=bar[0], ue=ue3, obs=obs[0], l_cumu=l_cumu)
                bar_inside_disp(bar=bar[2], ue=ue3, obs=obs[2], l_cumu=l_cumu)
                
            # else if the new barrier is downwind
            elif check[3] == 1:
                # apply advection changes (disp included) based on new barrier downwind
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
    
    
    # 2c) if the recirc covers the upwind existing and new GI placed UPWIND, 
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


new_barrier_mir = new_barrier_pattern(check=check_mirror, l_cumu=l_cumu_mirror, 
                                       bar=bar_mirror, obs=obs_mirror, rec=rec_mirror, 
                                       ue3=ue3_mir, ua3=ua3_mir, we3=we3_mir, wa3=wa3_mir, 
                                       h=h_mirror, l=l_mirror, rec_ncol=rec_ncol_mir, 
                                       zone=zone_mirror, row=row_mirror, rec_nrow=rec_nrow_mir)

ue3_mir = new_barrier_mir[0]
ua3_mir = new_barrier_mir[1]
we3_mir = new_barrier_mir[2]
wa3_mir = new_barrier_mir[3]

# print("ue3_mir")
# print(ue3_mir)
# print("ua3_mir")
# print(ua3_mir)
# print("we3_mir")
# print(we3_mir)
# print("wa3_mir")
# print(wa3_mir)

# ___________ Parallel wind: dispersion only ___________

# we don't want any influence of recirculation zones; for these to occur wind 
# would need to be across the street

# determine ACH
# diffusion at boundary between 'canyon' and air above from Lui et al 2015

hb = np.array([0.050220445242621700,0.08079710966000090,0.11132828240100300,
               0.14194339944960800,0.16935309231015000,0.20093573286600600,
               0.23447311340516000,0.2661502802429710,0.2971943299070450,
               0.39162616262406600,0.3373974717744690,0.5021058431332970,
               0.7683252782298300,1.2562405019233200,1.9981998376352400,
               3.3218595969318200,5.000314809594420])
ACH = np.array([0.0727961068441183,0.07525731521502540,0.07479088280108210,
                0.07138592617929660,0.0671476103112659,0.06248872788332970,
                0.05698948972293910,0.05149413849933150,0.04852918312136570,
                0.04259149849186860,0.04514055163406820,0.03960710842998850,
                0.0370191859199602,0.03357924686712890,0.03140722659286670,
                0.028386299325227800,0.027907428713579400])

# calculate aspect ratio
ar = ((row_original[0]+row_original[1])/2)/roadw
ACH_value = np.interp(ar, xp = hb, fp = ACH)

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
ue1_par[1,2] = ACH_value*0.8 #c11 and c12
ue1_par[1,3] = ACH_value*0.8 #c12 and c13
ue1_par[1,4] = ACH_value*0.8 #c13 and c14
ue1_par[1,5] = ACH_value*0.8 #c14 and c15
ue1_par[2,1] = 0 #c21 to wall
ue1_par[2,2] = ACH_value*0.9 #c21 and c22
ue1_par[2,3] = ACH_value*0.9 #c22 and c23
ue1_par[2,4] = ACH_value*0.9 #c23 and c24
ue1_par[2,5] = ACH_value*0.9 #c24 and c25
ue1_par[3,1] = 0 #c31 to wall
ue1_par[3,2] = ACH_value #c31 and c32
ue1_par[3,3] = ACH_value #c32 and c33
ue1_par[3,4] = ACH_value #c33 and c34
ue1_par[3,5] = ACH_value #c34 and c35

#vertical dispersion co-efficients
#we[1,1] = 0 #ground to c11
#we[1,2] = 0 #ground to c12
#we[1,3] = 0 #ground to c13
#we[1,4] = 0 #ground to c14
#we[1,5] = 0 #ground to c15
we1_par[2,1] = ACH_value*0.8 #c11 to c21
we1_par[2,2] = ACH_value*0.8 #c12 and c22
we1_par[2,3] = ACH_value*0.8 #c13 and c23
we1_par[2,4] = ACH_value*0.8 #c13 and c24
we1_par[2,5] = ACH_value*0.8 #c15 and c25
we1_par[3,1] = ACH_value*0.9 #c21 and c31
we1_par[3,2] = ACH_value*0.9 #c22 and c32
we1_par[3,3] = ACH_value*0.9 #c23 and c33
we1_par[3,4] = ACH_value*0.9 #c24 and c34
we1_par[3,5] = ACH_value*0.9 #c25 and c35
we1_par[4,1] = ACH_value #c31 and cb
we1_par[4,2] = ACH_value #c32 and cb
we1_par[4,3] = ACH_value #c33 and cb
we1_par[4,4] = ACH_value #c24 and cb
we1_par[4,5] = ACH_value #c35 and cb

ue2_par[1,1] = 0 #c11 to wall
ue2_par[1,2] = ACH_value*0.8 #c11 and c12
ue2_par[1,3] = ACH_value*0.8 #c12 and c13
ue2_par[1,4] = ACH_value*0.8 #c13 and c14
ue2_par[1,5] = ACH_value*0.8 #c14 and c15
ue2_par[2,1] = 0 #c21 to wall
ue2_par[2,2] = ACH_value*0.9 #c21 and c22
ue2_par[2,3] = ACH_value*0.9 #c22 and c23
ue2_par[2,4] = ACH_value*0.9 #c23 and c24
ue2_par[2,5] = ACH_value*0.9 #c24 and c25
ue2_par[3,1] = 0 #c31 to wall
ue2_par[3,2] = ACH_value #c31 and c32
ue2_par[3,3] = ACH_value #c32 and c33
ue2_par[3,4] = ACH_value #c33 and c34
ue2_par[3,5] = ACH_value #c34 and c35

#vertical dispersion co-efficients
#we[1,1] = 0 #ground to c11
#we[1,2] = 0 #ground to c12
#we[1,3] = 0 #ground to c13
#we[1,4] = 0 #ground to c14
#we[1,5] = 0 #ground to c15
we2_par[2,1] = ACH_value*0.8 #c11 to c21
we2_par[2,2] = ACH_value*0.8 #c12 and c22
we2_par[2,3] = ACH_value*0.8 #c13 and c23
we2_par[2,4] = ACH_value*0.8 #c13 and c24
we2_par[2,5] = ACH_value*0.8 #c15 and c25
we2_par[3,1] = ACH_value*0.9 #c21 and c31
we2_par[3,2] = ACH_value*0.9 #c22 and c32
we2_par[3,3] = ACH_value*0.9 #c23 and c33
we2_par[3,4] = ACH_value*0.9 #c24 and c34
we2_par[3,5] = ACH_value*0.9 #c25 and c35
we2_par[4,1] = ACH_value #c31 and cb
we2_par[4,2] = ACH_value #c32 and cb
we2_par[4,3] = ACH_value #c33 and cb
we2_par[4,4] = ACH_value #c24 and cb
we2_par[4,5] = ACH_value #c35 and cb

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
#upwind
if check_mirror[0] == 1:
    bar_inside_disp(bar=bar_mirror[0], ue=ue2_par, obs=obs_mirror[0], l_cumu=l_cumu_mirror)
#downwind    
if check_mirror[2] == 1:
    bar_inside_disp(bar=bar_mirror[2], ue=ue2_par, obs=obs_mirror[2], l_cumu=l_cumu_mirror)

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
    
# make copies of existing dispersion patterns
ue3_par = ue1_par.copy()
ua3_par = ua1_par.copy()
we3_par = we1_par.copy()
wa3_par = wa1_par.copy()

ue4_par = ue2_par.copy()
ua4_par = ua2_par.copy()
we4_par = we2_par.copy()
wa4_par = wa2_par.copy()

# apply function to new barriers
if check_original[1] == 1:
    bar_inside_disp(bar=bar_original[1], ue=ue3_par, obs=obs_original[1], l_cumu=l_cumu_original)

if check_original[3] == 1:
    bar_inside_disp(bar=bar_original[3], ue=ue3_par, obs=obs_original[3], l_cumu=l_cumu_original)

if check_mirror[1] == 1:
    bar_inside_disp(bar=bar_mirror[1], ue=ue4_par, obs=obs_mirror[1], l_cumu=l_cumu_mirror)
    
if check_mirror[3] == 1:
    bar_inside_disp(bar=bar_mirror[3], ue=ue4_par, obs=obs_mirror[3], l_cumu=l_cumu_mirror)

# print("ue3_par")
# print(ue3_par)
# print("ua3_par")
# print(ua3_par)
# print("we3_par")
# print(we3_par)
# print("wa3_par")
# print(wa3_par)

# print("ue4_par")
# print(ue4_par)
# print("ua4_par")
# print(ua4_par)
# print("we4_par")
# print(we4_par)
# print("wa4_par")
# print(wa4_par)
    
###############################################################################
# EMISSIONS
###############################################################################

def emis_calc(veh_per_hour):
    # Generate an emissions value from AADT provided by user
    # units: vehicles per day
    aadt = np.array([0,0,0,0,0], dtype = float)
    # cars and taxis (81%)
    aadt[0] = veh_per_hour*0.81
    # LGVs (15%)
    aadt[1] = veh_per_hour*0.15
    # HGVs (1.5%)
    aadt[2] = veh_per_hour*0.015
    # Buses and coaches (1.5% - remainder from above)
    aadt[3] = veh_per_hour*0.015
    # Motorcycles (1%)
    aadt[4] = veh_per_hour*0.01
    
    # emission factors
    # datasets used: Fleet Weighted Road Transport Emission Factor 2017
              # Primary NO2 Emission factors for road transport (2019 version)
              # Vehicle fleet composition projections (Base 2018)
              # combined hot and cold start emissions from NAEI
              # 2019 f-NO2 values used
    # unit: g/km
    no2_ef = np.array([0,0,0,0,0], dtype = float) 
    pm25_ef = np.array([0,0,0,0,0], dtype = float)
    
    # cars
    # (petrol NOx ef * f-NO2) + (diesel NOx ef * f-NO2)/ 2 to represent 50:50 split of petrol and diesel vehicles
    no2_ef[0] = ((0.082*0.030)+(0.573*0.333))/2

    # pm2.5 from exhausts + tyre wear + brake wear + road abrasion
    # exhausts is averaged for petrol and diesel while the others are already combined
    pm25_ef[0] = ((0.001+0.011)/2)+0.005+0.003+0.004

    # lgv: just used diesel EF as fleet % was so low for petrol, multiplied by f-NO2
    no2_ef[1] = 1.241*0.327
    # pm2.5 from exhausts + tyre wear + brake wear + road abrasion
    pm25_ef[1] = 0.018+0.008+0.004+0.004

    # hgv: assume 50:50 split between artic and rigid, format: (rigid_hgv_NOx_ef*f-NO2)+(artic_hgv_NOx_ef*f-NO2)/2
    no2_ef[2] = ((1.400*0.096)+(0.693*0.081))/2
    # pm2.5 from exhausts + tyre wear + brake wear + road abrasion
    pm25_ef[2] = ((0.020+0.012+0.010+0.021)+(0.013+0.024+0.007+0.021))/2

    # buses and coaches
    no2_ef[3] = 3.119*0.096
    # pm2.5 from exhausts + tyre wear + brake wear + road abrasion
    pm25_ef[3] = 0.036+0.013+0.016+0.021

    # motorcycles
    no2_ef[4] = 0.190*0.040
    # pm2.5 from exhausts + tyre wear + brake wear + road abrasion
    pm25_ef[4] = 0.008+0.002+0.002+0.002
    
    # emission (g/km/hr) = ef (g/km/veh) * activity (veh/hr)
    no2_gkmhr = np.array([0,0,0,0,0], dtype = float) 
    no2_gkmhr[0] = no2_ef[0]*(aadt[0])
    no2_gkmhr[1] = no2_ef[1]*(aadt[1])
    no2_gkmhr[2] = no2_ef[2]*(aadt[2])
    no2_gkmhr[3] = no2_ef[3]*(aadt[3])
    no2_gkmhr[4] = no2_ef[4]*(aadt[4])

    pm25_gkmhr = np.array([0,0,0,0,0], dtype = float) 
    pm25_gkmhr[0] = pm25_ef[0]*(aadt[0])
    pm25_gkmhr[1] = pm25_ef[1]*(aadt[1])
    pm25_gkmhr[2] = pm25_ef[2]*(aadt[2])
    pm25_gkmhr[3] = pm25_ef[3]*(aadt[3])
    pm25_gkmhr[4] = pm25_ef[4]*(aadt[4])
    
    # convert into ug/m/s
    no2_ugms = np.array([0,0,0,0,0], dtype = float) 
    pm25_ugms = np.array([0,0,0,0,0], dtype = float) 
    # g to ug = *1e+6
    # 1 km to 1 m = /1e+3 (we only want 1m worth of emission, not 1000 m worth)
    # 1 hour to 1 s = /3600
    no2_ugms[0] = (no2_gkmhr[0]*1e+3)/3600
    no2_ugms[1] = (no2_gkmhr[1]*1e+3)/3600
    no2_ugms[2] = (no2_gkmhr[2]*1e+3)/3600
    no2_ugms[3] = (no2_gkmhr[3]*1e+3)/3600
    no2_ugms[4] = (no2_gkmhr[4]*1e+3)/3600

    pm25_ugms[0] = (pm25_gkmhr[0]*1e+3)/3600
    pm25_ugms[1] = (pm25_gkmhr[1]*1e+3)/3600
    pm25_ugms[2] = (pm25_gkmhr[2]*1e+3)/3600
    pm25_ugms[3] = (pm25_gkmhr[3]*1e+3)/3600
    pm25_ugms[4] = (pm25_gkmhr[4]*1e+3)/3600
    
    # total emissions from all types of vehicle
    # emissions in 1 dimenson (ug/m/s) i.e. line source
    e1d_no2 = sum(no2_ugms)  
    e1d_pm25 = sum(pm25_ugms)

    return e1d_no2, e1d_pm25

ez1_veh = 0
ez2_veh = 0
ez1_emis_no2 = 0
ez2_emis_no2 = 0
ez1_emis_pm25 = 0
ez2_emis_pm25 = 0

# for each emission zone, if emissions are provided by user in 
# "current_exposure_no2"then use these figures, 
# if not then use the vehicle movement

for item in objects:
    if item["name"] == "EZ1":
        ez1_emis_no2 = item.get("current_emissions_no2")
        if ez1_emis_no2 == None:
            ez1_veh = float(item.get("vmovement"))
            ez1_emis_no2 = float(emis_calc(ez1_veh)[0])

for item in objects:
    if item["name"] == "EZ2":
        ez2_emis_no2 = item.get("current_emissions_no2")
        if ez2_emis_no2 == None:
            ez2_veh = float(item.get("vmovement"))
            ez2_emis_no2 = float(emis_calc(ez2_veh)[0])
            
for item in objects:
    if item["name"] == "EZ1":
        ez1_emis_pm25 = item.get("current_emissions_PM2_5")
        if ez1_emis_pm25 == None:
            ez1_veh = float(item.get("vmovement"))
            ez1_emis_pm25 = float(emis_calc(ez1_veh)[1])


for item in objects:
    if item["name"] == "EZ2":
        ez2_emis_pm25 = item.get("current_emissions_PM2_5")
        if ez2_emis_pm25 == None:
            ez2_veh = float(item.get("vmovement"))
            ez2_emis_pm25 = float(emis_calc(ez2_veh)[1])

#print(ez1_emis_no2)
#print(ez1_emis_pm25)

#print(ez2_emis_no2)
#print(ez2_emis_pm25)

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

# wind left to right, after new barrier
a2_orig = a_matrix(r = r1, u = ue3_orig, w = we3_orig, U = ua3_orig, W = wa3_orig)

# wind right to left, before new barrier
a1_mir = a_matrix(r = r2, u = ue2_mir, w = we2_mir, U = ua2_mir, W = wa2_mir)
# wind right to left, after new barrier
a2_mir = a_matrix(r = r2, u = ue3_mir, w = we3_mir, U = ua3_mir, W = wa3_mir)

# wind parallel, before new barrier, dimensioning from wind left to right
a1_par_1 = a_matrix(r = r1, u = ue1_par, U = ua1_par, w = we1_par, W = wa1_par)
# wind parallel, before new barrier, dimensioning from wind right to left
a1_par_2 = a_matrix(r = r2, u = ue2_par, U = ua2_par, w = we2_par, W = wa2_par)
# wind parallel, after new barrier, dimensioning from wind left to right
a2_par_1 = a_matrix(r = r1, u = ue3_par, U = ua3_par, w = we3_par, W = wa3_par)
# wind parallel, after new barrier, dimensioning from wind right to left
a2_par_2 = a_matrix(r = r2, u = ue4_par, U = ua4_par, w = we4_par, W = wa4_par)


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


# d4_orig = inputs for new conditions, PM2.5
d4_orig = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype = float)
d4_orig[0] = ez_tot_pm25_orig[1]/l_original[1] # averaging line emission source over area of 'ground' = (ug/m2/s)
d4_orig[1] = ez_tot_pm25_orig[2]/l_original[2]
d4_orig[2] = ez_tot_pm25_orig[3]/l_original[3]
d4_orig[3] = ez_tot_pm25_orig[4]/l_original[4]
d4_orig[4] = ez_tot_pm25_orig[5]/l_original[5]
d4_orig[5] = 0      # no inputs into middle boxes from either emissions or background
d4_orig[6] = 0
d4_orig[7] = 0
d4_orig[8] = 0
d4_orig[9] = 0
d4_orig[10] = (we3_orig[4,1] - beta(wa3_orig[4,1])*wa3_orig[4,1])*cB_pm25  # background inputs into top boxes
d4_orig[11] = (we3_orig[4,2] - beta(wa3_orig[4,2])*wa3_orig[4,2])*cB_pm25  # negative for beta because if true, the sign of W[4,x] would be negative
d4_orig[12] = (we3_orig[4,3] - beta(wa3_orig[4,3])*wa3_orig[4,3])*cB_pm25  # yet as here being treated as an input, it should be positive, hence signs cancel out
d4_orig[13] = (we3_orig[4,4] - beta(wa3_orig[4,4])*wa3_orig[4,4])*cB_pm25
d4_orig[14] = (we3_orig[4,5] - beta(wa3_orig[4,5])*wa3_orig[4,5])*cB_pm25

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


# d2_mir = inputs for new conditions, NO2
d2_mir = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype = float)
d2_mir[0] = ez_tot_no2_mir[1]/l_mirror[1] # averaging line emission source over area of 'ground' = (ug/m2/s)
d2_mir[1] = ez_tot_no2_mir[2]/l_mirror[2]
d2_mir[2] = ez_tot_no2_mir[3]/l_mirror[3]
d2_mir[3] = ez_tot_no2_mir[4]/l_mirror[4]
d2_mir[4] = ez_tot_no2_mir[5]/l_mirror[5]
d2_mir[5] = 0      # no inputs into middle boxes from either emissions or background
d2_mir[6] = 0
d2_mir[7] = 0
d2_mir[8] = 0
d2_mir[9] = 0
d2_mir[10] = (we3_mir[4,1] - beta(wa3_mir[4,1])*wa3_mir[4,1])*cB_no2  # background inputs into top boxes
d2_mir[11] = (we3_mir[4,2] - beta(wa3_mir[4,2])*wa3_mir[4,2])*cB_no2  # negative for beta because if true, the sign of W[4,x] would be negative
d2_mir[12] = (we3_mir[4,3] - beta(wa3_mir[4,3])*wa3_mir[4,3])*cB_no2  # yet as here being treated as an input, it should be positive, hence signs cancel out
d2_mir[13] = (we3_mir[4,4] - beta(wa3_mir[4,4])*wa3_mir[4,4])*cB_no2
d2_mir[14] = (we3_mir[4,5] - beta(wa3_mir[4,5])*wa3_mir[4,5])*cB_no2


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


# d4_mir = inputs for new conditions, PM2.5
d4_mir = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype = float)
d4_mir[0] = ez_tot_pm25_mir[1]/l_mirror[1] # averaging line emission source over area of 'ground' = (ug/m2/s)
d4_mir[1] = ez_tot_pm25_mir[2]/l_mirror[2]
d4_mir[2] = ez_tot_pm25_mir[3]/l_mirror[3]
d4_mir[3] = ez_tot_pm25_mir[4]/l_mirror[4]
d4_mir[4] = ez_tot_pm25_mir[5]/l_mirror[5]
d4_mir[5] = 0      # no inputs into middle boxes from either emissions or background
d4_mir[6] = 0
d4_mir[7] = 0
d4_mir[8] = 0
d4_mir[9] = 0
d4_mir[10] = (we3_mir[4,1] - beta(wa3_mir[4,1])*wa3_mir[4,1])*cB_pm25  # background inputs into top boxes
d4_mir[11] = (we3_mir[4,2] - beta(wa3_mir[4,2])*wa3_mir[4,2])*cB_pm25  # negative for beta because if true, the sign of W[4,x] would be negative
d4_mir[12] = (we3_mir[4,3] - beta(wa3_mir[4,3])*wa3_mir[4,3])*cB_pm25  # yet as here being treated as an input, it should be positive, hence signs cancel out
d4_mir[13] = (we3_mir[4,4] - beta(wa3_mir[4,4])*wa3_mir[4,4])*cB_pm25
d4_mir[14] = (we3_mir[4,5] - beta(wa3_mir[4,5])*wa3_mir[4,5])*cB_pm25

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


# d4_orig_par = inputs for new conditions, PM2.5
d4_orig_par = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype = float)
d4_orig_par[0] = ez_tot_pm25_orig[1]/l_original[1] # averaging line emission source over area of 'ground' = (ug/m2/s)
d4_orig_par[1] = ez_tot_pm25_orig[2]/l_original[2]
d4_orig_par[2] = ez_tot_pm25_orig[3]/l_original[3]
d4_orig_par[3] = ez_tot_pm25_orig[4]/l_original[4]
d4_orig_par[4] = ez_tot_pm25_orig[5]/l_original[5]
d4_orig_par[5] = 0      # no inputs into middle boxes from either emissions or background
d4_orig_par[6] = 0
d4_orig_par[7] = 0
d4_orig_par[8] = 0
d4_orig_par[9] = 0
d4_orig_par[10] = (we3_par[4,1] - beta(wa3_par[4,1])*wa3_par[4,1])*cB_pm25  # background inputs into top boxes
d4_orig_par[11] = (we3_par[4,2] - beta(wa3_par[4,2])*wa3_par[4,2])*cB_pm25  # negative for beta because if true, the sign of W[4,x] would be negative
d4_orig_par[12] = (we3_par[4,3] - beta(wa3_par[4,3])*wa3_par[4,3])*cB_pm25  # yet as here being treated as an input, it should be positive, hence signs cancel out
d4_orig_par[13] = (we3_par[4,4] - beta(wa3_par[4,4])*wa3_par[4,4])*cB_pm25
d4_orig_par[14] = (we3_par[4,5] - beta(wa3_par[4,5])*wa3_par[4,5])*cB_pm25

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


# d2_mir_par = inputs for new conditions, NO2
d2_mir_par = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype = float)
d2_mir_par[0] = ez_tot_no2_mir[1]/l_mirror[1] # averaging line emission source over area of 'ground' = (ug/m2/s)
d2_mir_par[1] = ez_tot_no2_mir[2]/l_mirror[2]
d2_mir_par[2] = ez_tot_no2_mir[3]/l_mirror[3]
d2_mir_par[3] = ez_tot_no2_mir[4]/l_mirror[4]
d2_mir_par[4] = ez_tot_no2_mir[5]/l_mirror[5]
d2_mir_par[5] = 0      # no inputs into middle boxes from either emissions or background
d2_mir_par[6] = 0
d2_mir_par[7] = 0
d2_mir_par[8] = 0
d2_mir_par[9] = 0
d2_mir_par[10] = (we4_par[4,1] - beta(wa4_par[4,1])*wa4_par[4,1])*cB_no2  # background inputs into top boxes
d2_mir_par[11] = (we4_par[4,2] - beta(wa4_par[4,2])*wa4_par[4,2])*cB_no2  # negative for beta because if true, the sign of W[4,x] would be negative
d2_mir_par[12] = (we4_par[4,3] - beta(wa4_par[4,3])*wa4_par[4,3])*cB_no2  # yet as here being treated as an input, it should be positive, hence signs cancel out
d2_mir_par[13] = (we4_par[4,4] - beta(wa4_par[4,4])*wa4_par[4,4])*cB_no2
d2_mir_par[14] = (we4_par[4,5] - beta(wa4_par[4,5])*wa4_par[4,5])*cB_no2


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


# d4_mir_par = inputs for new conditions, PM2.5
d4_mir_par = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype = float)
d4_mir_par[0] = ez_tot_pm25_mir[1]/l_mirror[1] # averaging line emission source over area of 'ground' = (ug/m2/s)
d4_mir_par[1] = ez_tot_pm25_mir[2]/l_mirror[2]
d4_mir_par[2] = ez_tot_pm25_mir[3]/l_mirror[3]
d4_mir_par[3] = ez_tot_pm25_mir[4]/l_mirror[4]
d4_mir_par[4] = ez_tot_pm25_mir[5]/l_mirror[5]
d4_mir_par[5] = 0      # no inputs into middle boxes from either emissions or background
d4_mir_par[6] = 0
d4_mir_par[7] = 0
d4_mir_par[8] = 0
d4_mir_par[9] = 0
d4_mir_par[10] = (we4_par[4,1] - beta(wa4_par[4,1])*wa4_par[4,1])*cB_pm25  # background inputs into top boxes
d4_mir_par[11] = (we4_par[4,2] - beta(wa4_par[4,2])*wa4_par[4,2])*cB_pm25  # negative for beta because if true, the sign of W[4,x] would be negative
d4_mir_par[12] = (we4_par[4,3] - beta(wa4_par[4,3])*wa4_par[4,3])*cB_pm25  # yet as here being treated as an input, it should be positive, hence signs cancel out
d4_mir_par[13] = (we4_par[4,4] - beta(wa4_par[4,4])*wa4_par[4,4])*cB_pm25
d4_mir_par[14] = (we4_par[4,5] - beta(wa4_par[4,5])*wa4_par[4,5])*cB_pm25

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
    # _________________________________________________________________________
    # WIND: LEFT TO RIGHT
    # NO2
    C1_orig = solve(a1_orig, d1_orig)
    C2_orig = solve(a2_orig, d2_orig)
    
    # PM2.5
    C3_orig = solve(a1_orig, d3_orig)
    C4_orig = solve(a2_orig, d4_orig)
    
    # calculate the percentage changes in concentrations before and after
    # NO2
    per_change_no2_orig = ((C2_orig - C1_orig)/C1_orig)*100
    #print("NO2 % Change (wind L->R)", per_change_no2_orig, sep = '\n')
    
    # PM2.5
    per_change_pm25_orig = ((C4_orig - C3_orig)/C3_orig)*100
    #print("PM 2.5 % Change (wind L->R)", per_change_pm25_orig, sep = '\n')
    
    # _________________________________________________________________________
    # WIND: RIGHT TO LEFT
    # NO2
    C1_mir = solve(a1_mir, d1_mir)
    C2_mir = solve(a2_mir, d2_mir)
    
    # PM2.5
    C3_mir = solve(a1_mir, d3_mir)
    C4_mir = solve(a2_mir, d4_mir)
    
    # calculate the percentage changes in concentrations before and after
    # NO2
    per_change_no2_mir = ((C2_mir - C1_mir)/C1_mir)*100
    
    # PM2.5
    per_change_pm25_mir = ((C4_mir - C3_mir)/C3_mir)*100
    
    # flip back so each box corresponds to the same space in the street
    per_change_no2_mir_flipped = flip_conc(C=per_change_no2_mir)
    per_change_pm25_mir_flipped = flip_conc(C=per_change_pm25_mir)
    l_cumu_flipped = flip_column(l=l_mirror, l_cumu=l_cumu_mirror)
    
    #print("NO2 % Change (wind R->L)", per_change_no2_mir_flipped, sep = "\n")
    #print("PM 2.5 % Change (wind R->L)",per_change_pm25_mir_flipped, sep = '\n')
    
    # _________________________________________________________________________
    # WIND: PARALLEL
    # NO2
    # dimensioning from left to right
    C1_par_1 = solve(a1_par_1, d1_orig_par)
    C2_par_1 = solve(a2_par_1, d2_orig_par)
    
    # dimensioning from right to left
    C1_par_2 = solve(a1_par_2, d1_mir_par)
    C2_par_2 = solve(a2_par_2, d2_mir_par)
    
    # % change of NO2 based on dispersion only (parallel wind) with dimensioning from wind left-to-right
    per_change_no2_par_1 = ((C2_par_1 - C1_par_1)/C1_par_1)*100
    # % change of NO2 based on dispersion only (parallel wind) with dimensioning from wind right-to-left
    per_change_no2_par_2 = ((C2_par_2 - C1_par_2)/C1_par_2)*100
    
    # PM2.5
    # dimensioning from left to right
    C3_par_1 = solve(a1_par_1, d3_orig_par)
    C4_par_1 = solve(a2_par_1, d4_orig_par)
    
    # dimensioning from right to left
    C3_par_2 = solve(a1_par_2, d3_mir_par)
    C4_par_2 = solve(a2_par_2, d4_mir_par)
    
    # % change of PM2.5 based on dispersion only (parallel wind) with dimensioning from wind left-to-right
    per_change_pm25_par_1 = ((C4_par_1 - C3_par_1)/C3_par_1)*100
    # % change of PM2.5 based on dispersion only (parallel wind) with dimensioning from wind right-to-left
    per_change_pm25_par_2 = ((C4_par_2 - C3_par_2)/C3_par_2)*100
    
    # flip back those calculated with dimensioning from right-to-left
    per_change_no2_par_flipped = flip_conc(C=per_change_no2_par_2)
    per_change_pm25_par_flipped = flip_conc(C=per_change_pm25_par_2)
    
    #print("NO2 % Change (Parallel wind, dimensioning: L->R)", per_change_no2_par_1, sep = '\n')
    #print("NO2 % Change (Parallel wind, dimensioning: R->L - flipped back)",per_change_no2_par_flipped, sep = '\n')
    #print("PM2.5 % Change (Parallel wind, dimensioning: L->R)",per_change_pm25_par_1, sep = '\n')
    #print("PM2.5 % Change (Parallel wind, dimensioning: R->L - flipped back)",per_change_pm25_par_flipped, sep = '\n')
    
    #__________________________________________________________________________
    # WEIGHT RESULTS BASED ON FREQUENCY OF CLIMATOLOGY
    # NO2 values
    weighted_row1_no2 = weighting_concs(l_cumu_LR=l_cumu_original, l_cumu_RL=l_cumu_flipped, 
                                C_LR=per_change_no2_orig[0:5], C_LR_par=per_change_no2_par_1[0:5], 
                                C_RL=per_change_no2_mir_flipped[0:5], C_RL_par=per_change_no2_par_flipped[0:5], 
                                LR_freq=LR_freq, LR_par_freq=LR_par_freq, RL_freq=RL_freq, RL_par_freq=RL_par_freq)
    
    l_cumu_total = weighted_row1_no2[1]
    weighted_row1_no2 = weighted_row1_no2[0]
    #print(weighted_row1_no2)
    
    weighted_row2_no2 = weighting_concs(l_cumu_LR=l_cumu_original, l_cumu_RL=l_cumu_flipped, 
                                C_LR=per_change_no2_orig[5:10], C_LR_par=per_change_no2_par_1[5:10], 
                                C_RL=per_change_no2_mir_flipped[5:10], C_RL_par=per_change_no2_par_flipped[5:10], 
                                LR_freq=LR_freq, LR_par_freq=LR_par_freq, RL_freq=RL_freq, RL_par_freq=RL_par_freq)
    weighted_row2_no2 = weighted_row2_no2[0]
    #print(weighted_row2_no2)
    
    weighted_row3_no2 = weighting_concs(l_cumu_LR=l_cumu_original, l_cumu_RL=l_cumu_flipped, 
                                C_LR=per_change_no2_orig[10:15], C_LR_par=per_change_no2_par_1[10:15], 
                                C_RL=per_change_no2_mir_flipped[10:15], C_RL_par=per_change_no2_par_flipped[10:15], 
                                LR_freq=LR_freq, LR_par_freq=LR_par_freq, RL_freq=RL_freq, RL_par_freq=RL_par_freq)
    weighted_row3_no2 = weighted_row3_no2[0]
    #print(weighted_row3_no2)
    
    per_change_no2 = np.concatenate((weighted_row1_no2, weighted_row2_no2, weighted_row3_no2))
    #print("Weighted NO2 % Change", per_change_no2, sep = '\n')
    
    # pm25 values
    weighted_row1_pm25 = weighting_concs(l_cumu_LR=l_cumu_original, l_cumu_RL=l_cumu_flipped, 
                                C_LR=per_change_pm25_orig[0:5], C_LR_par=per_change_pm25_par_1[0:5], 
                                C_RL=per_change_pm25_mir_flipped[0:5], C_RL_par=per_change_pm25_par_flipped[0:5], 
                                LR_freq=LR_freq, LR_par_freq=LR_par_freq, RL_freq=RL_freq, RL_par_freq=RL_par_freq)
    weighted_row1_pm25 = weighted_row1_pm25[0]
    #print(weighted_row1_pm25)
    
    weighted_row2_pm25 = weighting_concs(l_cumu_LR=l_cumu_original, l_cumu_RL=l_cumu_flipped, 
                                C_LR=per_change_pm25_orig[5:10], C_LR_par=per_change_pm25_par_1[5:10], 
                                C_RL=per_change_pm25_mir_flipped[5:10], C_RL_par=per_change_pm25_par_flipped[5:10], 
                                LR_freq=LR_freq, LR_par_freq=LR_par_freq, RL_freq=RL_freq, RL_par_freq=RL_par_freq)
    weighted_row2_pm25 = weighted_row2_pm25[0]
    #print(weighted_row2_pm25)
    
    weighted_row3_pm25 = weighting_concs(l_cumu_LR=l_cumu_original, l_cumu_RL=l_cumu_flipped, 
                                C_LR=per_change_pm25_orig[10:15], C_LR_par=per_change_pm25_par_1[10:15], 
                                C_RL=per_change_pm25_mir_flipped[10:15], C_RL_par=per_change_pm25_par_flipped[10:15], 
                                LR_freq=LR_freq, LR_par_freq=LR_par_freq, RL_freq=RL_freq, RL_par_freq=RL_par_freq)
    weighted_row3_pm25 = weighted_row3_pm25[0]
    #print(weighted_row3_pm25)
    
    per_change_pm25 = np.concatenate((weighted_row1_pm25, weighted_row2_pm25, weighted_row3_pm25))
    #print("Weighted PM2.5 % Change", per_change_pm25, sep = '\n')
    
    
    # format to correct layout for feeding back to UI
    street = {}
    street["columns"] = l_cumu_total.tolist()
    street["rows"] = h_cumu_original_list
    street["per_change_no2"] = per_change_no2.tolist()
    street["per_change_pm25"] = per_change_pm25.tolist()
    
    # send data back
    json_data = json.dumps(street)
    print(json_data, sep = '\n')
    


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
