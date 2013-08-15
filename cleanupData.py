# -*- coding: utf-8 -*-
"""
Created on August 13, 2013

@author: Brant
"""

import numpy as np
import pandas as pd
import toolbox as tb

badShots = [(22,range(27)+[105]), #0-27 had random issues, plus gain (sat level) tweaks. 105 cabinet was open
    (23,[]),
    (24,[]),
    (25,[207])] # 207: cabinet was open

def maskBadShots(d):
  badShotMasks = [np.logical_and(d.day==day,np.in1d(d.shot,bad)) for (day,bad) in badShots]
  return np.logical_not(reduce(np.logical_and,badShotMasks))


dRaw = pd.load("df_brant.pandas")

d = dRaw[maskBadShots(dRaw)]

#######################################
# ADD SATURATION AND AFTERPULSE FLAGS #
#######################################

# satAmp[scope,chan] = threshold above which saturation is possible, in volts.
# these are determined by examining data from later in the week, to avoid issues
# with the gain fluctuations earlier in the week.
satAmp = {}
satAmp[2,1] = 0.35
satAmp[2,2] = 0.35
satAmp[2,3] = 0.6
satAmp[2,4] = 0.14
satAmp[3,1] = 0.4
satAmp[3,2] = 0.4
satAmp[3,3] = 0.4
satAmp[3,4] = 0.04

def satThr(sc,ch,amp):
  return amp > satAmp[sc,ch]
vsatThr = np.vectorize(satThr)
d['satPoss'] = vsatThr(d.scope,d.chan,d.amp)

def addAfterPulseFlag(df):
  satAmpFrac = 0.1
  satCtThr = 20 # anything saturated for more than 20 is deemed to produce afterpulses
  df.sort(columns='iStart',inplace=True)
  apf = np.zeros(len(df.index),dtype=np.int64)
  apf[1:] = np.cumsum(np.where(np.logical_and(df.satPoss,df.satCt > satCtThr)[:-1],1,0))
  df['APflag'] = apf
  return df

d = d.groupby(['day','shot','det']).apply(addAfterPulseFlag)
  
# 
# 
# ####################
# # SLICE AND DICE
# ####################
# 
# 
# #################### TUESDAY
# 
# tue = d[d.day == 22]
# cal1 = tue #all shots on tuesday were calibration with same geom.
# # positioning: 110cm up, 84cm from gnd, 80cm from HV.
# 
# #################### WEDNESDAY
# 
# wed = d[d.day == 23]
# 
# rad = wed[np.logical_and(wed.shot>=0,wed.shot<100)]
# # detector positions (all numbers in cm)
# # | det   | rHV | rgnd | vert | rgap |
# # |-------+-----+------+------+------|
# # | UB4   |  40 |  100 |  146 |   39 |
# # | UB3   |  60 |  104 |  140 |   59 |
# # | UB2   |  80 |  104 |  128 |   74 |
# # | UB1   | 100 |  109 |  117 |   93 |
# 
# azim1 = wed[np.logical_and(wed.shot>=100,wed.shot<200)]
# # detector positions (all numbers in cm)
# # 20 cm spacing between detectors
# #| det | rHV | rgnd | vert |
# #|-----+-----+------+------+
# #| H3  |  84 |  107 |  125 |
# #| H1  |  84 |  106 |  125 |
# #| UB1 |  84 |  106 |  125 |
# #| UB2 |  84 |  106 |  125 |
# #| UB3 |  84 |  108 |  126 |
# 
# #################### THURSDAY
# 
# thu = d[d.day == 24]
# 
# azim2 = thu[np.logical_and(thu.shot>=0,thu.shot<100)]
# # broader azimuth than before
# # 50 cm spacing between detectors
# # | det | rHV | rgnd | vert floor |
# # |-----+-----+------+------------|
# # | H3  |  92 |  107 |        124 |
# # | UB1 |  80 |  102 |        126 |
# # | H1  |  79 |  105 |        131 |
# # | UB2 |  80 |  106 |        130 |
# # | UB3 |  84 |  108 |        128 |
# 
# azim3 = thu[np.logical_and(thu.shot>=100,thu.shot<200)]
# # lowered down
# # still 50 cm spacing between detectors
# # | det | vert floor | rHV | rgnd |
# # |-----+------------+-----+------|
# # | H3  |         92 |  97 |   78 |
# # | UB1 |         96 |  92 |   77 |
# # | H1  |         98 |  90 |   80 |
# # | UB2 |         99 |  90 |   80 |
# # | UB3 |        101 |  90 |   86 |
# 
# 
# pol = thu[np.logical_and(thu.shot>=200,thu.shot<300)]
# # | det | vert floor | rHV | rgnd |
# # |-----+------------+-----+------|
# # | UB1 |        149 | 76  | 121  |
# # | UB2 |        136 | 77  | 109  |
# # | UB3 |        121 | 79  | 96   |
# # | UB4 |        108 | 84  | 84   |
# # | H1  |         94 | 86  | 70   |
# 
# 
# #################### FRIDAY
# 
# fri = d[d.day == 25]
# # roughly the same as the first calibration shots.
# # | det | vert floor | rgnd | rhv |
# # |-----+------------+------+-----|
# # | all | 112        | 88   | 80  |
# 
# cal2 = fri[np.logical_and(fri.shot>=0,fri.shot<50)]
# 
# att1 = fri[np.logical_and(fri.shot>=50,fri.shot<150)]
# 
# att2 = fri[np.logical_and(fri.shot>=150,fri.shot<250)]
# 
# att3 = fri[np.logical_and(fri.shot>=250,fri.shot<300)]
# 
# ###########################
# # HIT BY HIT CORRELATIONS
# ###########################
# 
# dH = d.groupby(['day','shot']).apply(tb.findCorrHits)
# 
# def statsByHit(df):
#   return df.groupby(['shot']).apply(tb.findCorrHits)
# 
# tueH  = statsByHit(tue)
# cal1H = statsByHit(cal1)
# wedH = statsByHit(wed)
# radH = statsByHit(rad)
# azim1H = statsByHit(azim1)
# thuH = statsByHit(thu)
# azim2H = statsByHit(azim2)
# azim3H = statsByHit(azim3)
# polH = statsByHit(pol)
# friH = statsByHit(fri)
# cal2H = statsByHit(cal2)
# att1H = statsByHit(att1)
# att2H = statsByHit(att2)
# att3H = statsByHit(att3)
# 
# ##########################
# # WHOLE SHOT CORRELATIONS
# ##########################
# 
# dS = d.groupby(['day','shot','det']).apply(tb.detStatsForShot).unstack(2).fillna(0)
# 
# def statsByShot(df):
#   return df.groupby(['shot','det']).apply(tb.detStatsForShot).unstack(1).fillna(0)
# 
# tueS  = statsByShot(tue)
# cal1S = statsByShot(cal1)
# wedS = statsByShot(wed)
# radS = statsByShot(rad)
# azim1S = statsByShot(azim1)
# thuS = statsByShot(thu)
# azim2S = statsByShot(azim2)
# azim3S = statsByShot(azim3)
# polS = statsByShot(pol)
# friS = statsByShot(fri)
# cal2S = statsByShot(cal2)
# att1S = statsByShot(att1)
# att2S = statsByShot(att2)
# att3S = statsByShot(att3)
