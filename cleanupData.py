# -*- coding: utf-8 -*-
"""
Created on August 13, 2013

This file both defines functions and does a fair bit of processing and categorization.  It will take a while to load.  The goal is that the data frames defined by this file can be used for analysis.

This file assumes df_brant.pandas is a file containing a data frame of the type produced by findSpikes_makeAndSave.py.

@author: Brant
"""

import numpy as np
import pandas as pd

#############
# READ DATA # --> d0
#############

d0 = pd.load("df_brant.pandas")
d0['hitID'] = np.arange(len(d0.index))
d0.set_index('hitID',inplace=True)

#######################################
# ADD SATURATION AND AFTERPULSE FLAGS # --> d0 (still)
#######################################

# satAmp[scope,chan] = threshold above which saturation is possible, in volts.
# these are determined by examining data from later in the week, to avoid issues
# with the gain fluctuations earlier in the week.
satAmp = {}
satAmp['LaBr1'] = 0.35 # satAmp[2,1] = 0.35
satAmp['LaBr2'] = 0.35 # satAmp[2,2] = 0.35
satAmp['H1'] = 0.6  # satAmp[2,3] = 0.6
satAmp['H2'] = 0.14 # satAmp[2,4] = 0.14
satAmp['UB1'] = 0.4  # satAmp[3,1] = 0.4
satAmp['UB2'] = 0.4  # satAmp[3,2] = 0.4
satAmp['UB3'] = 0.4  # satAmp[3,3] = 0.4
satAmp['UB4'] = 0.04 # satAmp[3,4] = 0.04

def satThr(det):
  return satAmp[det]
vsatThr = np.vectorize(satThr)
d0['satThr'] = vsatThr(d0.det)
d0['satPoss'] = d0.amp > vsatThr(d0.det)

d0.reset_index(inplace=True)
d0['dayBS'] = d0['day']  # the bullshit suffix is because I'm annoyed at pandas
d0['shotBS'] = d0['shot'] # I can't figure out how else to get index info into a groupby / apply function
d0['detBS'] = d0['det']
d0.set_index(['day','shot','det','hitID'],inplace=True)

def addAfterPulseFlag(df):
  print df.dayBS.values[0],df.shotBS.values[0], df.detBS.values[0],df.shape
  satAmpFrac = 0.1
  satCtThr = 20 # anything saturated for more than 20 is deemed to produce afterpulses
  df.sort(columns='iStart',inplace=True)
  apf = np.zeros(len(df.index),dtype=np.int64)
  apf[1:] = np.cumsum(np.where(np.logical_and(df.satPoss,df.satCt > satCtThr)[:-1],1,0))
  apf = np.logical_and(apf,df.amp < satAmpFrac*vsatThr(df.detBS))
  df['APflag'] = apf

  return df

d0 = d0.groupby(level=[0,1,2]).apply(addAfterPulseFlag)

d0.reset_index(inplace=True)
  

##########################
# REMOVE KNOWN-BAD SHOTS # --> d1
##########################

badShots = [(22,range(27)+[105]), #0-27 had random issues, plus gain (sat level) tweaks. 105 cabinet was open
    (23,[]),
    (24,[]),
    (25,[207])] # 207: cabinet was open

def maskBadShots(d):
  badShotMasks = [np.logical_and(d.day==day,np.in1d(d.shot,bad)) for (day,bad) in badShots]
  return np.logical_not(reduce(np.logical_and,badShotMasks))


d1 = d0[maskBadShots(d0)]

###################################################
# REMOVE EARLY (noise) AND LATE (LIGHT LEAK) HITS # --> d2
###################################################

d2 = d1[np.logical_and(d0.tStart > 0.3e-6, d0.tStart<1.2e-6)]

####################
# SLICE AND DICE
####################


#################### TUESDAY

tue = d2[d2.day == 22]
cal1 = tue #all shots on tuesday were calibration with same geom.
# positioning: 110cm up, 84cm from gnd, 80cm from HV.

#################### WEDNESDAY

wed = d2[d2.day == 23]

rad = wed[np.logical_and(wed.shot>=0,wed.shot<100)]
# detector positions (all numbers in cm)
# | det   | rHV | rgnd | vert | rgap |
# |-------+-----+------+------+------|
# | UB4   |  40 |  100 |  146 |   39 |
# | UB3   |  60 |  104 |  140 |   59 |
# | UB2   |  80 |  104 |  128 |   74 |
# | UB1   | 100 |  109 |  117 |   93 |

azim1 = wed[np.logical_and(wed.shot>=100,wed.shot<200)]
# detector positions (all numbers in cm)
# 20 cm spacing between detectors
#| det | rHV | rgnd | vert |
#|-----+-----+------+------+
#| H3  |  84 |  107 |  125 |
#| H1  |  84 |  106 |  125 |
#| UB1 |  84 |  106 |  125 |
#| UB2 |  84 |  106 |  125 |
#| UB3 |  84 |  108 |  126 |

#################### THURSDAY

thu = d2[d2.day == 24]

azim2 = thu[np.logical_and(thu.shot>=0,thu.shot<100)]
# broader azimuth than before
# 50 cm spacing between detectors
# | det | rHV | rgnd | vert floor |
# |-----+-----+------+------------|
# | H3  |  92 |  107 |        124 |
# | UB1 |  80 |  102 |        126 |
# | H1  |  79 |  105 |        131 |
# | UB2 |  80 |  106 |        130 |
# | UB3 |  84 |  108 |        128 |

azim3 = thu[np.logical_and(thu.shot>=100,thu.shot<200)]
# lowered down
# still 50 cm spacing between detectors
# | det | vert floor | rHV | rgnd |
# |-----+------------+-----+------|
# | H3  |         92 |  97 |   78 |
# | UB1 |         96 |  92 |   77 |
# | H1  |         98 |  90 |   80 |
# | UB2 |         99 |  90 |   80 |
# | UB3 |        101 |  90 |   86 |


pol = thu[np.logical_and(thu.shot>=200,thu.shot<300)]
# | det | vert floor | rHV | rgnd |
# |-----+------------+-----+------|
# | UB1 |        149 | 76  | 121  |
# | UB2 |        136 | 77  | 109  |
# | UB3 |        121 | 79  | 96   |
# | UB4 |        108 | 84  | 84   |
# | H1  |         94 | 86  | 70   |


#################### FRIDAY

fri = d2[d2.day == 25]
# roughly the same as the first calibration shots.
# | det | vert floor | rgnd | rhv |
# |-----+------------+------+-----|
# | all | 112        | 88   | 80  |

cal2 = fri[np.logical_and(fri.shot>=0,fri.shot<50)]

att1 = fri[np.logical_and(fri.shot>=50,fri.shot<150)]

att2 = fri[np.logical_and(fri.shot>=150,fri.shot<250)]

att3 = fri[np.logical_and(fri.shot>=250,fri.shot<300)]

###########################
# HIT BY HIT CORRELATIONS
###########################

def findCorrHits(df):
  df = df[df.sig > 15]

  ii = np.arange(len(df.index))
  df = df.set_index([ii]) # reset index so I can re-order later.

  dets = df.det.unique()
  dets.sort()
  df['detID'] = np.searchsorted(dets,df.det)

  ts = df.tMax.values
  ds = df.detID.values

  t1,t2 = np.meshgrid(ts,ts)
  d1,d2 = np.meshgrid(ds,ds)
  i1,i2 = np.meshgrid(ii,ii)
  t1 = t1.flatten()
  t2 = t2.flatten()
  d1 = d1.flatten()
  d2 = d2.flatten()
  i1 = i1.flatten()
  i2 = i2.flatten()

  # remove detector matches with itself, order-redundant matches.
  mask = (d1 < d2)
  t1 = t1[mask]; t2 = t2[mask]
  d1 = d1[mask]; d2 = d2[mask]
  i1 = i1[mask]; i2 = i2[mask]

  # remove everything that isn't a close pair
  dtTol = 0.03e-6
  pairMask = np.abs(t2-t1) < dtTol

  #t1 = t1[pairMask]; t2 = t2[pairMask]
  #d1 = d1[pairMask]; d2 = d2[pairMask]
  i1 = i1[pairMask]; i2 = i2[pairMask]

  pairID = np.arange(i1.shape[0])

  unpairedHits = df.ix[np.logical_not(np.in1d(ii,i1))]

  df1 = df.ix[i1];
  df2 = df.ix[i2];
  df1['pairID'] = pairID
  df2['pairID'] = pairID
  unpairedHits['pairID'] = np.arange(-1,-len(unpairedHits.index)-1,-1)

  pairs = pd.concat([df1,df2,unpairedHits])
  return pairs.set_index(['pairID','det']).unstack(1)

dH = d2.groupby(['day','shot']).apply(findCorrHits)

def statsByHit(df):
  return df.groupby(['shot']).apply(findCorrHits)

tueH  = statsByHit(tue)
cal1H = statsByHit(cal1)
wedH = statsByHit(wed)
radH = statsByHit(rad)
azim1H = statsByHit(azim1)
thuH = statsByHit(thu)
azim2H = statsByHit(azim2)
azim3H = statsByHit(azim3)
polH = statsByHit(pol)
friH = statsByHit(fri)
cal2H = statsByHit(cal2)
att1H = statsByHit(att1)
att2H = statsByHit(att2)
att3H = statsByHit(att3)

##########################
# WHOLE SHOT CORRELATIONS
##########################

def detStatsForShot(df):
  """process a set of hits for a shot, return a data frame of statistics.
  rejects hits that occur before 0.3us and after 1.4us."""

  # time limits set emperically to avoid light leaks, initial portion.
  #sel = np.logical_and(df.tStart > 0.3e-6, df.tStart < 1.2e-6)

  df = df.ix[sel]

  if df.amp.count()>0:
    return pd.DataFrame({'nHit':len(df.index),
      'ampMax':df.amp.max(),
      'ampSum':df.amp.sum(),
      'intMax':df.int.max(),
      'intSum':df.int.sum(),
      'satMax':df.satCt.max(),
      'satSum':df.satCt.sum(),
      'tmax':df.start.values[df.amp.argmax()]},index=[1])
  else:
    return pd.DataFrame()

dS = d2.groupby(['day','shot','det']).apply(detStatsForShot).unstack(2).fillna(0)

def statsByShot(df):
  return df.groupby(['shot','det']).apply(detStatsForShot).unstack(1).fillna(0)

tueS  = statsByShot(tue)
cal1S = statsByShot(cal1)
wedS = statsByShot(wed)
radS = statsByShot(rad)
azim1S = statsByShot(azim1)
thuS = statsByShot(thu)
azim2S = statsByShot(azim2)
azim3S = statsByShot(azim3)
polS = statsByShot(pol)
friS = statsByShot(fri)
cal2S = statsByShot(cal2)
att1S = statsByShot(att1)
att2S = statsByShot(att2)
att3S = statsByShot(att3)
