# -*- coding: utf-8 -*-
"""
Created on August 13, 2013

This file both defines functions and does a fair bit of processing and categorization.  It will take a while to load.  The goal is that the data frames defined by this file can be used for analysis.

This file assumes hitData/df_brant.pandas is a file containing a data frame of the type produced by findSpikes_makeAndSave.py.

@author: Brant
"""

import numpy as np
import pandas as pd
import networkx as nx

#############
# READ DATA # --> d0
#############

d0 = pd.read_pickle("hitData/df_brant.pandas")
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

# saturation threshold specific to detector, vectorized.
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
  """function for pandas groupby(...).apply to add an afterpulse flag."""
  # I had far more trouble on this function than you'd think,
  # but I'm too lazy to reduce it down to a form I could file as a bug report in Pandas.
  # the ...BS stuff above and the using the MultiIndex for the groupby below is an
  # attempt to work-around what I suspect are bugs (or at least unintelligible error messages).

  #print df.dayBS.values[0],df.shotBS.values[0], df.detBS.values[0],df.shape
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
  return np.logical_not(reduce(np.logical_or,badShotMasks))


d1 = d0[maskBadShots(d0)]

###################################################
# REMOVE EARLY (noise) AND LATE (LIGHT LEAK) HITS # --> d2
###################################################

d2 = d1[np.logical_and(d0.tStart > 0.3, d0.tStart<1.2)]

#####################################
# SLICE AND DICE, ADD POSITION INFO #
#####################################

def addPos(df,map):
  """
  Adds a 'pos' column to the given data frame according as given by
  the map passed in.  df['pos'] = map[df.det], essentially, but vectorized.
  LaBr1 and LaBr2 are assumed to have position NA.
  """
  map['LaBr1'] = np.nan
  map['LaBr2'] = np.nan

  vMap = np.vectorize(lambda det: float(map[det]))
  df['pos'] = vMap(df.det)

def centralAngle(opp,adj1,adj2):
  """For a triangle with sides opp, adj1, and adj2, returns the angle opposite side opp."""
  return np.arccos((opp**2-adj1**2-adj2**2)/(-2*adj1*adj2))
def rCyl(rHV,rGnd,gap):
  """Returns the radius of a cylinder """
  return rGnd*np.sin(centralAngle(rHV,gap,rGnd))

#################### TUESDAY

tue = d2[d2.day == 22]
cal1 = tue #all shots on tuesday were calibration with same geom.
# positioning: 110cm up, 84cm from gnd, 80cm from HV.
addPos(cal1,{'H2':1,'H1':1,'UB1':1,'UB2':1,'UB3':1,'UB4':1}) # 1 = dummy position, all same

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
# | H1/H3 | 120 |  116 |  106 |  110 |
addPos(rad,{'UB4':39.0,'UB3':59.0,'UB2':74.0,'UB1':93.0,'H1':110.0,'H2':110.0}) # cm
# NOTE: substitution of H2 for H3 in map...
# "H2" is whatever was connected to scope 2 channel 4

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
#| UB4 |  84 |  110 |  127 |
# detector separation 20.0...
rc = rCyl(84.0,107.0,107.0)
dAngle = centralAngle(20.0,rc,rc)*180/np.pi
addPos(azim1,{'H2':dAngle*0,'H1':dAngle*1,'UB1':dAngle*2,'UB2':dAngle*3,
              'UB3':dAngle*4,'UB4':dAngle*5}) # degrees of azimuth
# 19 degrees is what I get by doing the geometry rCyl(84,107,107.0) (gap length is 107cm)

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
# | UB4 |  88 |  111 |        129 |
rc = rCyl(84.0,107.0,106.5)
dAngle = centralAngle(50.0,rc,rc)*180/np.pi
addPos(azim2,{'H2':dAngle*0,'UB1':dAngle*1,'H1':dAngle*2,'UB2':dAngle*3,
              'UB3':dAngle*4,'UB4':dAngle*5}) # degrees of azimuth

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
# | UB4 |         97 |  88 |   81 |
rc = rCyl(92.0,107.0,80.3)
dAngle = centralAngle(50.0,rc,rc)*180/np.pi
addPos(azim3,{'H2':dAngle*0,'UB1':dAngle*1,'H1':dAngle*2,'UB2':dAngle*3,
              'UB3':dAngle*4,'UB4':dAngle*5}) # degrees of azimuth

pol = thu[np.logical_and(thu.shot>=200,thu.shot<300)]
# | det | vert floor | rHV | rgnd |
# |-----+------------+-----+------|
# | UB1 |        149 | 76  | 121  |
# | UB2 |        136 | 77  | 109  |
# | UB3 |        121 | 79  | 96   |
# | UB4 |        108 | 84  | 84   |
# | H1  |         94 | 86  | 70   |
# | H3  |         86 | 88  | 59   |
rgnd = np.array([121,109,96,84,70,59.0])
rhv = np.array([76,77,79,84,86,88.0])
th = centralAngle(rgnd,rhv,107.0)*180/np.pi
addPos(pol,{'UB1':th[0],'UB2':th[1],'UB3':th[2],'UB4':th[3],
            'H1':th[4],'H2':th[5]}) # degrees from HV-gnd axis

#################### FRIDAY

fri = d2[d2.day == 25]
# roughly the same as the first calibration shots.
# | det | vert floor | rgnd | rhv |
# |-----+------------+------+-----|
# | all | 112        | 88   | 80  |
addPos(fri,{'UB1':2,'UB2':2,'UB3':2,'UB4':2,'H1':2,'H2':2}) # dummy position again

cal2 = fri[np.logical_and(fri.shot>=0,fri.shot<50)]
att1 = fri[np.logical_and(fri.shot>=50,fri.shot<150)]
att2 = fri[np.logical_and(fri.shot>=150,fri.shot<250)]
att3 = fri[np.logical_and(fri.shot>=250,fri.shot<300)]

###########################
# HIT BY HIT CORRELATIONS
###########################

def detStats(df):
  """process a set of hits for a detector within a shot, return a data frame of statistics."""

  print df.dayBS.values[0],df.shotBS.values[0], df.detBS.values[0],df.shape

  if df.amp.count()>0:
    d = pd.DataFrame({'nHit':len(df.index),
      'ampMax':df.amp.max(),
      'ampSum':df.amp.sum(),
      'intMax':df.int.max(),
      'intSum':df.int.sum(),
      'satMax':df.satCt.max(),
      'satSum':df.satCt.sum(),
      'tMax':df.tMax.values[df.amp.argmax()],
      'tMaxA':df.tMax.min(),
      'tMaxB':df.tMax.max(),
      'pos':df.pos.values[0]},index=[1])
    if 'pairID' in df.columns:
      d['pairID'] = df.pairID.values[0]
    return d
  else:
    return pd.DataFrame()

def findCorrHits(df):
  """
  Returns a data frame with correlated hit pairs unstacked by a pairID.
  Unpaired hits are included, but with zeros for all other hit intensities.
  Each valid hit should appear exactly once in the output.
  """

  #print df.dayBS.values[0],df.shotBS.values[0], df.detBS.values[0],df.shape

  ii = np.arange(len(df.index))
  df = df.set_index([ii]) # reset index so I can re-order later.
  g = nx.Graph()
  g.add_nodes_from(ii) # add hit index to graph as nodes

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
  #d1 = d1[mask]; d2 = d2[mask] # done with d1?
  i1 = i1[mask]; i2 = i2[mask]

  # remove everything that isn't a close pair
  dtTol = 0.03
  pairMask = np.abs(t2-t1) < dtTol

  #t1 = t1[pairMask]; t2 = t2[pairMask] # done with t1 now, also
  #d1 = d1[pairMask]; d2 = d2[pairMask]
  i1 = i1[pairMask]; i2 = i2[pairMask]
  g.add_edges_from(zip(i1,i2))

  groups = nx.connected_components(g)

  def pairID(hitidx):
    return np.where([hitidx in grp for grp in groups])[0][0]
  vpairID = np.vectorize(pairID)

  df['pairID'] = vpairID(ii)

  df.set_index(['pairID','det'],inplace=True)

  df = df.groupby(level=[0,1]).apply(detStats)

  df = df.unstack(1) # unstack detector ID

  return df

def fillStatNAs(df):
  """replace NAs in relevant statistics data frame columns with zeros, i.e. leave time columns as NA."""
  for col in df.columns:
    if not (col[0] in ['tMax','tMaxA','tMaxB','pos']):
      df[col].fillna(0,inplace=True)

## # Calculate stats by hit group for everything.  This is redundant with the subsets calculated below,
## # so it should be commented out.
## # throw out less-significant hits and afterpulses, find stats by hit.
## dH = d2.ix[np.logical_or(d2.sig>15,np.logical_not(d2.APflag))].groupby(['day','shot']).apply(findCorrHits)
## fillStatNAs(dH)

def statsByHit(df):
  tmp = df.ix[np.logical_and(df.sig>15,np.logical_not(df.APflag))].groupby(['shot']).apply(findCorrHits)
  fillStatNAs(tmp)
  return tmp

cal1H = statsByHit(cal1);
tueH  = cal1H

radH = statsByHit(rad);
azim1H = statsByHit(azim1);
wedH = pd.concat([radH,azim1H])

azim2H = statsByHit(azim2);
azim3H = statsByHit(azim3);
polH = statsByHit(pol);
thuH = pd.concat([azim2H,azim3H,polH])

cal2H = statsByHit(cal2);
att1H = statsByHit(att1);
att2H = statsByHit(att2);
att3H = statsByHit(att3);
friH = pd.concat([cal2H,att1H,att2H,att3H])

##########################
# WHOLE SHOT CORRELATIONS
##########################

## # Calculate stats by shot for everything.  This is redundant with the subsets calculated below,
## # so it should be commented out.
## # less-significant hits and afterpulses are not removed here since they won't bias the shot
## # statistics as badly as they would bias individual hit group statistics.
## dS = d2.groupby(['day','shot','det']).apply(detStats).unstack(2)
## fillStatNAs(dS)

def statsByShot(df):
  tmp = df.groupby(['shot','det']).apply(detStats).unstack(1) # unstack detector id
  fillStatNAs(tmp)
  return tmp

cal1S = statsByShot(cal1);
tueS  = cal1S

radS = statsByShot(rad);
azim1S = statsByShot(azim1);
wedS = pd.concat([radS,azim1S])

azim2S = statsByShot(azim2);
azim3S = statsByShot(azim3);
polS = statsByShot(pol);
thuS = pd.concat([azim2S,azim3S,polS])

cal2S = statsByShot(cal2);
att1S = statsByShot(att1);
att2S = statsByShot(att2);
att3S = statsByShot(att3);
friS = pd.concat([cal2S,att1S,att2S,att3S])


#####################
# UTILITY FUNCTIONS #
#####################

satCtThr = 20
def noSat(df):
  return df.ix[df.satCt < satCtThr]

def noSatS(df,dets=None):
  if not dets:
    dets = np.unique([x[1] for x in df.columns])
  for det in dets:
    df = df.ix[df[('satMax',det)] < satCtThr]
  return df

