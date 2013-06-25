# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 11:09:29 2013

@author: Kyle Weber
"""

import matplotlib.pyplot as plt

day = int(raw_input("Testing Day?"))
scopeNo = int(raw_input("Oscilloscope Number?"))
chan = int(raw_input("Channel Number?"))
shot = int(raw_input("Shot Number?"))

mydata = open("J:/SURE/sparkData_2013/lex/2013JanVisit/sparkData/%d_01_2013_osc%d/C%dosc%d-%05d.txt" % (day, scopeNo, chan, scopeNo, shot), "r")

mydata.close                     