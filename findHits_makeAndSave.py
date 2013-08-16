"""
This is a script to find hits in all shots (see findHits.py) and save the results in a file.
"""

import findHits as fh
import os
import time
import conf

x = fh.readAndProcessAllShots()
x.to_pickle(conf.hitDataFilePath)
