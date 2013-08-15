import findHits as fh
import os
import time

#os.system('ipcluster start -n 4&')

#time.sleep(10) # wait for cluster to start up

x = fh.readAndProcessAllShots()
x.save('df_brant.pandas')

#os.system('ipcluster stop')
