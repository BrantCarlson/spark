"""
Configuration local to the user's computer goes here, to avoid constant conflicts in other files.
"""

#for Brant's computer:
dataDir = "/home/brant/sure/sparkData/"
hitDataFilePath = "hitData/df_brant.pandas"
dataPlotDir = "dataPlots/"

#for Kyle's computer:
kdataDir = "C:/Users/Kyle Weber/Documents/Carthage/SURE/sparkData_2013/lex/2013JanVisit/sparkData/"

# Time calibration information
# signals arrive at the oscilloscope later than things actually happen...
# propagation delays, and all that.  Here's the delays that need to be subtracted,
# information from Pavlo.  Should be accurate for everything but UB1-4...
#| sch | reading | mult by | 1 equals | delay (us)                 |
#|-----+---------+---------+----------+----------------------------|
#|  11 | voltage |    0.08 | 1 MV     | 0.0519                     |
#|  12 | Ignd    |     0.8 | 500 A    | 0.0336                     |
#|  13 | IHV     |     9.8 | 500 A    | 0.220                      |
#|  14 | cam     |       1 | TTL      | 0.0463                     |
#|-----+---------+---------+----------+----------------------------|
#|  21 | LaBr1   |   13.90 | 1 MeV    | 0.053 + 30 ns (see below)  |
#|  22 | LaBr2   |     9.6 | 1 MeV    | 0.053 + 30 ns (see below)  |
#|  23 | H1      |     ??? | ???      | 0.0645 + 30 ns (see below) |
#|  24 | H3      |     ??? | ???      | 0.0505 + 30 ns (see below) |
#|-----+---------+---------+----------+----------------------------|
#|  31 | UB1     |         |          | X                          |
#|  32 | UB2     |         |          | X                          |
#|  33 | UB3     |         |          | X                          |
#|  34 | UB4     |         |          | X                          |
# X = 1.5 m of cable + [PMT electron transit time] + 3 m of fiber - 40 ns (scope delay).
# = 1.5m / (c*0.66) + 48 ns + 3m / c - 40 ns
# note - signs for scope delays below...  I'm assuming this is a delay in the trigger time, 
# in which case it makes things appear earlier than they should.  delays in the data should be subtracted from the scope time,
# but if the scope time itself is delayed, that will partially offset the delays in the data.

timeDelay = {}
timeDelay[1,1] = 0.0519
timeDelay[1,2] = 0.0336
timeDelay[1,3] = 0.220
timeDelay[1,4] = 0.0463

timeDelay[2,1] = 0.053 - 0.030
timeDelay[2,2] = 0.053 - 0.030
timeDelay[2,3] = 0.0645 - 0.030 - 0.01 # this 0.01 is a fudge factor I'm adding to line things up visually.
timeDelay[2,4] = 0.0505 - 0.030 - 0.01

timeDelay[3,1] = (1.5/(0.66*2.9979e8) + 48.0e-9 + 3.0/2.9979e8 - 40.0e-9)* 1.0e6 # formula above, converted to microseconds
timeDelay[3,2] = timeDelay[3,1]
timeDelay[3,3] = timeDelay[3,1]
timeDelay[3,4] = timeDelay[3,1]

ampMult = {}
ampMult[1,1] = 0.08*1.0e6 # HV in volts
ampMult[1,2] = 0.8*500 # Ignd in amps
ampMult[1,3] = 9.8*500 # IHV in amps
ampMult[1,4] = 1.0 # TTL

ampMult[2,1] = 1.0 # all PMTs left uncalibrated for now.
ampMult[2,2] = 1.0
ampMult[2,3] = 1.0
ampMult[2,4] = 1.0

ampMult[3,1] = 1.0
ampMult[3,2] = 1.0
ampMult[3,3] = 1.0
ampMult[3,4] = 1.0
