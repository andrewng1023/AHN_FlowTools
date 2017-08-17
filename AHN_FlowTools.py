import numpy as np
import pandas as pd
import math
from FlowCytometryTools import *

def gate(FCMeasurement,gatedata,minimum):
    try:
        FCMgate = FCMeasurement.data[FCMeasurement.data[gatedata] > minimum]
        return FCMgate
    except AttributeError:
        return np.nan

def cleandata(array,thresh):
    for i in range(len(array)):
        if array[i] < thresh:
            array[i] = np.nan
    return array

def FCdatastats(platesort,normalized,rows,cols,FITCthresh,SSCthresh):

    #Calculate the linear median, mean, and SD for each of the wells. Create two different Panels, one for FITC and one
    #for mCherry. In each Panel store a DataFrame containing the median, mean, SD, and CV

    empty = pd.DataFrame(index = rows, columns = cols)

    FITCstats = pd.Panel({'raw':empty, 'med':empty, 'avg':empty, 'sd':empty, 'cv':empty})
    mCherrystats = pd.Panel({'raw':empty, 'med':empty, 'avg':empty, 'sd':empty,'cv':empty})

    for row in rows:
        for col in cols:

            try:
                FCM = gate(platesort.loc[row,col],'FITC-H',FITCthresh)

                FCM2 = FCM[FCM['SSC-H'] > SSCthresh]
            except TypeError:
                continue

            try:
                if normalized == 1:
                    FITC = FCM2['FITC-H']/FCM2['SSC-H']
                    mCherry = FCM2['mCherry-H']/FCM2['SSC-H']

                elif normalized == 0:
                    FITC = FCM2['FITC-H']
                    mCherry = FCM2['mCherry-H']

                FITCstats.raw.set_value(row, col, FITC)
                FITCstats.med.set_value(row,col,FITC.median(axis=0))
                FITCstats.avg.set_value(row,col,FITC.mean(axis=0))
                FITCstats.sd.set_value(row,col,FITC.std(axis=0))
                FITCstats.cv.set_value(row,col,FITCstats.avg.loc[row,col]/FITCstats.sd.loc[row,col])

                mCherrystats.raw.set_value(row, col, mCherry)
                mCherrystats.med.set_value(row,col,mCherry.median(axis=0))
                mCherrystats.avg.set_value(row,col,mCherry.mean(axis=0))
                mCherrystats.sd.set_value(row,col,mCherry.std(axis=0))
                mCherrystats.cv.set_value(row,col,mCherrystats.avg.loc[row,col]/mCherrystats.sd.loc[row,col])

            except (AttributeError, TypeError):
                continue

    return [FITCstats, mCherrystats]


def splitPlate(file,wells=None,smooth=None,smooththresh=None,diffthresh=None,eventthresh=None):

    if wells is None:
        wells = 96
    if smooth is None:
        smooth = 50
    if smooththresh is None:
        smooththresh = 50
    if diffthresh is None:
        diffthresh = 50
    if eventthresh is None:
        eventthresh = 100

    wholeFCS = FCMeasurement(ID = 'WholeTC', datafile = file) #read the file
    tdiff = wholeFCS.data.Time.diff() #Calculate the difference in time between events, proportional to event rate
    smoothtdiff = tdiff.rolling(window=smooth,center=True).mean() #smooth over 50 events

    downtimes = np.where(smoothtdiff > smooththresh) # try a threshold of 50

    testpoints = np.where(np.diff(downtimes) > diffthresh) #try a threshold of 50

    breakpoints = list()
    for tp in testpoints[1]:
        breakpoints.append(downtimes[0][tp])

    breakpointdiffs = wholeFCS.data.Time.loc[breakpoints].diff()/100 #find time between each of the breakpoints, in seconds

    candidate_endinds = breakpointdiffs[wells*10 < breakpointdiffs] < wells*15 #find the breakpoint diffs that are longer than 10s per well and less than 15s per well

    candidate_ends = list()
    for ind in candidate_endinds.index:
        candidate_ends.append(np.where(breakpointdiffs.index == ind)[0][0])

    candidate_starts = [x-1 for x in candidate_ends]

    candidate_starts = np.asarray(candidate_starts)
    candidate_ends = np.asarray(candidate_ends)

    # Now if the number of events between the start and end is < some # of events per well then reject those points
    passcandidates = breakpointdiffs.iloc[candidate_ends].index - breakpointdiffs.iloc[candidate_starts].index > wells * eventthresh

    candidate_starts = candidate_starts[passcandidates]
    candidate_ends = candidate_ends[passcandidates]

    # Look at what the starting and stopping events are, as well as the times

    start_events = breakpointdiffs.index[candidate_starts]
    end_events = breakpointdiffs.index[candidate_ends]

    pDict = {('Plate '+ str(idx)): wholeFCS.data.loc[start:end] for (idx,start,end) in zip(range(len(start_events)),start_events,end_events)}

    # Now split the plate into individual wells
    rows = ['A','B','C','D','E','F','G','H']
    cols = ['01','02','03','04','05','06','07','08','09','10','11','12']
    plateraw = pd.DataFrame(index = rows, columns = cols)

    pDict_wells = pd.Panel({plate: plateraw for plate in pDict})
    verification = {plate: np.empty([wells,2]) for plate in pDict} #print start and stop times for wells to compare to Kieran's script

    for plate in pDict:
        bounds = np.empty([wells,2])
        finebounds = np.empty([wells,2])
        welltime = 1062 #THIS IS VERY IMPORTANT

        for tt in range(wells):
            bounds[tt] = [pDict[plate].Time.iloc[0] + tt*welltime, pDict[plate].Time.iloc[0] + (tt+1)*welltime]
            tenths = np.linspace(bounds[tt,0],bounds[tt,1],12)
            finebounds[tt] = [tenths[2],tenths[9]]

        for ii in range(8):
            for jj in range(12):
                pDict_wells.set_value(plate,rows[ii],cols[jj],pDict[plate][pDict[plate].Time.between(finebounds[ii*12+jj,0],finebounds[ii*12+jj,1],inclusive=True)]) #Take all the events between the bounds of each of the wells
        verification[plate] = finebounds

    return pDict_wells, verification
