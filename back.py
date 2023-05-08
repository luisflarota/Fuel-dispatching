#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
import pandas as pd
import tensorflow as tf

from utils import *


class TransformData(object):
    """ 
    Load 4 main data files.
    """
    def __init__(self, time_run, fuelData="SfdShiftFueltrucksAllUpdatedFin.csv", 
                shiftLoadsDumps="StdShiftLoadsDumpsFuel.csv",
                filePit_loc="StdPitloc.csv",
                filePit_travel="PITTravel.csv",
                mlmodel = "model89_tail_1000e_32b.h5",
                threshold = 20):
        # reading files
        self.fuelData = pd.read_csv(fuelData) 
        self.shiftLoadsDumps = pd.read_csv(shiftLoadsDumps)
        self.filePit_loc = pd.read_csv(filePit_loc)
        self.filePit_travel = pd.read_csv(filePit_travel)
        self.Graph = return_grafo(self.filePit_travel, self.filePit_loc)
        self.mlmodel = tf.keras.models.load_model(mlmodel)
        # Formatting data
        ## for shiftload dumps
        self.shiftLoadsDumps['ArriveTimestamp_x'] = pd.to_datetime(self.shiftLoadsDumps[
            'ArriveTimestamp_x'
            ])
        self.shiftLoadsDumps['AssignTimestamp_x'] = pd.to_datetime(self.shiftLoadsDumps[
            'AssignTimestamp_x'
            ])
        self.shiftLoadsDumps['EmptyTimestamp'] = pd.to_datetime(self.shiftLoadsDumps[
            'EmptyTimestamp'
            ])
        self.shiftLoadsDumps['Day'] = self.shiftLoadsDumps[
            'ArriveTimestamp_x'
            ].dt.to_period('d')
        ## for fueldata
        self.fuelData['StartTimestamp'] = pd.to_datetime(self.fuelData['StartTimestamp'])
        self.fuelData['EndTimestamp'] = pd.to_datetime(self.fuelData['EndTimestamp'])
        self.fuelData['Day'] = self.fuelData['StartTimestamp'].dt.to_period('d')
        self.threshold = threshold
        self.cap = 3000
        self.rand = np.random.seed(5)
        self.speed = 6
        self.time_run = time_run

    def fuelratio(self):
        """ This returns the last time truck fueled"""
        # get shiftid from the initial time
        shiftTimeRun = convert_to_shift(self.time_run)
        # filter self.shiftLoadsDumps for the shift corresponding to time_run
        self.loaddumpsSelected = self.shiftLoadsDumps[
            self.shiftLoadsDumps['ShiftId_x'] == shiftTimeRun
            ]
        # get trucks in self.shiftLoadsDumps selected
        trucksSelected = np.unique(self.loaddumpsSelected['Truck'])
            # print(trucksSelected)
        trucksFuelTime = get_last_time_fuel(self.fuelData, self.time_run, trucksSelected)
        self.currentFuelTrucks = self.fuelData[(self.fuelData['ShiftId'] == shiftTimeRun)&
                                    (self.fuelData['StartTimestamp'] >= self.time_run)&
                                    (self.fuelData['StartTimestamp'] <= self.time_run+pd.Timedelta(4, 'h'))]
        # excluded if > than fuel ratio threshold
        self.truckFuelRatio = []
        excluded = []
        for data in trucksFuelTime:
            fuelRatio = get_fuel_ratio(
                self.mlmodel, data[0], data[1:3],
                  self.time_run,self.shiftLoadsDumps, cap=self.cap
            )
            if fuelRatio[2]< self.threshold and fuelRatio[2] > 0.2:
                self.truckFuelRatio.append([data[0]] + fuelRatio)
            else:
                excluded.append([data[0],fuelRatio])
        self.truckFuelRatio = [
            x for x in self.truckFuelRatio if x[0] in self.currentFuelTrucks['Equipment'].values
            ]
        self.currentFuelTrucks = self.currentFuelTrucks[
            self.currentFuelTrucks['Equipment'].isin(np.array(self.truckFuelRatio)[:,0])
            ]
        self.addmfactor()
        self.fratio = self.return_possible_dumptime()
        self.mfactor = self.returnmfactor() 
        self.currentFuelTrucks = self.currentFuelTrucks[
            self.currentFuelTrucks['Equipment'].isin(self.fratio.keys())
            ]
        return self.fratio, self.mfactor, self.currentFuelTrucks

    def addmfactor(self):
        self.currentFuelTrucks['MatchFactor'] = 0.3
        for index, row in self.currentFuelTrucks.iterrows():
            truck = row['Equipment']
            time = row['StartTimestamp']
            mfactor = return_mfactor_current(
                time, self.shiftLoadsDumps, truck, ran=self.rand
                )
            self.currentFuelTrucks.loc[index, 'MatchFactor'] = mfactor
    def returnmfactor(self):
        mfactorTrucks = {}
        for truck in self.fratio:
            infoTruck = self.fratio[truck]
            mfactorTrucks[truck] = list()
            #print(truck,infoTruck)
            for info in infoTruck:
                #print(truck, info)
                time = info[0]
                mfactorTrucks[truck].append(return_mfactor_optmodel(time, self.shiftLoadsDumps, truck, ran=self.rand))
        return mfactorTrucks

    def return_possible_dumptime(self):
        rand = self.rand
        # filter self.shiftLoadsDumps for the shift corresponding to time_run
        dictTruckEmpty = {}
        for data in self.truckFuelRatio:
            truck = data[0]
            # data for truck
            dataTruck = self.loaddumpsSelected[(self.loaddumpsSelected['Truck'] == truck)]
            # cycles during the time_run
            ldTruck = dataTruck[(dataTruck['AssignTimestamp_x'] <= self.time_run)&
                                        (dataTruck['EmptyTimestamp'] >= self.time_run)]
            # cycles after the time_run - should be 
            ldTruckAfter = dataTruck[
                (dataTruck['AssignTimestamp_x'] >=  self.time_run)
            ].sort_values('AssignTimestamp_x')
            # If in between cycles (TODO: We still need to compute if one more cycle)
            if ldTruck.shape[0]>0:
                #print('yes'*10)
                emptyTime = ldTruck['EmptyTimestamp'].values[0]
                dataleft = dataTruck[(dataTruck['EmptyTimestamp'] >= emptyTime)]
                # Should be for load next: datanext = dataTruck[(dataTruck['EmptyTimestamp'] > emptyTime)]
                dictTruckEmpty[truck] = list()
                #empty_fuel_tones: This will add efh empty, fuel, and tons continuously
                emptyfueltones = [0,0,0]
                # loop over data for trucks with >1 cycle and in-between cycles
                for i in range(dataleft.shape[0]):
                    dataqueried = dataleft.iloc[i]
                    dumplocation = dataqueried['DumpLocation']
                    # First one should go to fuel
                    distance = return_dump_to_fuel_distance(dumplocation, self.Graph)
                    timetoStation = distance/(self.speed)
                    timetoStation += np.random.uniform(1,100)
                    timetoStation = round(timetoStation/60, 1)
                    # add speed but which oneeeee!!!
                    # first cycle
                    if i ==0: 
                        fuelSpent = self.mlmodel.predict(
                            returnstd(distance,0,0
                                      ),verbose = 0)[0][0]
                    # second cycles and beyond
                    else:
                        currentefhtons = dataqueried[
                            ['EFH_empty(m)','EFH_full(m)','Tonnage_y']
                            ].values
                        # sums up for following cycles 
                        emptyfueltones = [
                            emptyfueltones[i] +  currentefhtons[i] for i in range(3)
                            ]
                        # This sums up the distance empty 
                        fuelSpent = self.mlmodel.predict(
                            returnstd(
                            emptyfueltones[0]+distance,emptyfueltones[1],emptyfueltones[2]
                            ),verbose = 0
                        )[0][0]
                    newfuelRatio = round((data[1] - data[2] - fuelSpent)*100/3800,1)
                    # Does not compute if trucks run out of fuel with an extra cycle 
                    if newfuelRatio >0:
                        arriveFuelStation = dataqueried[
                            'EmptyTimestamp'
                            ] + pd.Timedelta(timetoStation, 'm') 
                        #dictTruckEmpty[truck].append([dataqueried['EmptyTimestamp'], dumplocation, newfuelRatio])
                        dictTruckEmpty[truck].append([
                            arriveFuelStation, timetoStation, newfuelRatio
                            ])
            if len(dictTruckEmpty) ==0:
                raise 
        return dictTruckEmpty
