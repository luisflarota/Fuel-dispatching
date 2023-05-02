#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

import datetime
from calendar import monthrange
from textwrap import wrap

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from utils import *


class TransformData:
    """ 
    Load 4 main data files.
    """
    def __init__(self, fuelData="SfdShiftFueltrucksAllUpdated.csv", 
                shiftLoadsDumps="StdShiftLoadsDumpsFuel.csv",
                filePit_loc="StdPitloc.csv",
                filePit_travel="PITTravel.csv",
                mlmodel = "model89_tail_1000e_32b.h5"):
        # reading files
        self.fuelData = pd.read_csv(fuelData) 
        self.shiftLoadsDumps = pd.read_csv(shiftLoadsDumps)
        self.filePit_loc = pd.read_csv(filePit_loc)
        self.filePit_travel = pd.read_csv(filePit_travel)
        self.Graph = return_grafo(self.filePit_travel, self.filePit_loc)
        self.mlmodel = tf.keras.models.load_model(mlmodel)
        # Formatting data
        ## for shiftload dumps
        self.shiftLoadsDumps['ArriveTimestamp_x'] = pd.to_datetime(self.shiftLoadsDumps['ArriveTimestamp_x'])
        self.shiftLoadsDumps['AssignTimestamp_x'] = pd.to_datetime(self.shiftLoadsDumps['AssignTimestamp_x'])
        self.shiftLoadsDumps['EmptyTimestamp'] = pd.to_datetime(self.shiftLoadsDumps['EmptyTimestamp'])
        self.shiftLoadsDumps['Day'] = self.shiftLoadsDumps['ArriveTimestamp_x'].dt.to_period('d')
        ## for fueldata
        self.fuelData['StartTimestamp'] = pd.to_datetime(self.fuelData['StartTimestamp'])
        self.fuelData['EndTimestamp'] = pd.to_datetime(self.fuelData['EndTimestamp'])
        self.fuelData['Day'] = self.fuelData['StartTimestamp'].dt.to_period('d')

        