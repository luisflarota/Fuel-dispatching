import numpy as np
import pandas as pd
import networkx as nx
import datetime
import random
## First case: Fuel01 == 00 ang merging
np.random.seed(5)
RAND = np.random.seed(5)
def returnstd(empty, full, ton):
    file = pd.read_csv("meanstd.csv")
    mean_empty = file.loc[1]['Empty(m)']
    std_empty = file.loc[2]['Empty(m)']
    mean_full = file.loc[1]['Full(m)']
    std_full = file.loc[2]['Full(m)']
    mean_ton = file.loc[1]['Tons']
    std_ton = file.loc[2]['Tons']

    norm_empty = (empty-mean_empty)/std_empty
    norm_full = (full-mean_full)/std_full
    norm_ton = (ton-mean_ton)/std_ton
    return np.array([[norm_empty, norm_full, norm_ton]])

def clean_full01_equal0(shiftFuelTrucks):
    first_fuel = []
    # Loop over equipment
    for uniq_fuel in np.unique(shiftFuelTrucks['Equipment']):
        dataUniqueFuel = shiftFuelTrucks[shiftFuelTrucks['Equipment'] == uniq_fuel].sort_values(['ShiftId','FuelStartTime']).reset_index(drop=True)
        # get Fuel01 columns from data for each equ. 
        array_fuel01 = np.array(dataUniqueFuel['Fuel01'])
        i=0
        if 0 in array_fuel01 or 1 in array_fuel01:
            indexes = np.argwhere(array_fuel01<=1).flatten()
            # help stop iteratios - through indexes that have 0/1 in Fuel01
            while i < len(indexes):
                ind = indexes[i]
                current_time = dataUniqueFuel.loc[ind]['FuelEndTime']
                # case when 2 consecutive indexes are zero
                if len(indexes)>1 and i+1<len(indexes) and ind+1 == indexes[i+1]:
                    # case before
                    if ind>0:
                        t_before_index = dataUniqueFuel.loc[ind-1]['FuelEndTime']            
                        # check before index
                        if np.abs(t_before_index - dataUniqueFuel.loc[ind]['FuelEndTime'])<5:
                            dataUniqueFuel.loc[ind+1,'FuelStartTime'] = dataUniqueFuel.loc[ind-1]['FuelStartTime']
                            dataUniqueFuel.loc[ind+1,'Fuel01'] = dataUniqueFuel.loc[ind,'Fuel01']
                            dataUniqueFuel.drop(index=[ind], inplace= True)
                            dataUniqueFuel.drop(index=[ind-1], inplace= True)
                    # check after index +1
                    elif ind+2<max(dataUniqueFuel.index):
                        t_after_index = dataUniqueFuel.loc[ind+2]['FuelEndTime']
                        if np.abs(t_after_index - dataUniqueFuel.loc[ind]['FuelEndTime'])<5:
                            dataUniqueFuel.loc[ind+2,'FuelStartTime'] = dataUniqueFuel.loc[ind]['FuelStartTime']
                            dataUniqueFuel.loc[ind+2,'Fuel01'] = dataUniqueFuel.loc[ind,'Fuel01']
                            #dataUniqueFuel.loc[ind+1,'Duration'] += dataUniqueFuel.loc[ind-1,'Duration']
                            dataUniqueFuel.drop(index=[ind], inplace= True)
                            dataUniqueFuel.drop(index=[ind+1], inplace= True)
                    elif ind+2 == len(indexes):
                        pass
                    i+=2
                elif ind>0:
                    time_before = dataUniqueFuel.loc[ind-1]['FuelEndTime']
                    if np.abs(current_time- time_before)<180:
                        dataUniqueFuel.loc[ind,'Fuel00'] = dataUniqueFuel.loc[ind]['Fuel00'] + dataUniqueFuel.loc[ind-1]['Fuel00']
                        dataUniqueFuel.loc[ind,'FuelStartTime'] = dataUniqueFuel.loc[ind-1]['FuelStartTime']
                        dataUniqueFuel.loc[ind,'Fuel01'] = dataUniqueFuel.loc[ind-1,'Fuel01']
                        dataUniqueFuel.loc[ind,'Duration'] += dataUniqueFuel.loc[ind-1,'Duration']
                        dataUniqueFuel.drop(index=[ind-1], inplace= True)
                        i+=1
                    else:
                        i+=1
                elif ind != max(dataUniqueFuel.index):
                    time_after = dataUniqueFuel.loc[ind+1]['FuelEndTime']
                    if np.abs(current_time- time_after)<180:
                        dataUniqueFuel.loc[ind+1,'Fuel00'] = dataUniqueFuel.loc[ind+1]['Fuel00'] + dataUniqueFuel.loc[ind]['Fuel00']
                        dataUniqueFuel.loc[ind+1,'FuelStartTime'] = dataUniqueFuel.loc[ind]['FuelStartTime']
                        dataUniqueFuel.loc[ind+1,'Fuel01'] = dataUniqueFuel.loc[ind,'Fuel01']
                        dataUniqueFuel.loc[ind+1,'Duration'] += dataUniqueFuel.loc[ind,'Duration']
                        dataUniqueFuel.drop(index=[ind], inplace= True)
                        i+=1
                    else:
                        i+=1
            first_fuel.append(dataUniqueFuel)
        else:
            first_fuel.append(dataUniqueFuel)
    first_fuel= pd.concat(first_fuel)
    return first_fuel.reset_index(drop=True)


def merge_dataframes(df1, df2, left_field , right_field, df2_tag, how_field = 'left'):
    print("len before merge:", df1.shape[0])
    columns_1 = list(df1.columns)
    columns_2 = list(df2.columns)
    df = df1.merge(df2, left_on=left_field, right_on=right_field, how= how_field)
    df.columns = columns_1 + [name + df2_tag for name in columns_2]
    print("len after merge:", df.shape[0])
    print("")
    return df


def format_datetime(str_datetime):
    yy, mm, dd = map(int,str_datetime.split(' ')[0].split('-'))
    HH, MM, SS = map(int,str_datetime.split(' ')[1].split('.')[0].split(':'))
    dt = datetime(yy,mm,dd,HH,MM,SS)
    return dt


def return_grafo(pitravel, pitloc):
    travel_columns = ["Id", "FieldLocstart", "FieldLocend", "FieldDist", "FieldClosed"]
    locs_columns = ["Id",  "FieldPit", "FieldRegion", "FieldUnit", "FieldXloc", "FieldYloc", "FieldZloc",'FieldId']
    df_travel = pitravel[travel_columns]

    # Cruzar locs con enums
    df_locs = pitloc[locs_columns]

    # Primer cruce con loc: start
    columns_travel = list(df_travel.columns)
    columns_locs = list(df_locs.columns)
    df_travel=df_travel.merge(df_locs, left_on="FieldLocstart", right_on="Id")
    df_travel.columns = columns_travel + [name + "_start" for name in columns_locs]
    columns_travel = list(df_travel.columns)

    df_travel = df_travel.merge(df_locs, left_on="FieldLocend", right_on="Id")
    df_travel.columns = columns_travel + [name + "_end" for name in columns_locs]
    travel_columns = list(df_travel.columns)
    df_travel[['FieldXloc_start','FieldYloc_start','FieldXloc_end', 'FieldYloc_end','FieldDist']]
    df_travel['DistEuc'] = ((df_travel['FieldXloc_start'] - df_travel['FieldXloc_end'])**2+
                            (df_travel['FieldYloc_start'] - df_travel['FieldYloc_end'])**2)**0.5
    #Nodos que se desean eliminar del gráfico
    nodos_filtrados = ['']

    # Identificar nodos únicos
    Nodos=pd.unique(list(df_travel["FieldId_start"])+list(df_travel["FieldId_end"]))
    Nodos = list(set(Nodos)-set(nodos_filtrados))

    # Construir diccionario con posiciones de nodos
    dict_locs = df_locs.set_index('FieldId').to_dict()
    Dpos = {nodo:{'pos':(dict_locs["FieldXloc"][nodo],dict_locs["FieldYloc"][nodo])} for nodo in Nodos}

    # Crear grafo
    Grafo=nx.DiGraph()
    Grafo.add_nodes_from(Nodos)

    # Llenar arcos
    for idx,row in df_travel.iterrows():
        if row["FieldId_start"] not in nodos_filtrados and row["FieldId_end"] not in nodos_filtrados:
            Grafo.add_edge(row["FieldId_start"], row["FieldId_end"], distance = row["FieldDist"])
    nx.set_node_attributes(Grafo,Dpos)
    return Grafo

def overlap_time(first_a, end_a, first_b, end_b):
    from collections import namedtuple
    Range = namedtuple('Range', ['start', 'end'])
    r1 = Range(start=first_a, end=end_a)
    r2 = Range(start=first_b, end=end_b)
    latest_start = max(r1.start, r2.start)
    earliest_end = min(r1.end, r2.end)
    delta = (earliest_end - latest_start).seconds + 1
    ov = round(max(0, delta)/60,1)
    return ov    

def convert_to_shift(time):
    """This will look for all trucks working in this shift"""
    year = str(time.year)[2:]
    month = str(time.month)
    day  = time.day
    if day<10:
        day = '0'+str(day)
    else:
        day = str(day)
    hour = time.hour
    shift = '002'
    if hour >= 8 and hour<20:
        shift = '001'
    shiftid = year+month+day+shift
    return int(shiftid)

def returnnameofnodeloads(nodeini, graph):
    split = nodeini.split('-')
    new_word = '-'.join(split[:-1])
    choice = [x for x in graph.nodes if new_word in x]
    if len(choice)>0:
        return choice[0]
    else:
        choice = [x for x in graph.nodes if split[0] in x]
        return choice[0]
def returnnameofnodedumps(nodeini, graph):
    split = nodeini.split('_')
    new_word = '_'.join(split[:-1])
    choice = [x for x in graph.nodes if new_word in x]
    if len(choice)>0:
        return choice[0]
    else:
        choice = [x for x in graph.nodes if split[0] in x]
        return choice[0]
    
# Ml model
# new_model = tf.keras.models.load_model(r"C:\Users\101114992\Documents\Research\05SDSMT_Masters\02Thesis\04DataAnalysis\03FuelDispatch\01FuelOther\0Data\1_MLData\model89_tail_1000e_32b.h5")

def get_last_time_fuel(datafuel, time_run, trucksSelected):
    """ This returns the last time truck fueled"""
    listTruckFtimeLiters = []
    for truck in trucksSelected:
        dataSelect = datafuel[(datafuel['Equipment'] == truck)&
                                (datafuel['EndTimestamp'] <= time_run)].sort_values('EndTimestamp').iloc[-1:]
        endTime = dataSelect['EndTimestamp'].values[0]
        if time_run - endTime >= pd.Timedelta(14,'h') and  time_run - endTime <= pd.Timedelta(24,'h'):
            listTruckFtimeLiters.append([truck, endTime,dataSelect['Fuel02'].values[0]])
    return listTruckFtimeLiters

def get_EFH_tonnage(truck, timefuel_liters, time_run, loaddumps):
    """ Returns sum of EFH from Load and Dumps and Tonnage"""
    time_fuel = timefuel_liters[0]
    # This does not count the last cycle (MIGHT DO - bypass for now)
    loaddumpsSelected = loaddumps[(loaddumps['Truck'] == truck)&
                                  (loaddumps['AssignTimestamp_x'] >= time_fuel)&
                                  (loaddumps['EmptyTimestamp'] <= time_run)]
    return loaddumpsSelected[['EFH_empty(m)', 'EFH_full(m)','Tonnage_y']].sum().values

def get_fuel_ratio(mlmodel, truck, timefuel_liters, time_run, loaddumps, cap=3800,ext=400):
    """ Returns liters fueled before + spent """
    liters = timefuel_liters[1] + ext
    EFH_tonnage = get_EFH_tonnage(
        truck, timefuel_liters, time_run, loaddumps
    )
    EFH_tonnageNorm = returnstd(
        EFH_tonnage[0],
        EFH_tonnage[1],
        EFH_tonnage[2]
    )
    fuelSpent = mlmodel.predict(EFH_tonnageNorm,verbose = 0)[0][0] 
    fuelRatio = round((liters - fuelSpent)*100/cap,1)
    return [liters, fuelSpent, fuelRatio, timefuel_liters[0]]


def return_fuel_ratio(datafuel, time_run, loaddumps,mlmodel, threshold = 20, cap=3000):
    """ This returns the last time truck fueled"""
    # get shiftid from the initial time
    shiftTimeRun = convert_to_shift(time_run)
    # filter loaddumps for the shift corresponding to time_run
    loaddumpsSelected = loaddumps[loaddumps['ShiftId_x'] == shiftTimeRun]
    # get trucks in loaddumps selected
    trucksSelected = np.unique(loaddumpsSelected['Truck'])
        # print(trucksSelected)
    trucksFuelTime = get_last_time_fuel(datafuel, time_run, trucksSelected)
    currentFuelTrucks = datafuel[(datafuel['ShiftId'] == shiftTimeRun)&
                                 (datafuel['StartTimestamp'] >= time_run)&
                                 (datafuel['StartTimestamp'] <= time_run+pd.Timedelta(4, 'h'))]
    # excluded if > than fuel ratio threshold
    truckFuelRatio = []
    excluded = []
    for data in trucksFuelTime:
        # mlmodel, truck, timefuel_liters, time_run, loaddumps, cap=3800
        fuelRatio = get_fuel_ratio(
            mlmodel, data[0], data[1:3], time_run,loaddumps, cap=cap
        )
        print(data[0], fuelRatio)
        if fuelRatio[2]< threshold and fuelRatio[2] > 0.1:
            truckFuelRatio.append([data[0]] + fuelRatio)
        else:
            excluded.append([data[0],fuelRatio])
    # matching expected and next
    
    truckFuelRatio = [x for x in truckFuelRatio if x[0] in currentFuelTrucks['Equipment'].values]
    currentFuelTrucks = currentFuelTrucks[
        currentFuelTrucks['Equipment'].isin(np.array(truckFuelRatio)[:,0])
        ]
    return truckFuelRatio, currentFuelTrucks

def return_dump_to_fuel_distance(dumplocation, Graph):
    if '_' in dumplocation:
        newnode = returnnameofnodedumps(dumplocation, Graph)
        distance = nx.shortest_path_length(Graph, source=newnode, target='PETROLERA SF', weight='distance')
        # empty, full,tonnage
    else:
        distance = nx.shortest_path_length(Graph, source=dumplocation, target='PETROLERA SF', weight='distance')
    return distance


def return_possible_dumptime(trucks_fratio, loaddumps, time_run, graph, mlmodel, cap=3800, speed=6, ran=RAND):
    """ This will compute the additional cycle for trucks and return the leftover fuel (l)"""
    shiftTimeRun = convert_to_shift(time_run)
    # filter loaddumps for the shift corresponding to time_run
    loaddumpsSelected = loaddumps[loaddumps['ShiftId_x'] == shiftTimeRun]
    dictTruckEmpty = {}
    for data in trucks_fratio:
        truck = data[0]
        # data for truck
        dataTruck = loaddumpsSelected[(loaddumpsSelected['Truck'] == truck)]
        # cycles during the time_run
        ldTruck = dataTruck[(dataTruck['AssignTimestamp_x'] <= time_run)&
                                    (dataTruck['EmptyTimestamp'] >= time_run)]
        # cycles after the time_run - should be 
        ldTruckAfter = dataTruck[
            (dataTruck['AssignTimestamp_x'] >=  time_run)
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
                distance = return_dump_to_fuel_distance(dumplocation, graph)
                timetoStation = round(distance/(speed*60), 1)
                # add speed but which oneeeee!!!
                # first cycle
                if i ==0: 
                    fuelSpent = mlmodel.predict(returnstd(distance,0,0),verbose = 0)[0][0]
                # second cycles and beyond
                else:
                    currentefhtons = dataqueried[['EFH_empty(m)','EFH_full(m)','Tonnage_y']].values
                    # sums up for following cycles 
                    emptyfueltones = [emptyfueltones[i] +  currentefhtons[i] for i in range(3)]
                    # This sums up the distance empty 
                    fuelSpent = mlmodel.predict(
                        returnstd(emptyfueltones[0]+distance,emptyfueltones[1],emptyfueltones[2]),verbose = 0
                    )[0][0]
                newfuelRatio = round((data[1] - data[2] - fuelSpent)*100/cap,1)
                # Does not compute if trucks run out of fuel with an extra cycle 
                if newfuelRatio >0:
                    arriveFuelStation = dataqueried['EmptyTimestamp'] + pd.Timedelta(timetoStation +np.random.uniform(1,100), 's') 
                    #dictTruckEmpty[truck].append([dataqueried['EmptyTimestamp'], dumplocation, newfuelRatio])
                    dictTruckEmpty[truck].append([arriveFuelStation, timetoStation, newfuelRatio])
        # I do not know what this does 
        else:
            firstRow = ldTruckAfter.iloc[0]
            #dictTruckEmpty[truck] = firstRow['AssignTimestamp_x'],firstRow['EmptyTimestamp']
        if len(dictTruckEmpty) ==0:
            raise 
    return dictTruckEmpty

# looking for a time window to ask for match factor - only uniques
def return_mfactor_optmodel(time_run, shiftLoadsDumps, truck, move = 50, ran=RAND):
    up_time = time_run + pd.Timedelta(move, 'm')
    dataSelectedTruck = shiftLoadsDumps[(shiftLoadsDumps['ArriveTimestamp_x'] >= time_run)&
                   (shiftLoadsDumps['ArriveTimestamp_x'] <= up_time)&
                    (shiftLoadsDumps['Truck'] == truck)]
    if dataSelectedTruck.shape[0]>0:
        shovels = np.unique(dataSelectedTruck['Excav_x'])
        #print(truck, shovels)
        for shovel in shovels:
            dataSelected = shiftLoadsDumps[(shiftLoadsDumps['ArriveTimestamp_x'] >= time_run)&
                       (shiftLoadsDumps['ArriveTimestamp_x'] <= up_time)&
                        (shiftLoadsDumps['Excav_x'] == shovel)&
                        (shiftLoadsDumps['Truck'] !=truck)] ## getting the truck out
            groups = dataSelected.groupby(['Excav_x','Truck']).apply(lambda df:pd.Series(dict(
                tcycle_div_tload = df['LoadingTime'].values[0]/((
                    df['EmptyTimestamp'] - df['ArriveTimestamp_x']
                ).values[0]/ np.timedelta64(1, 's')),
                tcycle = (
                    df['EmptyTimestamp'] - df['ArriveTimestamp_x']
                ).values[0]/ np.timedelta64(1, 's'),
                tload = df['LoadingTime'].values[0]))).reset_index()
            last = groups.groupby(['Excav_x']).apply(lambda df: pd.Series(dict(
                avg_matchfactor = df.shape[0]*np.mean(df['tcycle_div_tload']),
                weighted_matchfactor = df.shape[0]* np.sum(df['tload'])/np.sum(df['tcycle']))))
            #print(last.reset_index()['Excav_x'])
            m_factor = [shovel, round(last['avg_matchfactor'].values[0],1)]
    else:
        m_factor = ['PAB11',round(np.random.uniform(0.7, 0.95),1)]
    return m_factor
# Some use weighted, some of them avg of avg

def returnmfactor(fuelratios, shiftLoadsDumps):
    mfactorTrucks = {}
    for truck in fuelratios:
        infoTruck = fuelratios[truck]
        mfactorTrucks[truck] = list()
        #print(truck,infoTruck)
        for info in infoTruck:
            #print(truck, info)
            time = info[0]
            mfactorTrucks[truck].append(return_mfactor_optmodel(time, shiftLoadsDumps, truck))
    return mfactorTrucks

def converttodf(table, par=None):
    if par == 'fuel':
        columns = ["Truck", "ETA at fuel station (st.)",
                    "Time to st. (min)", "Fuel ratio at st. (%)"]
    elif par == 'mfactor':
        columns = ["Truck", "Next Shovel", "Match factor"] 
    df = pd.DataFrame([
            (k, *t) for k, v in table.items() for t in v
            ], columns=columns)
    return df

# Optimization side 

from datetime import datetime, timedelta
def datetime_range(start, end, delta): 
    current = start
    while current <= end:
        yield current
        current += delta
def ret_timestamp(fratio):
    listt = []
    length = 0
    # fuel
    newdict = {}
    # mf
    matchf = {}
    #TRUCK, TIME PERIOD
    for i,k in fratio.items():
        newdict[i]=dict()
        matchf[i]=dict()
        length += len(k)
        for lst in k:
            # round to min
            newt = lst[0].round('min')
            newdict[i][newt] = lst[2:][0]
            matchf[i][newt] = lst[2:][1]
            listt.append(newt)
    return (length, listt, newdict,matchf)

def createBigdict(timestamps, trucks, frationew, matchfb):
    min_t = min(timestamps)
    max_t = max(timestamps)
    # all minutes
    timerange = [pd.to_datetime(dt.strftime('%Y-%m-%d T%H:%M')) for dt in 
       datetime_range(min_t, max_t, 
       timedelta(minutes=1))]
    newdict = {}
    # mf
    matchf = {}
    for truck in trucks:
        if truck not in newdict:
            newdict[truck] = dict()
        if truck not in matchf:
            matchf[truck] = dict()
        for time in timerange:
            if time in frationew[truck].keys():
                newdict[truck][time] = frationew[truck][time]
                matchf[truck][time] = matchfb[truck][time]
            else:
                newdict[truck][time] = -100
                matchf[truck][time] = -100
    return (newdict, timerange, matchf)
def return_index_dict(newdict):
    ndict = {}
    sort = sorted(newdict)
    i = 0
    for dat in sort:
        ndict[i] = dat
        i+=1
    return ndict

def returndataopt(fuelratios, mfactorTrucks):
    fuelratiosnew = {}
    for i,k in fuelratios.items():
        datamf = mfactorTrucks[i]
        fuelratiosnew[i] = k
        for j in range(len(datamf)):
            fuelratiosnew[i][j].append(datamf[j][1])
    return fuelratiosnew
# pd.Series(pd.to_datetime(fratio['CDH39'][0][0])).dt.round('min')
# pd.to_datetime(fratio['CDH39'][0][0]).round('min')
