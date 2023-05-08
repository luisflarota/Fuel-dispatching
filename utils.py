from datetime import datetime, timedelta

import networkx as nx
import numpy as np
import pandas as pd

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



def returntableparm(fuelratio, mfactor):
    ratiotab = converttodf(fuelratio, par='fuel')
    mfactortab = converttodf(mfactor, par='mfactor')
    ratiotab['Next Shovel']= mfactortab['Next Shovel']
    ratiotab['Match factor']= mfactortab['Match factor']
    ratiotab["ETA at fuel station (st.)"] = ratiotab["ETA at fuel station (st.)"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return ratiotab

def return_dump_to_fuel_distance(dumplocation, Graph):
    if '_' in dumplocation:
        newnode = returnnameofnodedumps(dumplocation, Graph)
        distance = nx.shortest_path_length(Graph, source=newnode, target='PETROLERA SF', weight='distance')
        # empty, full,tonnage
    else:
        distance = nx.shortest_path_length(Graph, source=dumplocation, target='PETROLERA SF', weight='distance')
    return distance

# looking for a time window to ask for match factor - only uniques
def substractdates(current, new):
    current = current.sort_values('Equipment')
    new = new.sort_values('Truck')
    current['StartTimestamp'] = pd.to_datetime(current['StartTimestamp'])
    new["ETA at fuel station (st.)"] = pd.to_datetime(new["ETA at fuel station (st.)"])
    diff = []
    average = []
    for x,y in zip(current['StartTimestamp'],new["ETA at fuel station (st.)"]):
        if x >= y:
            diff.append('+'+str(x-y))
            average.append((x-y).seconds)
        else:
            diff.append('-'+str(y-x))
            average.append((y-x).seconds * -1)
    current['Difference'] = diff
    return current[['Equipment', 'StartTimestamp', 'MatchFactor','Difference']], np.mean(average)
def return_mfactor_current(time_run, shiftLoadsDumps, truck, move = 50, ran=RAND):
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
            m_factor = round(last['avg_matchfactor'].values[0],1)
    else:
        m_factor = round(np.random.uniform(0.4, 0.7),1)
    return m_factor

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
