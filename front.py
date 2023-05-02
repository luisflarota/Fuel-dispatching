import streamlit as st
import datetime
import pandas as pd
from back import *
from schedule import *
np.random.seed(5)
def main():
    st.title("Automated Fuel Dispatching Project")
    
    start_time = st.slider(
        "Please select a time to run the optimization model",
        min_value=datetime(2022, 11, 11, 8, 0, 0),
        max_value=datetime(2022, 11, 11, 18, 0, 0),
        step=pd.Timedelta(2, "h"),
        format="YYYY-MM-DD HH:mm:ss")
    start_time = pd.to_datetime(start_time)
    st.write(start_time)
    # Show info of trucks 
    showTrucksInfo = st.radio(
        "Show me the parameters:"
                              ,['Yes', 'No'], index  =1)
    fuelData, shiftLoadsDumps, mlmodel, graph = returnData()
    try:
        trucks_fuelratio, currentFuel = return_fuel_ratio(
                fuelData, 
                start_time,
                shiftLoadsDumps,
                mlmodel)
        #st.dataframe(currentFuel)
        #st.write(trucks_fuelratio)
        fuelratio = return_possible_dumptime(
                trucks_fuelratio, shiftLoadsDumps, start_time, 
                graph, mlmodel)
        #st.write(fuelratio)
        mfactor = returnmfactor(fuelratio, shiftLoadsDumps)
        if showTrucksInfo == 'Yes':
            # print(fuelratio)
            ratiotab = converttodf(fuelratio, par='fuel')
            mfactortab = converttodf(mfactor, par='mfactor')
            ratiotab['Next Shovel']= mfactortab['Next Shovel']
            ratiotab['Match factor']= mfactortab['Match factor']
            ratiotab["ETA at fuel station (st.)"] = ratiotab["ETA at fuel station (st.)"].dt.strftime("%Y-%m-%d %H:%M:%S")
            st.dataframe(ratiotab.style.format({
                                        "Time to st. (min)": '{:.1f}',
                                        "Fuel ratio at st. (%)": '{:.1f}',
                                        "Match factor": '{:.1f}'}))
            #ratiotab.to_csv('Datareport.csv', index=False)
            #st.dataframe(currentFuel[['Equipment', 'StartTimestamp']])   #if you want to check the schedule
        # Run optimization model 
        optModelRun = st.radio("Run Optimization Model - 15 time window"
                                ,['Yes', 'No'], index  =1)
        dataopt = returndataopt(fuelratio, mfactor)
        length, timestamps, frationew, matchfb = ret_timestamp(dataopt)
        
        fratiopt, timestamps, mfopt = createBigdict(
            timestamps, frationew.keys(), frationew, matchfb
            )
        print(matchfb)
        cols_final = ['Truck', "ETA at fuel station (st.)", "Fuel ratio at st. (%)", 'Match factor']
        if optModelRun == 'Yes':
            scheduleTrucks = runOptimization(frationew, timestamps, fratiopt, mfopt)
            try:
                result = []
                for v in scheduleTrucks.getVars():
                    if v.X >0:
                        truck = v.VarName[9:14]
                        time = pd.to_datetime(v.VarName[15:34])
                        result.append([truck, time, frationew[truck][time], matchfb[truck][time]])
                result = pd.DataFrame(result, columns=cols_final)
                # result.to_csv('Timewindow1.csv', index=False)
                st.success('Schedule:')
                st.dataframe(result)  
                st.write("Average match factor: {}".format(np.mean(result['Match factor'])))        
            except:
                #print(scheduleTrucks)
                st.warning('Model is infeasible')

        optModelRun2 = st.radio("Run Optimization Model - Final modification"
                                ,['Yes', 'No'], index  =1, key="2")
        if optModelRun2 == 'Yes':
            scheduleTrucks = runOptimization_2(frationew, timestamps, fratiopt, mfopt)
            try:
                result = []
                for v in scheduleTrucks.getVars():
                    if v.X >0:
                        truck = v.VarName[9:14]
                        time = pd.to_datetime(v.VarName[15:34])
                        result.append([truck, time, frationew[truck][time], matchfb[truck][time]])
                result = pd.DataFrame(result, columns=cols_final)
                st.success('Schedule:')
                st.dataframe(result)
                #result.to_csv('final.csv', index=False)
                st.write("Average match factor: {:.1f}".format(np.mean(result['Match factor'])))        
            except:
                #print(scheduleTrucks)
                st.warning('Model is infeasible')
    except:
        st.warning('No trucks to fuel')
@st.experimental_singleton
def returnData():
    DataMaster = TransformData()
    return DataMaster.fuelData, DataMaster.shiftLoadsDumps, DataMaster.mlmodel, DataMaster.Graph


if __name__ == '__main__':
    main()
