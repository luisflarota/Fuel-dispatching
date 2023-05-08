import streamlit as st
import datetime
import pandas as pd
from back import *
from schedule import *
import utils
np.random.seed(5)
def main():
    st.title("Automated Fuel Dispatching Project")
    st.success("Select a time to run the optimization model")
    start_time = st.slider(
        "",
        min_value=datetime(2022, 11, 11, 8, 0, 0),
        max_value=datetime(2022, 11, 11, 18, 0, 0),
        step=pd.Timedelta(2, "h"),
        format="YYYY-MM-DD HH:mm:ss")
    start_time = pd.to_datetime(start_time)
    try:
        fuelratio, mfactor, currentFuel  = returnData(start_time)
        parametersTable = utils.returntableparm(fuelratio, mfactor)
        st.success("Parameters for the optimization model")
        st.dataframe(parametersTable.style.format({
                                    "Time to st. (min)": '{:.1f}',
                                    "Fuel ratio at st. (%)": '{:.1f}',
                                    "Match factor": '{:.1f}'}))
        #parametersTable.to_csv("params.csv")
        optModelRun = st.radio("Run Optimization Model - 15 time window"
                                ,['Yes', 'No'], index  =1, key=1)
        dataopt = returndataopt(fuelratio, mfactor)
        length, timestamps, frationew, matchfb = ret_timestamp(dataopt)
        fratiopt, timestamps, mfopt = createBigdict(
            timestamps, frationew.keys(), frationew, matchfb
            )
        cols_final = ['Truck', "ETA at fuel station (st.)", "Fuel ratio at st. (%)", 'Match factor']
        if optModelRun == 'Yes':
            scheduleTrucks = runOptimization(frationew, timestamps, fratiopt, mfopt, all='Yes')
            print('y'*10)
            print(scheduleTrucks)
            try:
                result = []
                for v in scheduleTrucks.getVars():
                    if v.X >0:
                        truck = v.VarName[9:14]
                        time = pd.to_datetime(v.VarName[15:34])
                        result.append([truck, time, frationew[truck][time], matchfb[truck][time]])
                result = pd.DataFrame(result, columns=cols_final)
                st.dataframe(result)  
                st.info("Average match factor: {}".format(np.mean(result['Match factor'])))  
                st.info("Run time: {:.5f} sec".format(scheduleTrucks.Runtime))
                scheduleTrucks.write('file.lp')
                with open('file.lp') as f:
                    lines = f.read()
                st.download_button(
                    label = 'Download file with model (.lp)',data = lines,
                    file_name = 'file.lp')
                # Save solutions
                ## Values for decision variables - option 1
                st.download_button(
                    label = 'Download file with solution (.txt)',data = str(scheduleTrucks.getVars()).replace(">", ">\n"),
                    file_name = 'file.txt')
                ## Values for decision variables - option 2
                scheduleTrucks.write('file.sol')
                with open('file.sol') as f:
                    lines = f.read()
                st.download_button(
                    label = "Download solution's file (.mps)",data = lines,
                    file_name = 'solution.sol')
            except:
                st.error('Model is infeasible, but if (some) trucks go to a mobile fuel station:')
                st.success('Schedule for fixed location')
                scheduleTrucks = runOptimization(frationew, timestamps, fratiopt, mfopt, all='No')
                result = []
                for v in scheduleTrucks.getVars():
                    if v.X >0:
                        truck = v.VarName[9:14]
                        time = pd.to_datetime(v.VarName[15:34])
                        result.append([truck, time, frationew[truck][time], matchfb[truck][time]])
                result = pd.DataFrame(result, columns=cols_final)
                result.to_csv('Timewindow1.csv')
                st.dataframe(result)  
                st.info("Average match factor: {:.1f}".format(np.mean(result['Match factor'])))  
                st.info("Run time: {:.5f} sec".format(scheduleTrucks.Runtime))
                # Save model file
                scheduleTrucks.write('file.lp')
                with open('file.lp') as f:
                    lines = f.read()
                st.download_button(
                    label = 'Download file with model (.lp)',data = lines,
                    file_name = 'file.lp')
                # Save solutions
                ## Values for decision variables - option 1
                st.download_button(
                    label = 'Download file with solution (.txt)',data = str(scheduleTrucks.getVars()).replace(">", ">\n"),
                    file_name = 'file.txt')
                ## Values for decision variables - option 2
                scheduleTrucks.write('file.sol')
                with open('file.sol') as f:
                    lines = f.read()
                st.download_button(
                    label = "Download solution's file (.mps)",data = lines,
                    file_name = 'solution.sol')
            
        st.markdown("""---""")
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
                # result.to_csv("result.csv")
                st.success('Schedule:')
                st.dataframe(result)
                st.info("Average match factor: {:.1f}".format(np.mean(result['Match factor']))) 
                st.info("Run time: {:.5f} sec".format(scheduleTrucks.Runtime))
                # Save model file
                scheduleTrucks.write('file.lp')
                with open('file.lp') as f:
                    lines = f.read()
                st.download_button(
                    label = 'Download file with model (.lp)',data = lines,
                    file_name = 'file.lp')
                objV = "Objective Function {}>>\n".format(scheduleTrucks.getObjective().getValue())
                # Save solutions
                ## Values for decision variables - option 1
                st.download_button(
                    label = 'Download file with solution (.txt)',data = objV+str(scheduleTrucks.getVars()).replace(">", ">\n"),
                    file_name = 'file.txt')
                ## Values for decision variables - option 2
                scheduleTrucks.write('file.sol')
                with open('file.sol') as f:
                    lines = f.read()
                st.download_button(
                    label = "Download solution's file (.mps)",data = lines,
                    file_name = 'solution.sol')
                st.markdown("""----""")
                st.write('Comparison with current assignments')
                showCurrent, averageTime = substractdates(currentFuel, result)
                st.dataframe(showCurrent)
                showCurrent.to_csv("current.csv")
                st.info("Average match factor: {:.1f}".format(np.mean(currentFuel['MatchFactor']))) 
                st.info("Average difference in time: (+/-) {:.2f} minutes".format(averageTime/60))

            except:
                #print(scheduleTrucks)
                st.warning('Model is infeasible')
    except:
        st.error('No trucks to fuel')
@st.cache_data
def returnData(start_time):
    fuelratio, mfactor, currentfueltrucks = TransformData(start_time).fuelratio()
    return fuelratio, mfactor, currentfueltrucks

if __name__ == '__main__':
    main()

