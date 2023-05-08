import gurobipy as gp
from gurobipy import GRB
import pandas as pd
from utils import *
np.random.seed(5)
# Create a new optimization model with the given name
def runOptimization(frationew, timestamps, fratiopt, mfopt, all='Yes'):
    # Create a new optimization model
    model = gp.Model('Truckfake')
    
    # Get the set of all trucks
    trucks = frationew.keys()
    print('z'*10)
    print(len(trucks))
    # Get the set of all time periods
    periods = timestamps
    
    # Add binary decision variables for each truck and time period
    y_truck_time = model.addVars(trucks, periods, vtype=GRB.BINARY, name='truckbin')

    # Add constraints to the model
    
    ## Only one truck can be active at each time period
    one_truck_time = model.addConstrs(
        gp.quicksum(y_truck_time[truck, time] for truck in trucks) <= 1 for time in timestamps)
    
    ## Each truck can only fuel once
    time_taken = model.addConstrs(
        gp.quicksum(y_truck_time[truck, time] for time in timestamps) <= 1 for truck in trucks)

    ## Each truck must have enough fuel to complete its assigned time periods
    norunfuel = model.addConstrs(
        gp.quicksum(y_truck_time[truck, time]* fratiopt[truck][time] for time in timestamps) >=0 for truck in trucks)
    
    ## All trucks should fuel
    if all=='Yes':
        model.addConstr(gp.quicksum(
            y_truck_time[truck, time] for truck in trucks for time in timestamps
                ) >= len(trucks))

    ## Only 15 minutes allowed for fueling (previously 5)
    time_to_fuel = pd.Timedelta(15, 'm')
    
    ### Find the earliest and latest time periods
    min_t = min(periods)
    max_t = max(periods)
    
    ### Add constraints to ensure that no truck is refueling while on duty
    for t in periods:
        t_fuel_before = time_to_fuel
        t_fuel_after = time_to_fuel       
        
        if t - min_t < time_to_fuel:
            t_fuel_before = t - min_t
        if max_t - t < time_to_fuel:
            t_fuel_after = max_t - t
        
        # Define the range of times when the truck cannot refuel
        timerange_a = [pd.to_datetime(dt.strftime('%Y-%m-%d T%H:%M')) for dt in datetime_range(t-t_fuel_before, t, timedelta(minutes=1))]
        timerange_b = [pd.to_datetime(dt.strftime('%Y-%m-%d T%H:%M')) for dt in datetime_range(t+pd.Timedelta(1, 'm'), t+t_fuel_after, timedelta(minutes=1))]
        if len(timerange_a) > 1 and len(timerange_b) > 1:
            bef = gp.quicksum(y_truck_time[i, tt] for tt in timerange_a for i in trucks)
            after = gp.quicksum(y_truck_time[i, tt] for tt in timerange_b for i in trucks)
            model.addConstr(bef+after <=1)
        if len(timerange_a) <= 1 and len(timerange_a) > 0 and len(timerange_b) > 1:
            current = gp.quicksum(y_truck_time[i,timerange_a[0]] for i in trucks)
            after = gp.quicksum(y_truck_time[i, tt] for tt in timerange_b for i in trucks)
            model.addConstr(current+after <=1)
        if len(timerange_b) <= 1 and len(timerange_b) > 0 and len(timerange_a) > 1:
            bef = gp.quicksum(y_truck_time[i, tt] for tt in timerange_a for i in trucks)
            current = gp.quicksum(y_truck_time[i,timerange_b[0]] for i in trucks)
            model.addConstr(bef+current <=1)
    # Objective function
    max_mf = gp.quicksum(
        mfopt[truck][time]*y_truck_time[truck,time]
        for truck in trucks
        for time in periods)

    ## Set the objective of the model to maximize the difference between the match factor and the time taken
    model.setObjective(max_mf, GRB.MAXIMIZE)

    # Solve the optimization model
    model.optimize()

    return model

# Create a new optimization model with the given name
def runOptimization_2(frationew, timestamps, fratiopt, mfopt):
    # Create a Gurobi optimization model instance
    model = gp.Model('Truckfake2')

    # Define the set of all trucks
    trucks = frationew.keys()

    # Define the set of all time periods
    periods = timestamps

    # Add binary decision variables for each truck and time period
    y_truck_time = model.addVars(trucks, periods, vtype=GRB.BINARY, name='truckbin')

    # Add constraints to the model
    ## Only one truck can be active at each time period
    one_truck_time = model.addConstrs(
        gp.quicksum(y_truck_time[truck, time] for truck in trucks) <= 1 for time in timestamps)

    ## Each truck can only fuel once and all should fuel
    time_taken = model.addConstrs(
        gp.quicksum(y_truck_time[truck, time] for time in timestamps) <= 1 for truck in trucks)

    ## All trucks should fuel
    model.addConstr(gp.quicksum(y_truck_time[truck, time] for truck in trucks for time in timestamps) >= len(trucks))

    ## Each truck must have enough fuel to complete its assigned time periods
    norunfuel = model.addConstrs(
        gp.quicksum(y_truck_time[truck, time] * fratiopt[truck][time] for time in timestamps) >= 0 for truck in trucks)

    # Calculate maximum match factor
    max_mf = gp.quicksum(mfopt[truck][time] * y_truck_time[truck, time] for truck in trucks for time in periods)

    # Set the objective of the model to maximize the difference between the match factor and the time taken
    model.setObjective(max_mf, GRB.MAXIMIZE)

    # Solve the optimization model
    model.optimize()

    return model


# Create a new optimization model with the given name
def runOptimization_3(frationew, timestamps, fratiopt, mfopt):
    # Create a Gurobi optimization model instance
    model = gp.Model('Truckfake2')

    # Define the set of all trucks
    trucks = frationew.keys()

    # Define the set of all time periods
    periods = timestamps

    # Add binary decision variables for each truck and time period
    y_truck_time = model.addVars(trucks, periods, vtype=GRB.BINARY, name='truckbin')

    # Add constraints to the model
    ## Only one truck can be active at each time period
    one_truck_time = model.addConstrs(
        gp.quicksum(y_truck_time[truck, time] for truck in trucks) <= 1 for time in timestamps)

    ## Each truck can only fuel once and all should fuel
    time_taken = model.addConstrs(
        gp.quicksum(y_truck_time[truck, time] for time in timestamps) <= 1 for truck in trucks)

    ## All trucks should fuel
    model.addConstr(gp.quicksum(y_truck_time[truck, time] for truck in trucks for time in timestamps) >= len(trucks))

    ## Each truck must have enough fuel to complete its assigned time periods
    norunfuel = model.addConstrs(
        gp.quicksum(y_truck_time[truck, time] * fratiopt[truck][time] for time in timestamps) >= 0 for truck in trucks)

    # Calculate maximum match factor
    max_mf = gp.quicksum(mfopt[truck][time] * y_truck_time[truck, time] for truck in trucks for time in periods)

    # Set the objective of the model to maximize the difference between the match factor and the time taken
    model.setObjective(max_mf, GRB.MAXIMIZE)

    # Solve the optimization model
    model.optimize()

    return model