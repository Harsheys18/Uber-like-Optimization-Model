import pandas as pd
import numpy as np
import os
from pyomo.environ import *

# Dynamic pricing data for past rides
dynamic_pricing_data = {
    "Number_of_Riders": [90, 58, 42, 89, 78, 59, 93, 62, 79, 42, 86, 60, 24, 36],
    "Number_of_Drivers": [45, 39, 31, 28, 22, 35, 43, 39, 14, 6, 17, 38, 8, 24],
    "Location_Category": ["Churchgate", "Goregaon", "Lonavala", "Virar", "Virar", "Bandra", "Goregaon", "Panvel", "Virar", "Malad", "Colaba", "Virar", "Kalyan", "Thane"],
    "Customer_Loyalty_Status": ["Silver", "Silver", "Silver", "Regular", "Regular", "Silver", "Regular", "Gold", "Silver", "Silver", "Regular", "Gold", "Gold", "Silver"],
    "Number_of_Past_Rides": [13, 72, 0, 67, 74, 83, 44, 83, 71, 21, 99, 15, 50, 88],
    "Average_Ratings": [4.47, 4.06, 3.99, 4.31, 3.77, 3.51, 4.41, 3.59, 3.74, 3.85, 4.69, 4.14, 4.12, 4.22],
    "Time_of_Booking": ["Night", "Evening", "Afternoon", "Afternoon", "Afternoon", "Night", "Afternoon", "Afternoon", "Evening", "Night", "Morning", "Evening", "Morning", "Night"],
    "Vehicle_Type": ["Premium", "Economy", "Premium", "Premium", "Economy", "Economy", "Premium", "Premium", "Economy", "Premium", "Economy", "Premium", "Economy", "Premium"],
    "Expected_Ride_Duration": [90, 43, 76, 134, 149, 128, 16, 47, 128, 128, 167, 144, 164, 83],
    "Historical_Cost_of_Ride": [
        284.257273, 173.8747527, 329.795469, 470.2012318, 579.6814224, 339.9553606, 104.0615413, 235.8118636, 501.4125175,
        398.9933646, 669.2986265, 414.9901047, 490.4107217, 296.242876
    ]
}

# Convert dynamic pricing data to a DataFrame
dynamic_pricing_df = pd.DataFrame(dynamic_pricing_data)

# Extract relevant parameters
passengers = list(range(dynamic_pricing_df["Number_of_Riders"].sum()))  # Total number of passengers
drivers = list(range(dynamic_pricing_df["Number_of_Drivers"].sum()))  # Total number of drivers
locations = dynamic_pricing_df["Location_Category"].unique().tolist()  # Locations
times_of_day = dynamic_pricing_df["Time_of_Booking"].unique().tolist()  # Times of day

# Create a dummy distance matrix based on some logic, assuming distances between passengers and drivers (example)
distance_matrix = np.array([[abs(p - d) * 10 for d in drivers] for p in passengers])

# Create a revenue matrix based on historical cost of ride for each passenger
revenue = np.array([[dynamic_pricing_df["Historical_Cost_of_Ride"].mean() for d in drivers] for p in passengers])

# Create a model
model = ConcreteModel() 

# Decision variable: tij is a binary variable for passenger-driver assignment
model.tij = Var(passengers, drivers, domain=Binary)

# Objective 1: Maximize Revenue
model.revenue_obj = Objective(expr=sum(model.tij[p, d] * revenue[p][d] for p in passengers for d in drivers), sense=maximize)

# Constraints for maximizing revenue
model.constraints_revenue = ConstraintList()

# Each passenger is assigned to at most one driver
for p in passengers:
    model.constraints_revenue.add(sum(model.tij[p, d] for d in drivers) <= 1)

# Each driver serves at most one passenger
for d in drivers:
    model.constraints_revenue.add(sum(model.tij[p, d] for p in passengers) <= 1)

# Driver availability by location and time (considering dynamic pricing)
for idx, location in enumerate(locations):
    for time in times_of_day:
        available_drivers = dynamic_pricing_df[(dynamic_pricing_df["Location_Category"] == location) & (dynamic_pricing_df["Time_of_Booking"] == time)]["Number_of_Drivers"].sum()
        model.constraints_revenue.add(sum(model.tij[p, d] for p in passengers for d in drivers) <= available_drivers)

# Solver and solve the revenue maximization problem
# Set a directory for temporary files
os.environ['TMPDIR'] = '/mnt/c/Users/inamp/Desktop/Uber_SCO/'

solver = SolverFactory('cbc_latest', executable='/mnt/c/Users/inamp/Desktop/Uber_SCO/cbc/bin/cbc.exe')  # Or use 'glpk' or any solver available
solver.solve(model, tee=True, keepfiles=True)
# Specify solver options if needed
solver.options['write_level'] = 0  # Optional, reduces verbosity

# Print results for revenue maximization
print("Optimal solution for revenue maximization:")
for p in passengers:
    for d in drivers:
        if model.tij[p, d].value > 0.5:
            print(f"Passenger {p} is assigned to Driver {d} for Revenue Maximization")

# Objective 2: Minimize Distance
model.distance_obj = Objective(expr=sum(model.tij[p, d] * distance_matrix[p][d] for p in passengers for d in drivers), sense=minimize)

# Constraints for minimizing distance are the same as the revenue maximization problem
model.constraints_distance = ConstraintList()

# Solve the distance minimization problem
solver.solve(model, tee=True)

# Print results for distance minimization
print("\nOptimal solution for distance minimization:")
for p in passengers:
    for d in drivers:
        if model.tij[p, d].value > 0.5:
            print(f"Passenger {p} is assigned to Driver {d} for Distance Minimization")