import cvxpy as cp
import pandas as pd
import numpy as np

# Expanded dynamic_pricing_data (20 entries)
dynamic_pricing_data = {
    "Number_of_Riders": np.random.randint(30, 100, size=20),
    "Number_of_Drivers": np.random.randint(20, 50, size=20),
    "Location_Category": np.random.choice(["Urban", "Suburban", "Rural"], size=20),
    "Customer_Loyalty_Status": np.random.choice(["Silver", "Gold", "Regular"], size=20),
    "Number_of_Past_Rides": np.random.randint(0, 100, size=20),
    "Average_Ratings": np.random.uniform(3.5, 5.0, size=20),
    "Time_of_Booking": np.random.choice(["Morning", "Afternoon", "Evening", "Night"], size=20),
    "Vehicle_Type": np.random.choice(["Premium", "Economy"], size=20),
    "Expected_Ride_Duration": np.random.randint(30, 150, size=20),
    "Historical_Cost_of_Ride": np.random.uniform(100, 600, size=20)
}

# Expanded forecast_data (20 locations, 4 times of day)
locations = ["Andheri", "Bandra", "Churchgate", "Colaba", "Dadar", "Goregaon", "Kalyan", "Lonavala", "Malad", "Mulund", 
             "Navi Mumbai", "Panvel", "Powai", "Thane", "Virar", "Borivali", "Kurla", "Vashi", "Jogeshwari", "Versova"]
forecast_data = {
    "Morning": np.random.randint(100, 800, size=20),
    "Afternoon": np.random.randint(100, 800, size=20),
    "Evening": np.random.randint(100, 800, size=20),
    "Night": np.random.randint(100, 800, size=20),
}

# Create pandas DataFrame for forecast
forecast_df = pd.DataFrame(forecast_data, index=locations)

# Number of passengers and drivers
num_passengers = len(dynamic_pricing_data["Number_of_Riders"])
num_drivers = len(forecast_df)

# Create variables tij for assignment (passenger i to driver j)
tij = cp.Variable((num_passengers, num_drivers), boolean=True)

# Historical cost and forecast-based costs
historical_costs = dynamic_pricing_data["Historical_Cost_of_Ride"]

# Reshape historical costs into a column vector for element-wise multiplication
historical_costs_matrix = cp.Constant(historical_costs).reshape((num_passengers, 1))

# Element-wise multiplication of tij and historical costs (to calculate the total cost)
cost = cp.multiply(tij, historical_costs_matrix)

# Forecast cost (driver availability)
forecast_cost = 0
for p in range(num_passengers):
    for d in range(num_drivers):
        time_for_passenger = dynamic_pricing_data["Time_of_Booking"][p]
        location_for_driver = locations[d]
        
        # Ensure the forecast data exists for that time and location
        if time_for_passenger in forecast_df.columns and location_for_driver in forecast_df.index:
            forecast_value = forecast_df.at[location_for_driver, time_for_passenger]
            forecast_cost += tij[p, d] * forecast_value

# Create a simulated distance matrix for optimization (example: randomly generated)
distance_matrix = np.random.randint(1, 100, size=(num_passengers, num_drivers))  # Simulated distances

# Distance cost
distance_cost = cp.sum(cp.multiply(tij, distance_matrix))

# Example revenue matrix (randomly generated for this example)
revenue_matrix = np.random.randint(50, 200, size=(num_passengers, num_drivers))  # Revenue per passenger-driver pair

# Revenue cost (maximize revenue)
revenue_obj = cp.sum(cp.multiply(tij, revenue_matrix))

# Objective function weights
weight_cost = 0.3
weight_forecast = 0.3
weight_distance = 0.2
weight_revenue = 0.2

# New combined objective function (maximize revenue and minimize cost, forecast, and distance)
objective = cp.Minimize(
    weight_cost * cp.sum(cost) + weight_forecast * forecast_cost + weight_distance * distance_cost - weight_revenue * revenue_obj
)

# Constraints: Each passenger must be assigned to exactly one driver
constraints = []
for p in range(num_passengers):
    constraints.append(cp.sum(tij[p, :]) == 1)

# Constraints: Each driver can have at most one passenger
for d in range(num_drivers):
    constraints.append(cp.sum(tij[:, d]) <= 1)

# Driver availability constraint based on forecast
for p in range(num_passengers):
    for d in range(num_drivers):
        time_for_passenger = dynamic_pricing_data["Time_of_Booking"][p]
        location_for_driver = locations[d]
        if time_for_passenger in forecast_df.columns and location_for_driver in forecast_df.index:
            forecast_value = forecast_df.at[location_for_driver, time_for_passenger]
            constraints.append(tij[p, d] <= forecast_value)

# Solve the optimization problem using CBC solver
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.CBC, verbose=True)

# Display the results
print(f"Optimized Objective Value: {problem.value}")
assignments = []
for p in range(num_passengers):
    for d in range(num_drivers):
        if tij.value[p, d] > 0.5:  # Binary solution (0 or 1)
            assignments.append((p, d, distance_matrix[p][d], revenue_matrix[p][d]))
            print(f"Passenger {p} is assigned to Driver {d} with Distance Cost: {distance_matrix[p][d]} and Revenue: {revenue_matrix[p][d]}")