import matplotlib.pyplot as plt

# Data for number of vehicles
num_vehicles = [20, 40, 60, 80, 100]

# Detection time for each case (in second)
DT_model_1 = [18.3, 20.8, 26.1, 29.5,32.9]  # Proposed model
DT_model_2 = [24.2, 28.9, 30.1, 36.6, 39.6]  # Ke et al. model

# Plotting the line graph
plt.figure(figsize=(10, 6))
plt.plot(num_vehicles, DT_model_1, marker='o', label='Proposed Model', linestyle='-', color='blue')
plt.plot(num_vehicles, DT_model_2, marker='o', label='Ke et al. Model', linestyle='-', color='red')

plt.title('Detection Time vs Number of Vehicles')
plt.xlabel('Number of Vehicles')
plt.ylabel('Detection Time (%)')
plt.xticks(num_vehicles)
plt.ylim(0, 100)
plt.grid(True)
plt.legend()

# Show the plot
plt.show()
