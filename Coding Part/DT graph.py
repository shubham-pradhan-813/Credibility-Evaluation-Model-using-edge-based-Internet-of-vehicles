import matplotlib.pyplot as plt

# Data for number of vehicles
num_vehicles = [20, 40, 60, 80, 100]

# Detection time for each case (in second)
DT_model_1 = [8.3, 12.8, 16.1, 20.5,22.9]  # Proposed model
DT_model_2 = [15.2, 19.9, 23.1, 27.6, 31.6]  # Ke et al. model

# Plotting the line graph
plt.figure(figsize=(10, 6))
plt.plot(num_vehicles, DT_model_1, marker='o', label='Proposed Model', linestyle='-', color='blue')
plt.plot(num_vehicles, DT_model_2, marker='o', label='Changbo Ke et al.(2024)', linestyle='-', color='red')

plt.title('Detection Time vs Number of Vehicles', fontsize=18)
plt.xlabel('Number of Vehicles', fontsize=16)
plt.ylabel('Detection Time (in secs)', fontsize=16)
plt.xticks(num_vehicles, fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(0, 100)
plt.grid(True)
plt.legend(fontsize=14)

# Show the plot
plt.show()
