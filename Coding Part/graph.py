
import matplotlib.pyplot as plt

# Data for number of vehicles
num_vehicles = [20, 40, 60, 80, 100]

# Detection accuracy for each case (in percentage)
# Assuming linear downward trend for both models
accuracy_model_1 = [96, 93, 90, 87, 85]  # Proposed model
accuracy_model_2 = [91, 87, 85, 81, 80]  # Ke et al. model

# Plotting the line graph
plt.figure(figsize=(10, 6))
plt.plot(num_vehicles, accuracy_model_1, marker='o', label='Proposed Model', linestyle='-', color='blue')
plt.plot(num_vehicles, accuracy_model_2, marker='o', label='Changbo Ke et al.(2024)', linestyle='-', color='red')

# Increasing font size for title, labels, and legend
plt.title('Detection Accuracy vs Number of Vehicles', fontsize=18)
plt.xlabel('Number of Vehicles', fontsize=16)
plt.ylabel('Detection Accuracy (%)', fontsize=16)
plt.xticks(num_vehicles, fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(0, 100)
plt.grid(True)
plt.legend(fontsize=14)

# Show the plot
plt.show()
