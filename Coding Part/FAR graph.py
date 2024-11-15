import matplotlib.pyplot as plt

# Data for number of vehicles
num_vehicles = [20, 40, 60, 80, 100]

# False Alarm rate for each case (in percentage)

FAR_model_1 = [8, 13, 16, 18, 22]  # Proposed model
FAR_model_2 = [15, 21, 25, 28, 28 ]  # Ke et al. model

# Plotting the line graph
plt.figure(figsize=(10, 6))
plt.plot(num_vehicles, FAR_model_1, marker='o', label='Proposed model', linestyle='-', color='blue')
plt.plot(num_vehicles, FAR_model_2, marker='o', label='Changbo Ke et al.(2024)', linestyle='-', color='red')

plt.title('False Alarm Rate vs Number of Vehicles',fontsize=18)
plt.xlabel('Number of Vehicles',fontsize=16)
plt.ylabel('False Alarm Rate (%)',fontsize=16)
plt.xticks(num_vehicles,fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(0,100)
plt.grid(True)
plt.legend(fontsize=14)

# Show the plot
plt.show()
