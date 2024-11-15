import random
import numpy as np
from datetime import datetime



def generate_vehicle_data(rows, num_genuine, num_malicious):
    matrix = []
    if num_genuine + num_malicious != rows:
        raise ValueError("The total number of genuine and malicious vehicles must equal the total number of vehicles.")

    for _ in range(num_genuine):
        x = random.randint(100, 180)
        y = random.randint(100, 180)
        speed = random.randint(30, 60)
        direction = random.randint(0, 1)
        vehicle_type = "Genuine"
        matrix.append([x, y, speed, direction, vehicle_type])

    for _ in range(num_malicious):
        x = random.randint(100, 200)
        y = random.randint(100, 200)
        speed = random.randint(20, 70)
        direction = random.randint(0, 1)
        vehicle_type = "Malicious"
        matrix.append([x, y, speed, direction, vehicle_type])
    
    return matrix

def print_matrix(matrix):
    print(f"{'Vehicle':<10}{'X':<10}{'Y':<10}{'Speed':<10}{'Direction':<10}{'Type':<10}")
    for i, row in enumerate(matrix):
        print(f"{i+1:<10}{row[0]:<10}{row[1]:<10}{row[2]:<10}{row[3]:<10}{row[4]:<10}")

def pearson_correlation(x, y):
    return np.corrcoef(x, y)[0, 1]

def generate_lq_matrix(rows, vehicle_data):
    lq_matrix = np.zeros((rows, rows))
    for i in range(rows):
        for j in range(rows):
            if i != j:
                if vehicle_data[i][4] == "Genuine" and vehicle_data[j][4] == "Genuine":
                    lq_matrix[i][j] = random.uniform(7, 10) / 10
                elif vehicle_data[i][4] == "Genuine" and vehicle_data[j][4] == "Malicious":
                    lq_matrix[i][j] = random.uniform(6, 7) / 10
                elif vehicle_data[i][4] == "Malicious" and vehicle_data[j][4] == "Genuine":
                    lq_matrix[i][j] = random.uniform(5, 6) / 10
                else:  # Malicious to Malicious
                    lq_matrix[i][j] = random.uniform(1, 4) / 10
    return lq_matrix

def generate_plr_matrix(lq_matrix):
    return 1 - lq_matrix

def generate_per_matrix(lq_matrix):
    per_matrix = np.zeros(lq_matrix.shape)
    for i in range(lq_matrix.shape[0]):
        for j in range(lq_matrix.shape[1]):
            if i != j:
                per_matrix[i][j] = max(0, lq_matrix[i][j] - random.uniform(0.1, 0.2))
    return per_matrix

start_time = datetime.now()


def generate_dt_matrix(rows, lq_matrix, gbss_matrix, plr_matrix, per_matrix, vehicle_data):
    RT = 0.0001
    T = 800
    Tmax = 1000

    genuine_weight = 0.5
    malicious_weight = 0.3

    dt_matrix = np.zeros((rows, rows))
    for i in range(rows):
        for j in range(rows):
            if i != j:
                a_weight = genuine_weight if vehicle_data[i][4] == "Genuine" else malicious_weight
                b_weight = genuine_weight if vehicle_data[j][4] == "Genuine" else malicious_weight
                c_weight = 0.2  # Keeping a constant weight for GBSS
                d_weight = 0.2  # Keeping a constant weight for PLR and PER

                numerator = (T / Tmax) * a_weight + lq_matrix[i][j] * b_weight + gbss_matrix[i][j] * c_weight
                denominator = 1 + RT * d_weight + plr_matrix[i][j] * d_weight + per_matrix[i][j] * d_weight
                dt_matrix[i][j] = numerator / denominator
    return dt_matrix

def generate_idt_matrix(dt_matrix):
    rows = dt_matrix.shape[0]
    idt_matrix = np.zeros((rows, rows))
    
    for i in range(rows):
        for j in range(rows):
            if i != j:
                # Sum all elements in column 'j' excluding the element DT[i][j]
                idt_sum = np.sum(dt_matrix[:, j]) - dt_matrix[i][j]
                idt_matrix[i][j] = idt_sum / (rows - 1)
            else:
                idt_matrix[i][j] = 0  # Diagonal elements are set to 0
    
    return idt_matrix

def generate_ct_matrix(dt_matrix, idt_matrix, alpha=0.5):
    rows = dt_matrix.shape[0]
    ct_matrix = np.zeros((rows, rows))
    
    for i in range(rows):
        for j in range(rows):
            if i != j:
                ct_matrix[i, j] = alpha * dt_matrix[i, j] + (1 - alpha) * idt_matrix[i, j]
    return np.round(ct_matrix, 2)

def generate_edge_node_table(vehicle_data):
    edge_node_table = []
    for i, vehicle in enumerate(vehicle_data):
        if vehicle[4] == "Genuine":
            previous_ft = round(random.uniform(0.70, 1.00), 2)
        else:  # Malicious
            previous_ft = round(random.uniform(0.00, 0.60), 2)
        
        edge_node_table.append([i+1, previous_ft])
    
    return edge_node_table

def print_edge_node_table(edge_node_table):
    print(f"{'Vehicle No.':<15}{'Previous FT':<15}")
    for row in edge_node_table:
        print(f"{row[0]:<15}{row[1]:<15}")

if __name__ == "__main__":
    rows = int(input("Enter the number of vehicles: "))
    num_genuine = int(input("Enter the number of genuine vehicles: "))
    num_malicious = int(input("Enter the number of malicious vehicles: "))
    
    vehicle_data = generate_vehicle_data(rows, num_genuine, num_malicious)
    print("\nVehicle Data")
    print_matrix(vehicle_data)

    gbss_matrix = np.zeros((rows, rows))
    for i in range(rows):
        for j in range(i + 1, rows):
            vehicle1_features = vehicle_data[i][:4]
            vehicle2_features = vehicle_data[j][:4]
            correlation = pearson_correlation(vehicle1_features, vehicle2_features)
            gbss = abs(correlation)
            gbss_matrix[i, j] = gbss
            gbss_matrix[j, i] = gbss

    print("\nGBSS Matrix:")
    header = f"{'':<10}" + ''.join(f"{i+1:<10}" for i in range(rows))
    print(header)
    
    for i in range(rows):
        row = f"Vehicle {i+1:<4}"
        for j in range(rows):
            row += f"{gbss_matrix[i, j]:<10.2f}"
        print(row)

    lq_matrix = generate_lq_matrix(rows, vehicle_data)
    print("\nLQ Matrix:")
    print(header)
    
    for i in range(rows):
        row = f"Vehicle {i+1:<4}"
        for j in range(rows):
            row += f"{lq_matrix[i, j]:<10.2f}"
        print(row)

    plr_matrix = generate_plr_matrix(lq_matrix)
    print("\nPLR Matrix:")
    print(header)
    
    for i in range(rows):
        row = f"Vehicle {i+1:<4}"
        for j in range(rows):
            row += f"{plr_matrix[i, j]:<10.2f}"
        print(row)

    per_matrix = generate_per_matrix(lq_matrix)
    print("\nPER Matrix:")
    print(header)
    
    for i in range(rows):
        row = f"Vehicle {i+1:<4}"
        for j in range(rows):
            row += f"{per_matrix[i, j]:<10.2f}"
        print(row)

    dt_matrix = generate_dt_matrix(rows, lq_matrix, gbss_matrix, plr_matrix, per_matrix, vehicle_data)
    print("\nDT Matrix:")
    print(header)
    
    for i in range(rows):
        row = f"Vehicle {i+1:<4}"
        for j in range(rows):
            row += f"{dt_matrix[i, j]:<10.2f}"
        print(row)


    # Generate the IDT matrix
    idt_matrix = generate_idt_matrix(dt_matrix)

    # Print the IDT matrix
    print("\nIDT Matrix:")
    header = f"{'':<10}" + ''.join(f"{i+1:<10}" for i in range(dt_matrix.shape[0]))
    print(header)

    for i in range(dt_matrix.shape[0]):
        row = f"Vehicle {i+1:<4}"
        for j in range(dt_matrix.shape[1]):
            row += f"{idt_matrix[i, j]:<10.2f}"
        print(row)

    ct_matrix = generate_ct_matrix(dt_matrix, idt_matrix, alpha=0.5)

    # Print the CT matrix
    print("\nCT Matrix:")
    header = f"{'':<10}" + ''.join(f"{i+1:<10}" for i in range(ct_matrix.shape[0]))
    print(header)

    # print_matrix(ct_matrix)
    for i in range(ct_matrix.shape[0]):
        row = f"Vehicle {i+1:<4}"
        for j in range(rows):
            row += f"{ct_matrix[i, j]:<10.2f}"
        print(row)

    
    # Generate the edge node table
    edge_node_table = generate_edge_node_table(vehicle_data)

    # Print the edge node table
    print("\nEdge Node Table:")
    print_edge_node_table(edge_node_table)    


def generate_ft_matrix(ct_matrix, edge_node_table, gamma=0.5):
    rows = ct_matrix.shape[0]
    ft_matrix = np.zeros((rows, rows))

    for i in range(rows):
        previous_ft = edge_node_table[i][1]
        for j in range(rows):
            if i != j:
                ft_matrix[i][j] = gamma * ct_matrix[i][j] + (1 - gamma) * previous_ft
    return np.round(ft_matrix, 2)

# Generate the FT matrix
ft_matrix = generate_ft_matrix(ct_matrix, edge_node_table)

# Print the FT matrix
print("\nFT Matrix:")
header = f"{'':<10}" + ''.join(f"{i+1:<10}" for i in range(ft_matrix.shape[0]))
print(header)

for i in range(ft_matrix.shape[0]):
    row = f"Vehicle {i+1:<4}"
    for j in range(ft_matrix.shape[1]):
        row += f"{ft_matrix[i, j]:<10.2f}"
    print(row)

end_time = datetime.now()

def generate_ft_table(ft_matrix):
    rows = ft_matrix.shape[0]
    ft_table = []

    for i in range(rows):
        # Sum all elements in row 'i' excluding the diagonal element FT[i][i]
        ft_sum = np.sum(ft_matrix[i, :]) - ft_matrix[i][i]
        # Calculate the final FT[i] value
        ft_value = ft_sum / (rows - 1)
        ft_table.append(round(ft_value, 2))

    return ft_table

# Generate the FT table
ft_table = generate_ft_table(ft_matrix)

# Print the FT table
print("\nFT Table:")
print(f"{'Vehicle No.':<15}{'FT Value':<10}")
for i, ft_value in enumerate(ft_table):
    print(f"{i + 1:<15}{ft_value:<10}")


def update_ft_table(ft_table, threshold=0.5, reward=0.125, penalty=-0.125):
    updated_ft_table = []

    for ft_value in ft_table:   
        if ft_value >= threshold:
            new_ft_value = ft_value + reward
        else:
            new_ft_value = max(0, ft_value + penalty)  # Ensure FT doesn't go below 0
        updated_ft_table.append(round(new_ft_value, 2))

    return updated_ft_table

# Update the FT table based on the threshold
updated_ft_table = update_ft_table(ft_table)

# Print the updated FT table
print("\nUpdated FT Table:")
print(f"{'Vehicle No.':<15}{'Updated FT Value':<15}")
for i, updated_ft_value in enumerate(updated_ft_table):
    print(f"{i + 1:<15}{updated_ft_value:<15}")


def update_edge_node_table(edge_node_table, updated_ft_table):
    updated_edge_node_table = []

    for i, edge_node in enumerate(edge_node_table):
        vehicle_no = edge_node[0]
        new_ft_value = updated_ft_table[i]
        updated_edge_node_table.append([vehicle_no, new_ft_value])
    
    return updated_edge_node_table

# Update the edge node table
updated_edge_node_table = update_edge_node_table(edge_node_table, updated_ft_table)

# Print the updated edge node table
print("\nUpdated Edge Node Table:")
print_edge_node_table(updated_edge_node_table)


def count_genuine_and_malicious(updated_edge_node_table, threshold=0.6):
    CG = 0  # Initialize count for Genuine vehicles
    CM = 0  # Initialize count for Malicious vehicles

    for entry in updated_edge_node_table:
        vehicle_no, updated_ft_value = entry
        if updated_ft_value > threshold:
            CG += 1  # Increment Genuine count if FT is greater than the threshold
        else:
            CM += 1  # Increment Malicious count if FT is not greater than the threshold

    return CG, CM

# Count the number of Genuine and Malicious vehicles based on the updated FT table
CG, CM = count_genuine_and_malicious(updated_edge_node_table)

# Print the counts
print(f"\nNumber of Genuine vehicles (CG): {CG}")
print(f"Number of Malicious vehicles (CM): {CM}")

total_given_vehicles = num_genuine + num_malicious
def calculate_accuracy(CG,  num_genuine, total_given_vehicles):
    if CG != num_genuine:
        # Calculate the difference and its modulus
        Val = abs(CG - num_genuine)
        
        # Calculate the accuracy
        accuracy = (total_given_vehicles - Val) / total_given_vehicles * 100
        
    else:
        # If detected and given genuine values are the same, accuracy is 100%
        accuracy = 100 
        falsealarmrate = 0
    
    return accuracy
accuracy = calculate_accuracy(CG,  num_genuine, total_given_vehicles)

# Print results with correct formatting
print(f"Accuracy: {accuracy:.2f}%")

def calculate_falsealarmrate(CG,CM, num_genuine):
    Val = abs(CG - num_genuine)
    falsealarmrate = Val / (Val + CM) * 100 
    return falsealarmrate

falsealarmrate = calculate_falsealarmrate(CG, CM ,num_genuine)

print(f"False Alarm Rate: {falsealarmrate:.2f}%")

# Calculate the detection time
detection_time = end_time - start_time

# Convert detection time to seconds
detection_time_seconds = detection_time.total_seconds()

# Print the detection time
print(f"Detection Time: {detection_time_seconds:.2f} seconds")












 