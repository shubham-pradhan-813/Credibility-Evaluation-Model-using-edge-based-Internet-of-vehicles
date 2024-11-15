import random
import numpy as np

def generate_vehicle_data(rows, num_genuine, num_malicious):
    matrix = []

    if num_genuine + num_malicious != rows:
        raise ValueError("The total number of genuine and malicious vehicles must equal the total number of vehicles.")

    for _ in range(num_genuine):
        x = random.randint(100, 120)
        y = random.randint(100, 120)
        speed = random.randint(40, 60)
        direction = random.randint(0, 1)
        vehicle_type = "Genuine"
        matrix.append([x, y, speed, direction, vehicle_type])

    for _ in range(num_malicious):
        x = random.randint(100, 200)
        y = random.randint(100, 200)
        speed = random.randint(20, 80)
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
                    lq_matrix[i][j] = random.uniform(8, 10) / 10
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


def generate_dt_matrix(rows, lq_matrix, gbss_matrix, plr_matrix, per_matrix, vehicle_data):
    RT = 0.0001
    T = 800
    Tmax = 1000

    # Adjusted weights
    genuine_weight = 0.5
    malicious_weight = 0.1

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

def calculate_idt(dt_matrix):
    idt_values = np.zeros(dt_matrix.shape[0])
    for j in range(dt_matrix.shape[1]):
        idt_values[j] = np.sum(dt_matrix[:, j]) / dt_matrix.shape[0]  # Sum the j-th column and divide by total number of vehicles
    return idt_values

def calculate_ct(dt_matrix, idt_values):
    alpha = 0.5
    ct_values = np.zeros(dt_matrix.shape[0])
    dt_sums = np.sum(dt_matrix, axis=1) / dt_matrix.shape[0]  # Sum each row and divide by total number of vehicles
    
    for i in range(len(ct_values)):
        ct_values[i] = alpha * dt_sums[i] + (1 - alpha) * idt_values[i]
    
    return ct_values

def initialize_edge_node_table(vehicle_data):
    edge_node_table = []
    for i, vehicle in enumerate(vehicle_data):
        vehicle_no = i + 1
        if vehicle[4] == "Genuine":
            previous_ft = random.uniform(0.70, 1.00)  # Genuine FT between 0.70 and 1.00
        else:
            previous_ft = random.uniform(0.10, 0.50)  # Malicious FT between 0.10 and 0.50
        edge_node_table.append([vehicle_no, previous_ft])  # Store vehicle number and Previous FT
    return edge_node_table

def update_edge_node_table(edge_node_table, ft_values):
    for i in range(len(edge_node_table)):
        edge_node_table[i][1] = ft_values[i]  # Update Previous FT with calculated FT

if __name__ == "__main__":
    rows = int(input("Enter the number of vehicles: "))
    num_genuine = int(input("Enter the number of genuine vehicles: "))
    num_malicious = int(input("Enter the number of malicious vehicles: "))
    
    vehicle_data = generate_vehicle_data(rows, num_genuine, num_malicious)
    print_matrix(vehicle_data)

    # Initialize a matrix to store GBSS values
    gbss_matrix = np.zeros((rows, rows))

    # Calculate GBSS for each pair of vehicles
    for i in range(rows):
        for j in range(i + 1, rows):
            vehicle1_features = vehicle_data[i][:4]  # Exclude vehicle_type
            vehicle2_features = vehicle_data[j][:4]  # Exclude vehicle_type
            correlation = pearson_correlation(vehicle1_features, vehicle2_features)
            gbss = abs(correlation)
            gbss_matrix[i, j] = gbss
            gbss_matrix[j, i] = gbss  # Symmetric matrix

    # Print GBSS matrix
    print("\nGBSS Matrix:")
    header = f"{'':<10}" + ''.join(f"{i+1:<10}" for i in range(rows))
    print(header)
    
    for i in range(rows):
        row = f"Vehicle {i+1:<4}"
        for j in range(rows):
            row += f"{gbss_matrix[i, j]:<10.2f}"
        print(row)

    # Generate LQ matrix
    lq_matrix = generate_lq_matrix(rows, vehicle_data)
    print("\nLQ Matrix:")
    print(header)
    
    for i in range(rows):
        row = f"Vehicle {i+1:<4}"
        for j in range(rows):
            row += f"{lq_matrix[i, j]:<10.2f}"
        print(row)

    # Generate PLR matrix
    plr_matrix = generate_plr_matrix(lq_matrix)
    print("\nPLR Matrix:")
    print(header)
    
    for i in range(rows):
        row = f"Vehicle {i+1:<4}"
        for j in range(rows):
            row += f"{plr_matrix[i, j]:<10.2f}"
        print(row)

    # Generate PER matrix
    per_matrix = generate_per_matrix(lq_matrix)
    print("\nPER Matrix:")
    print(header)
    
    for i in range(rows):
        row = f"Vehicle {i+1:<4}"
        for j in range(rows):
            row += f"{per_matrix[i, j]:<10.2f}"
        print(row)

    # Generate DT matrix
    dt_matrix = generate_dt_matrix(rows, lq_matrix, gbss_matrix, plr_matrix, per_matrix, vehicle_data)
    print("\nDT Matrix:")
    print(header)
    
    for i in range(rows):
        row = f"Vehicle {i+1:<4}"
        for j in range(rows):
            row += f"{dt_matrix[i, j]:<10.2f}"
        print(row)

    # Calculate and print Indirect Trust (IDT) values
    idt_values = calculate_idt(dt_matrix)
    print("\nIndirect Trust (IDT) Values:")
    for i in range(len(idt_values)):
        print(f"Vehicle {i+1}: {idt_values[i]:.2f}")

    # Calculate and print Current Trust (CT) values
    ct_values = calculate_ct(dt_matrix, idt_values)
    print("\nCurrent Trust (CT) Values:")
    for i in range(len(ct_values)):
        print(f"Vehicle {i+1}: {ct_values[i]:.2f}")

      # Initialize Edge Node Table
    edge_node_table = initialize_edge_node_table(vehicle_data)
    print("\nEdge Node Table (Vehicle No, Previous FT):")
    for vehicle_no, previous_ft in edge_node_table:
        print(f"Vehicle {vehicle_no}: {previous_ft:.2f}")

    # Calculate Final Trust (FT) and update Edge Node Table
    gamma = 0.5
    final_trust_values = [gamma * edge_node_table[i][1] + (1 - gamma) * ct_values[i] for i in range(rows)]
    
    print("\nFinal Trust (FT) Values:")
    for i, ft in enumerate(final_trust_values):
        print(f"Vehicle {i+1}: {ft:.2f}")

    # Update Previous FT in Edge Node Table
    update_edge_node_table(edge_node_table, final_trust_values)
    print("\nUpdated Edge Node Table (Vehicle No, Previous FT):")
    for vehicle_no, previous_ft in edge_node_table:
        print(f"Vehicle {vehicle_no}: {previous_ft:.2f}")   