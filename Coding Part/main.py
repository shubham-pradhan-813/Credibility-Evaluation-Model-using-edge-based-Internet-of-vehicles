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

def calculate_packet_loss_rate(link_quality):
    return 1 - link_quality

def calculate_packet_error_rate(link_quality, a):
    return link_quality - a

# Define weights
a_weight = 0.2
b_weight = 0.2
c_weight = 0.2
d_weight = 0.2
e_weight = 0.2

# Define constants
throughput = 800  # bits/second
response_time = 0.0001  # seconds

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
        row = f"{i+1:<10}"
        for j in range(rows):
            row += f"{gbss_matrix[i, j]:<10.2f}"
        print(row)

    # Calculate Link Quality
    vehicles_link_quality = []

    for _ in range(num_genuine):
        link_quality = random.randint(7, 10) / 10
        vehicles_link_quality.append(link_quality)

    for _ in range(num_malicious):
        link_quality = random.randint(1, 5) / 10
        vehicles_link_quality.append(link_quality)

    vehicles_link_quality[0] = 0  # Ensure the first vehicle has link quality 0

    # Calculate packet error rates for the vehicles
    a = random.uniform(0.1, 0.2)
    packet_error_rates = [calculate_packet_error_rate(lq, a) for lq in vehicles_link_quality]

    # Print Vehicle Information
    print("\nVehicle Information:")
    print(f"{'Vehicle':<10}{'Link Quality':<15}{'Packet Loss Rate':<20}{'Packet Error Rate':<20}")
    for i in range(rows):
        link_quality = vehicles_link_quality[i]
        packet_loss_rate = calculate_packet_loss_rate(link_quality)
        packet_error_rate = packet_error_rates[i]
        print(f"{i+1:<10}{link_quality:<15.2f}{packet_loss_rate:<20.2f}{packet_error_rate:<20.2f}")

    # Calculate Direct Trust for each vehicle
    trust_matrix = np.zeros((rows, rows))
    throughput_value = [throughput] * rows  # Assign throughput to all vehicles
    response_time_value = [response_time] * rows  # Assign response time to all vehicles

    for i in range(rows):
        T = throughput_value[i]
        Tmax = 1000
        LQ = vehicles_link_quality[i]
        GBSS = np.mean(gbss_matrix[i])  # Mean GBSS value for the vehicle
        RT = response_time_value[i]
        PLR = calculate_packet_loss_rate(LQ)
        PER = packet_error_rates[i]

        numerator = (T / Tmax) * a_weight + LQ * b_weight + GBSS * c_weight
        denominator = 1 + RT * d_weight + PLR * e_weight + PER * e_weight
        trust = numerator / denominator

        for j in range(rows):
            if i != j:
                trust_matrix[i][j] = trust

    # Set the diagonal elements of trust matrix to 0
    np.fill_diagonal(trust_matrix, 0)

    # Print the trust matrix
    print("\nDirect Trust Matrix:")
    header = f"{'':<10}" + ''.join(f"{i+1:<10}" for i in range(rows))
    print(header)

    for i in range(rows):
        row = f"{i+1:<10}"
        for j in range(rows):
            row += f"{trust_matrix[i, j]:<10.2f}"
        print(row)
  