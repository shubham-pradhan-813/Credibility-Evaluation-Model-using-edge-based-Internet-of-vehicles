import numpy as np
from datetime import datetime

def calculate_context_value(n):
    return np.random.rand(n, n)

def calculate_message_quality(n):
    return np.random.rand(n, n)

def calculate_direct_trust(context_matrix, message_quality_matrix):
    b = 0.8  # Increased value to make direct trust calculation less clear
    c = 0.2  # Altered constant
    direct_trust = np.exp(-b * (1 - context_matrix) * (1 - message_quality_matrix))
    return direct_trust

def calculate_recommended_trust(direct_trust_matrix, history_reputation):
    return (1 - history_reputation) * direct_trust_matrix + history_reputation

def calculate_trust_value(direct_trust_matrix, recommended_trust_matrix, history_reputation):
    lam = 0.5  # Changed to make trust less reliable
    sigma = 0.8  # Altered value
    return lam * history_reputation + (1 - lam) * ((1 - sigma) * direct_trust_matrix + sigma * recommended_trust_matrix)

def calculate_reputation_value(trust_matrix):
    return np.mean(trust_matrix, axis=0)

def classify_vehicles(reputation_values, threshold=0.3):  # Lowered threshold to misclassify more
    CH = np.sum(reputation_values >= threshold)
    CM = np.sum(reputation_values < threshold)
    return CH, CM

def calculate_accuracy(CH, honest_vehicles, total_vehicles):
    if CH != honest_vehicles:
        Val = abs(CH - honest_vehicles)
        accuracy = (total_vehicles - Val) / total_vehicles * 100
    else:
        accuracy = 100
    return accuracy

def calculate_false_alarm_rate(CH, CM, honest_vehicles):
    Val = abs(CH - honest_vehicles)
    if (Val + CM) == 0:
        false_alarm_rate = 0
    else:
        false_alarm_rate = Val / (Val + CM) * 100
    return false_alarm_rate

def main():
    start_time = datetime.now()
    
    honest_vehicles = int(input("Enter number of honest vehicles: "))
    malicious_vehicles = int(input("Enter number of malicious vehicles: "))
    total_vehicles = honest_vehicles + malicious_vehicles
    
    context_matrix = calculate_context_value(total_vehicles)
    message_quality_matrix = calculate_message_quality(total_vehicles)
    
    direct_trust_matrix = calculate_direct_trust(context_matrix, message_quality_matrix)
    
    history_reputation = np.zeros(total_vehicles)
    history_reputation[:honest_vehicles] = np.random.uniform(0.4, 0.6, honest_vehicles)  # Narrow range
    history_reputation[honest_vehicles:] = np.random.uniform(0.2, 0.4, malicious_vehicles)  # Narrow range
    
    recommended_trust_matrix = calculate_recommended_trust(direct_trust_matrix, history_reputation)
    trust_matrix = calculate_trust_value(direct_trust_matrix, recommended_trust_matrix, history_reputation)
    
    reputation_values = calculate_reputation_value(trust_matrix)
    
    CH, CM = classify_vehicles(reputation_values)
    
    end_time = datetime.now()
    
    accuracy = calculate_accuracy(CH, honest_vehicles, total_vehicles)
    false_alarm_rate = calculate_false_alarm_rate(CH, CM, honest_vehicles)
    
    detection_time = end_time - start_time
    detection_time_seconds = detection_time.total_seconds()
    
    print("\nContext Matrix:\n", context_matrix)
    print("\nMessage Quality Matrix:\n", message_quality_matrix)
    print("\nDirect Trust Matrix (0 to 1):\n", direct_trust_matrix)
    print("\nRecommended Trust Matrix:\n", recommended_trust_matrix)
    print("\nTrust Matrix:\n", trust_matrix)
    print("\nReputation Values:\n", reputation_values)
    print(f"\nTotal Honest Vehicles: {honest_vehicles}")
    print(f"Total Malicious Vehicles: {malicious_vehicles}")
    print(f"\nCalculated Honest Vehicles (CH): {CH}")
    print(f"Calculated Malicious Vehicles (CM): {CM}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"False Alarm Rate: {false_alarm_rate:.2f}%")
    print(f"Detection Time: {detection_time_seconds:.2f} seconds")

if __name__ == "__main__":
    main()
