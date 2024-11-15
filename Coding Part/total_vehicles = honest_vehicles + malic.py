total_vehicles = honest_vehicles + malicious_vehices
    
def calculate_accuracy(CG,  num_genuine, total_vehicles):
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
