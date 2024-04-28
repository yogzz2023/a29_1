import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.stats import multivariate_normal

class CVFilter:
    def __init__(self):
        # Initialize filter parameters
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.pf = np.eye(6)  # Filter state covariance matrix
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time

    def Initialize_Filter_state_covariance(self, x, y, z, vx, vy, vz, time):
        # Initialize filter state
        self.Sf = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = time

    def Filter_state_covariance(self, measurements, current_time):
        # Kalman filter implementation
        for measurement in measurements:
            M_rng1, M_az, M_el, M_time = measurement

            # Predict step
            dt = M_time - self.Meas_Time
            Phi = np.eye(6)
            Phi[0, 3] = dt
            Phi[1, 4] = dt
            Phi[2, 5] = dt
            Q = np.eye(6) * self.plant_noise
            self.Sf = np.dot(Phi, self.Sf)
            self.pf = np.dot(np.dot(Phi, self.pf), Phi.T) + Q

            # Update step
            Z = np.array([[M_rng1], [M_az], [M_el]])
            H = np.eye(3, 6)
            Inn = Z - np.dot(H, self.Sf)
            S = np.dot(H, np.dot(self.pf, H.T)) + self.R
            K = np.dot(np.dot(self.pf, H.T), np.linalg.inv(S))
            self.Sf = self.Sf + np.dot(K, Inn)
            self.pf = np.dot(np.eye(6) - np.dot(K, H), self.pf)

        self.Meas_Time = current_time  # Update measured time for the next iteration

        return self.Sf

class JPDA:
    def __init__(self, measurement_covariance):
        self.measurement_covariance = measurement_covariance

    def find_associated_measurement(self, target_measurement, measurements):
        # Convert target measurement to numpy array for convenience
        target = np.array(target_measurement)

        # List to store association probabilities
        association_probs = []

        # Calculate Mahalanobis distance and association probability for each measurement
        for measurement in measurements:
            # Convert measurement to numpy array and consider only the first three elements
            measurement = np.array(measurement)[:3]

            # Calculate Mahalanobis distance with small value added to covariance to avoid division by zero
            mahalanobis_dist = np.sqrt(np.sum(np.square(measurement - target) / (self.measurement_covariance[:3, :3] + 1e-6)))

            # Calculate association probability using multivariate normal distribution
            prob = multivariate_normal.pdf(mahalanobis_dist, mean=0, cov=1)

            # Append association probability to list
            association_probs.append(prob)

        # Find index of measurement with highest association probability
        max_prob_index = np.argmax(association_probs)

        # Return index of most likely associated measurement
        return max_prob_index

# Function to read measurements from CSV file
def read_measurements_from_csv(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            # Adjust column indices based on CSV file structure
            rng1 = float(row[7])  # Measurement range
            az = float(row[8])    # Measurement azimuth
            el = float(row[9])    # Measurement elevation
            time = float(row[11]) # Measurement time
            measurements.append((rng1, az, el, time))
    return measurements

# Create an instance of the CVFilter class
kalman_filter = CVFilter()

# Define initial state estimates
x = 0  # Initial x position
y = 0  # Initial y position
z = 0  # Initial z position
vx = 0  # Initial velocity in x direction
vy = 0  # Initial velocity in y direction
vz = 0  # Initial velocity in z direction
initial_time = 0  # Initial time

# Initialize the filter with initial state estimates
kalman_filter.Initialize_Filter_state_covariance(x, y, z, vx, vy, vz, initial_time)

# Define the path to your CSV files containing measurements and predicted values
csv_file_path = 'test2.csv'  # Provide the path to your measurements CSV file

# Read measurements from CSV file
measurements = read_measurements_from_csv(csv_file_path)

# Define measurement covariance matrix (assuming identity matrix for simplicity)
measurement_covariance = np.eye(3)

# Create an instance of the JPDA class
jpda = JPDA(measurement_covariance)

# Lists to store measured and predicted values
measured_time = []
predicted_time = []
measured_range = []
predicted_range = []
measured_azimuth = []
predicted_azimuth = []
measured_elevation = []
predicted_elevation = []

# Process measurements and get predicted state estimates at each time step
for measurement in measurements:
    filtered_state = kalman_filter.Filter_state_covariance([measurement], measurement[3])
    
    # Append measured and predicted values to lists
    measured_time.append(measurement[3])
    predicted_time.append(measurement[3])  # Using the same time as predicted time
    measured_range.append(measurement[0])
    predicted_range.append(filtered_state[0][0])
    measured_azimuth.append(measurement[1])
    predicted_azimuth.append(filtered_state[1][0])
    measured_elevation.append(measurement[2])
    predicted_elevation.append(filtered_state[2][0])

    # Find the most likely associated measurement
    most_likely_index = jpda.find_associated_measurement(filtered_state[:3], measurements)
    
    print("Most likely associated measurement:", measurements[most_likely_index])

    
    # Plot the measurements and most likely associated target
    plt.figure(figsize=(10, 6))
    measurements_array = np.array(measurements)
    plt.scatter(measurements_array[:, 0], measurements_array[:, 1], label='Measurements', color='blue')
    plt.scatter(filtered_state[0][0], filtered_state[1][0], label='Target (KF)', color='red')
    plt.scatter(measurements_array[most_likely_index, 0], measurements_array[most_likely_index, 1], label='Most Likely Associated', color='green')
    plt.xlabel('Range')
    plt.ylabel('Azimuth')
    plt.title('Measurements and Most Likely Associated')
    plt.legend()
    plt.grid(True)
    plt.show()
