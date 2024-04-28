import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import csv

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
        # JPDA algorithm implementation
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

            # Print predicted state vector elements
            print("Predicted State Vector Elements:")
            print("F_x:", self.Sf[0][0])
            print("F_y:", self.Sf[1][0])
            print("F_z:", self.Sf[2][0])
            print("F_vx:", self.Sf[3][0])
            print("F_vy:", self.Sf[4][0])
            print("F_vz:", self.Sf[5][0])

            # Update step
            Z = np.array([[M_rng1], [M_az], [M_el]])
            H = np.eye(3, 6)
            Inn = Z - np.dot(H, self.Sf)
            S = np.dot(H, np.dot(self.pf, H.T)) + self.R
            K = np.dot(np.dot(self.pf, H.T), np.linalg.inv(S))
            self.Sf = self.Sf + np.dot(K, Inn)
            self.pf = np.dot(np.eye(6) - np.dot(K, H), self.pf)

        # Compute probabilities for each measurement being associated with each target
        conditional_probs = []
        for measurement in measurements:
            M_rng1, M_az, M_el, M_time = measurement
            Z = np.array([[M_rng1], [M_az], [M_el]])
            H = np.eye(3, 6)
            Inn = Z - np.dot(H, self.Sf)
            S = np.dot(H, np.dot(self.pf, H.T)) + self.R
            prob = multivariate_normal.pdf(Inn.flatten(), mean=None, cov=S)
            conditional_probs.append(prob)

        # Calculate marginal probabilities
        marginal_probs = np.prod(conditional_probs, axis=0)

        # Find the most likely association
        most_likely_index = np.argmax(marginal_probs)
        most_likely_measurement = measurements[most_likely_index]

        # Print states
        print("Filter state:", self.Sf.flatten())
        print("Filter state covariance:", self.pf)

        # Print predicted range, azimuth, elevation, and time
        predicted_range = self.Sf[0][0]
        predicted_azimuth = self.Sf[1][0]
        predicted_elevation = self.Sf[2][0]
        predicted_time = current_time  # Use current time as predicted time
        print("Predicted Range:", predicted_range)
        print("Predicted Azimuth:", predicted_azimuth)
        print("Predicted Elevation:", predicted_elevation)
        print("Predicted Time:", predicted_time)

        self.Meas_Time = current_time  # Update measured time for the next iteration

        return self.Sf, most_likely_measurement

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

# Define the path to your CSV file containing measurements
csv_file_path = 'test2.csv'  # Provide the path to your CSV file

# Read measurements from CSV file
measurements = read_measurements_from_csv(csv_file_path)

# Lists to store measured and predicted values
measured_range = []
measured_azimuth = []
measured_elevation = []
predicted_range = []
predicted_azimuth = []
predicted_elevation = []

# Process measurements and get predicted state estimates at each time step
for measurement in measurements:
    filtered_state, most_likely_measurement = kalman_filter.Filter_state_covariance([measurement], measurement[3])
    
    # Append measured and predicted values to lists
    measured_range.append(measurement[0])
    measured_azimuth.append(measurement[1])
    measured_elevation.append(measurement[2])
    predicted_range.append(filtered_state[0][0])
    predicted_azimuth.append(filtered_state[1][0])
    predicted_elevation.append(filtered_state[2][0])
    
    # Print the predicted azimuth value for each measurement
    print("Predicted Azimuth:", filtered_state[1][0])

# Plotting measured vs predicted values
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(measured_range, label='Measured Range')
plt.plot(predicted_range, label='Predicted Range')
plt.xlabel('Time Step')
plt.ylabel('Range')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(measured_azimuth, label='Measured Azimuth')
plt.plot(predicted_azimuth, label='Predicted Azimuth')
plt.xlabel('Time Step')
plt.ylabel('Azimuth')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(measured_elevation, label='Measured Elevation')
plt.plot(predicted_elevation, label='Predicted Elevation')
plt.xlabel('Time Step')
plt.ylabel('Elevation')
plt.legend()

plt.tight_layout()
plt.show()
