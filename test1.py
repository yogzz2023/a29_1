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

    def Initialize_Filter_state_covariance(self, x, y, z, vx, vy, vz, time, sig_r, sig_a, sig_e_sqr):
        self.Sf = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Filtered_Time = time

        # Initialize R matrix
        self.R[0, 0] = sig_r * sig_r * np.cos(e) * np.cos(e) * np.sin(a) * np.sin(a) + \
                       r * r * np.cos(e) * np.cos(e) * np.cos(a) * np.cos(a) + sig_a * sig_a + \
                       r * r * np.sin(e) * np.sin(e) * np.sin(a) * np.sin(a) * sig_e_sqr

        self.R[1, 1] = sig_r * sig_r * np.cos(e) * np.cos(e) * np.cos(a) * np.cos(a) + \
                       r * r * np.cos(e) * np.cos(e) * np.sin(a) * np.sin(a) + sig_a * sig_a + \
                       r * r * np.sin(e) * np.sin(e) * np.cos(a) * np.cos(a) * sig_e_sqr

        self.R[2, 2] = sig_r * sig_r * np.cos(e) * np.cos(e) * np.sin(a) * np.sin(a) + \
                       r * r * np.cos(e) * np.cos(e) * np.cos(a) * np.cos(a) + sig_a * sig_a + \
                       r * r * np.sin(e) * np.sin(e) * np.sin(a) * np.sin(a) * sig_e_sqr

    def predict_state_covariance(self, delt):
        Phi = np.eye(6)
        Phi[0, 3] = Phi[1, 4] = Phi[2, 5] = delt
        Sp = np.dot(Phi, self.Sf)
        self.predicted_Time = self.Filtered_Time + delt

        T_3 = (delt ** 3) / 3.0
        T_2 = (delt ** 2) / 2.0
        Q = np.eye(6)
        Q[:3, :3] *= T_3
        Q[:3, 3:] = Q[3:, :3] = T_2 * np.eye(3)
        Q[3:, 3:] *= delt
        Q *= self.plant_noise
        self.Pp = np.dot(np.dot(Phi, self.pf), Phi.T) + Q

    def filter_state_covariance(self, Z):
        Prev_Sf = self.Sf.copy()
        Prev_Filtered_Time = self.Filtered_Time

        S = self.R + np.dot(np.dot(self.H, self.Pp), self.H.T)
        K = np.dot(np.dot(self.Pp, self.H.T), np.linalg.inv(S))
        Inn = Z - np.dot(self.H, self.Sf)
        self.Sf = self.Sf + np.dot(K, Inn)
        self.pf = np.dot((np.eye(6) - np.dot(K, self.H)), self.Pp)
        self.Filtered_Time = self.Meas_Time

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
sig_r = 1.0  # Example value for sigma_r
sig_a = 1.0  # Example value for sigma_a
sig_e_sqr = 1.0  # Example value for sigma_e_sqr

# Initialize the filter with initial state estimates
kalman_filter.Initialize_Filter_state_covariance(x, y, z, vx, vy, vz, initial_time, sig_r, sig_a, sig_e_sqr)

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
    kalman_filter.predict_state_covariance(measurement[3] - kalman_filter.Meas_Time)
    kalman_filter.filter_state_covariance(np.array([[measurement[0]], [measurement[1]], [measurement[2]]]))
    
    # Append measured and predicted values to lists
    measured_range.append(measurement[0])
    measured_azimuth.append(measurement[1])
    measured_elevation.append(measurement[2])
    predicted_range.append(kalman_filter.Sf[0][0])
    predicted_azimuth.append(kalman_filter.Sf[1][0])
    predicted_elevation.append(kalman_filter.Sf[2][0])

    # Print the predicted azimuth value for each measurement
    print("Predicted Azimuth:", kalman_filter.Sf[1][0])

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
