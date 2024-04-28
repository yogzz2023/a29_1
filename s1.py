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
            rng1 = float(row[6])  # Measurement range (column 7)
            az = float(row[7])    # Measurement azimuth (column 8)
            el = float(row[8])    # Measurement elevation (column 9)
            time_str = row[11]    # Measurement time (column 11)
            time = float(time_str) if time_str else 0.0  # Convert to float if not empty, else use 0.0
            measurements.append((rng1, az, el, time))
    return measurements


# Create an instance of the CVFilter class
kalman_filter = CVFilter()

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
    sig_r = measurement[0]  # Range
    sig_a = measurement[1]  # Azimuth
    sig_e_sqr = measurement[2]  # Elevation
    e = measurement[2]  # Elevation from CSV file
    a = measurement[1]  # Azimuth from CSV file
    r = measurement[0]  # Range from CSV file

    # Initialize R matrix
    R = np.zeros((3, 3))  # Initialize R matrix

    R[0, 0] = sig_r * sig_r * np.cos(e) * np.cos(e) * np.sin(a) * np.sin(a) + r * r * np.cos(e) * np.cos(e) * np.cos(a) * np.cos(a) + sig_a * sig_a + r * r * np.sin(e) * np.sin(e) * np.sin(a) * np.sin(a) * sig_e_sqr
    R[1, 1] = sig_r * sig_r * np.cos(e) * np.cos(e) * np.cos(a) * np.cos(a) + r * r * np.cos(e) * np.cos(e) * np.sin(a) * np.sin(a) + sig_a * sig_a + r * r * np.sin(e) * np.sin(e) * np.cos(a) * np.cos(a) * sig_e_sqr
    R[2, 2] = sig_r * sig_r * np.cos(e) * np.cos(e) * np.sin(a) * np.sin(a) + r * r * np.cos(e) * np.cos(e) * np.cos(a) * np.cos(a) + sig_a * sig_a + r * r * np.sin(e) * np.sin(e) * np.sin(a) * np.sin(a) * sig_e_sqr

    # Assign R to the filter instance
    kalman_filter.R = R

    # Perform state estimation for the current measurement
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
