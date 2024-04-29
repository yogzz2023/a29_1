import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
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
            # Convert measurement to numpy array
            measurement = np.array(measurement)

            # Calculate Mahalanobis distance with small value added to covariance to avoid division by zero
            mahalanobis_dist = np.sqrt(np.sum(np.square(measurement - target) / (self.measurement_covariance + 1e-6)))

            # Calculate association probability using multivariate normal distribution
            prob = multivariate_normal.pdf(mahalanobis_dist, mean=0, cov=1)

            # Append association probability to list
            association_probs.append(prob)

        # Find index of measurement with highest association probability
        max_prob_index = np.argmax(association_probs)

        # Return index of most likely associated measurement
        return max_prob_index

# Read measurements from CSV file
measurements = []
with open('test2.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip the header row
    for row in reader:
        measurements.append([float(row[7]), float(row[8]), float(row[9])])


# Define target 1 measurements
target_measurement = [20000, 120.8, 10.5]

# Define measurement covariance matrix (assuming identity matrix for simplicity)
measurement_covariance = np.eye(3)

# Create an instance of the JPDA class
jpda = JPDA(measurement_covariance)

# Find the most likely associated measurement
most_likely_index = jpda.find_associated_measurement(target_measurement, measurements)

# Print the most likely associated measurement
print("Most likely associated measurement:", measurements[most_likely_index])

# Convert measurements to numpy array for plotting
measurements_array = np.array(measurements)

# Plot the measurements
plt.figure(figsize=(10, 6))
plt.scatter(measurements_array[:, 0], measurements_array[:, 1], label='Measurements', color='blue')
plt.scatter(target_measurement[0], target_measurement[1], label='Target', color='red')
plt.scatter(measurements_array[most_likely_index, 0], measurements_array[most_likely_index, 1], label='Most Likely Associated', color='green')
plt.xlabel('Range')
plt.ylabel('Azimuth')
plt.title('Measurements and Most Likely Associated')
plt.legend()
plt.grid(True)
plt.show()
