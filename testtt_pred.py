import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
            
            # Print predicted range, azimuth, and elevation
            print("Predicted Range:", self.Sf[0, 0])
            print("Predicted Azimuth:", self.Sf[1, 0])
            print("Predicted Elevation:", self.Sf[2, 0])

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

# Define target 1 measurements
target_measurement = [20000, 120.8, 10.5]

# Define measurements
measurements = [
    [20050.0, 120.75, 10.4],
    [20010.0, 119.05, 9.8],
    [19005.0, 119.5, 9.9],
    [20500.0, 121.0, 10.2],
    [19500.6, 120.95, 10.4]
]

# Define measurement covariance matrix (assuming identity matrix for simplicity)
measurement_covariance = np.eye(3)

# Create an instance of the JPDA class
jpda = JPDA(measurement_covariance)

# Find the most likely associated measurement
most_likely_index = jpda.find_associated_measurement(target_measurement, measurements)

# Print the most likely associated measurement
print("Most likely associated measurement:", measurements[most_likely_index])

# Create an instance of the CVFilter class
filter = CVFilter()

# Initialize filter state covariance
filter.Initialize_Filter_state_covariance(target_measurement[0], target_measurement[1], target_measurement[2], 0, 0, 0, 0)

# Run the filter and print predicted range, azimuth, and elevation
predicted_state = filter.Filter_state_covariance(measurements, 1)

# Convert measurements to numpy array for 3D plotting
measurements_array = np.array(measurements)

# Create a 3D figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the measurements
ax.scatter(measurements_array[:, 0], measurements_array[:, 1], measurements_array[:, 2], label='Measurements', color='blue')
ax.scatter(target_measurement[0], target_measurement[1], target_measurement[2], label='Target', color='red')
ax.scatter(measurements_array[most_likely_index, 0], measurements_array[most_likely_index, 1], measurements_array[most_likely_index, 2], label='Most Likely Associated', color='green')

# Set labels and title
ax.set_xlabel('Range')
ax.set_ylabel('Azimuth')
ax.set_zlabel('Elevation')
ax.set_title('3D Plot of Measurements and Most Likely Associated')

# Add legend
ax.legend()

# Show plot
plt.show()
