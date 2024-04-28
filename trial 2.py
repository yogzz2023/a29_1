import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

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

# Convert measurements to numpy array for plotting
measurements_array = np.array(measurements)

print("Most likely associated measurement:", measurements[most_likely_index])


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
