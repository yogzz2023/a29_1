import numpy as np
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

# Function to read predicted values from CSV file
def read_predicted_values_from_csv(file_path):
    predicted_values = {'range': [], 'azimuth': [], 'elevation': [], 'time': []}
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            # Check if all values in the row are non-empty
            if all(row):
                # Adjust column indices based on CSV file structure
                rng = float(row[1])  # Predicted range
                az = float(row[2])   # Predicted azimuth
                el = float(row[3])   # Predicted elevation
                time = float(row[5]) # Predicted time
                predicted_values['range'].append(rng)
                predicted_values['azimuth'].append(az)
                predicted_values['elevation'].append(el)
                predicted_values['time'].append(time)
    return predicted_values

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
predicted_csv_file_path = 'test2.csv'  # Provide the path to your predicted values CSV file

# Read measurements from CSV file
measurements = read_measurements_from_csv(csv_file_path)

# Read predicted values from CSV file
predicted_values = read_predicted_values_from_csv(predicted_csv_file_path)

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

# Plotting measured vs predicted values along with CSV predicted values
plt.figure(figsize=(18, 12))

plt.subplot(3, 1, 1)
plt.plot(measured_time, measured_range, label='Measured Range')
plt.plot(predicted_time, predicted_range, label='Predicted Range (KF)')
plt.plot(predicted_values['time'], predicted_values['range'], label='Predicted Range (CSV)')
plt.xlabel('Time')
plt.ylabel('Range')
plt.title('Measurement Range vs Predicted Range')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(measured_time, measured_azimuth, label='Measured Azimuth')
plt.plot(predicted_time, predicted_azimuth, label='Predicted Azimuth (KF)')
plt.plot(predicted_values['time'], predicted_values['azimuth'], label='Predicted Azimuth (CSV)', color='black')
plt.ylabel('Azimuth')
plt.title('Measurement Azimuth vs Predicted Azimuth')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(measured_time, measured_elevation, label='Measured Elevation')
plt.plot(predicted_time, predicted_elevation, label='Predicted Elevation (KF)')
plt.plot(predicted_values['time'], predicted_values['elevation'], label='Predicted Elevation (CSV)')
plt.ylabel('Elevation')
plt.title('Measurement Elevation vs Predicted Elevation')
plt.legend()

plt.tight_layout()
plt.show()
