import pandas as pd
import numpy as np

class CVFilter:
    def __init__(self):
        self.Sf = np.zeros((6, 1))
        self.Pf = np.zeros((6, 6))
        self.Phi = np.zeros((6, 6))
        self.Sp = np.zeros((6, 1))
        self.Q = np.zeros((6, 6))
        self.K = np.zeros((6, 3))
        self.Inn = np.zeros((3, 1))
        self.S = np.zeros((3, 3))
        self.Pp = np.zeros((6, 6))
        self.R = np.zeros((3, 3))
        self.H = np.zeros((3, 6))
        self.Meas_Time = 0
        self.Filtered_Time = 0

    def initialize_filter_state_covariance(self, x, y, z, vx, vy, vz, time):
        self.Sf = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Filtered_Time = time
        for i in range(6):
            for j in range(6):
                self.Pf[i, j] = self.R[i % 3, j % 3]

    def predict_state_covariance(self, delt):
        self.Phi = np.eye(6)
        self.Phi[0, 3] = delt
        self.Phi[1, 4] = delt
        self.Phi[2, 5] = delt
        self.Sp = np.dot(self.Phi, self.Sf)
        self.predicted_Time = self.Filtered_Time + delt
        T_3 = (delt ** 3) / 3.0
        T_2 = (delt ** 2) / 2.0
        self.Q[0, 0] = T_3
        self.Q[1, 1] = T_3
        self.Q[2, 2] = T_3
        self.Q[0, 3] = T_2
        self.Q[1, 4] = T_2
        self.Q[2, 5] = T_2
        np.fill_diagonal(self.Q[3:, :], delt)
        self.Pp = np.dot(np.dot(self.Phi, self.Pf), self.Phi.T) + self.Q

    def filter_state_covariance(self, Z, plant_noise):
        self.Prev_Sf = self.Sf.copy()
        self.Prev_Filtered_Time = self.Filtered_Time
        self.S = self.R + np.dot(np.dot(self.H, self.Pp), self.H.T)
        self.K = np.dot(np.dot(self.Pp, self.H.T), np.linalg.inv(self.S))
        self.Inn = Z - np.dot(self.H, self.Sp)
        self.Sf = self.Sp + np.dot(self.K, self.Inn)
        self.Pf = np.dot((self.Inn - np.dot(self.K, self.H)), self.Pp)
        self.Filtered_Time = self.Meas_Time

# Read the CSV file
df = pd.read_csv("test2.csv")

# Extract measured range, azimuth, elevation, and time
measured_range = df.iloc[:, 6].values  # Assuming range is in column 7 (0-indexed)
measured_azimuth = df.iloc[:, 7].values  # Assuming azimuth is in column 8
measured_elevation = df.iloc[:, 8].values  # Assuming elevation is in column 9
measured_time = df.iloc[:, 10].values  # Assuming time is in column 11

# Initialize Kalman filter
filter = CVFilter()

predicted_range_list = []
predicted_azimuth_list = []
predicted_elevation_list = []
predicted_time_list = []

# Iterate over measured values
for i in range(len(df)):
    # Initialize filter state covariance with measured values
    filter.initialize_filter_state_covariance(
        x=measured_range[i],
        y=measured_azimuth[i],
        z=measured_elevation[i],
        vx=0.0,  # Initial velocity, assumed to be 0
        vy=0.0,
        vz=0.0,
        time=measured_time[i]
    )
    
    # Predict state covariance
    delt = 0.1  # Example time step, adjust as needed
    filter.predict_state_covariance(delt)
    
    # Example usage of measurement and plant noise
    Z = np.array([[measured_range[i]], [measured_azimuth[i]], [measured_elevation[i]]])
    plant_noise = np.eye(3)  # Example plant noise, adjust as needed
    
    # Filter state covariance
    filter.filter_state_covariance(Z, plant_noise)

    # Access predicted values
    predicted_range = filter.Sf[0, 0]
    predicted_azimuth = filter.Sf[1, 0]
    predicted_elevation = filter.Sf[2, 0]
    predicted_time = filter.Filtered_Time
    
    # Store predicted values
    predicted_range_list.append(predicted_range)
    predicted_azimuth_list.append(predicted_azimuth)
    predicted_elevation_list.append(predicted_elevation)
    predicted_time_list.append(predicted_time)

# Write predicted values to a new CSV file
predicted_df = pd.DataFrame({
    "Predicted_Range": predicted_range_list,
    "Predicted_Azimuth": predicted_azimuth_list,
    "Predicted_Elevation": predicted_elevation_list,
    "Predicted_Time": predicted_time_list
})

predicted_df.to_csv("predicted_values.csv", index=False)
