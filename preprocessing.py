import pandas as pd
import numpy as np
import scipy
import ahrs
from numpy.linalg import norm
from scipy.signal import convolve

def correct_offset_and_bias(g_s, a_s, g_offset, acc_scale, acc_offset):
    g_s[:,:] = g_s[:,:] + g_offset
    a_s[:,:] = a_s[:,:]  * acc_scale + acc_offset

    return g_s, a_s


def merge_data_based_on_tab(tab, imu, smoothness, step):
    """
    Input: 
        imu: pd.DataFrame with columns [host_timestamp, arduino_timestamp, ax, ay, az, gx, gy, gz, temperature]
        tab: pd.DataFrame with columns [host_timestamp, x, y, z, in_range, touch, pressure, reset]
    Output: 
        pd.DataFrame containing imu and tab data 
    """
    print("merging the data sets...")

    # 1. detect for time frame whole dataframe 

    t_left, t_right = tab["host_timestamp"].iloc[[0, -1]]
    i_left, i_right = imu["host_timestamp"].iloc[[0, -1]]
    left = max(t_left, i_left) #later start
    right = min(t_right, i_right) #earlier ending

    # use tab data set as base to merge on
    df = tab[(tab["host_timestamp"] >= left) & (tab["host_timestamp"] <= right)].copy() 

    is_entering_range = df["in_range"].diff() == 1
    is_reset_in_range = (df["in_range"] == 1) & (df["reset"] == 1)
    is_start_of_segment = is_entering_range | is_reset_in_range
    
    # Number segments
    df["segment"] = is_start_of_segment.cumsum()
    
    # Ignore parts that are out-of-range
    df = df[df["in_range"] == 1]

    # collect all segments in this dictonary with key= segment(int), value= dataframe
    dfs = dict()
    segment = 0

    for _, df in df.groupby("segment"):

        data = pd.DataFrame()

        # 2. detect time frame for each segment

        mask = df["touch"] > 0.0
        if not mask.any():
            continue
        start, end = df.index[mask][[0, -1]] # mask hat für die gesamte colum true/false werte und start/end sind erste letzte true werte
        df = df.loc[start:end]

        imu_start = imu.loc[(imu['host_timestamp']-df["host_timestamp"].loc[start]).abs().argsort()[:1]].index[0]
        imu_end = imu.loc[(imu['host_timestamp']-df["host_timestamp"].loc[end]).abs().argsort()[:1]].index[0]

        imu_df = imu.iloc[imu_start:imu_end]

        # Compute distance
        x = df["x"].values
        y = df["y"].values
        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])
        d = np.sqrt(dx ** 2 + dy ** 2)
        t = np.cumsum(d)
        length = t[-1]

        # Fit cubic splines
        mask = d > 1e-5
        if mask.sum() <= 3:
            continue
        tck, _ = scipy.interpolate.splprep([x[mask], y[mask]], u=t[mask], s=smoothness)

        # Sample spline at regular (spatial) interval
        t_r = np.arange(0, length, step)
        x_r, y_r = scipy.interpolate.splev(t_r, tck, der=0)

        x_r = x_r.astype(np.float32)
        y_r = y_r.astype(np.float32)

        data["x"] = x_r
        data["y"] = y_r

        t_imu = imu_df["host_timestamp"] - np.min(imu_df["host_timestamp"])
        imu_time_span = np.linspace(0, np.max(t_imu), len(data))

        # interpolate imu data to the same time span but accoring to the imu time values
        for column in ['ax', 'ay', 'az', 'gx', 'gy', 'gz']:
            data[column] = np.interp(imu_time_span, t_imu, imu_df[column])

        data['t'] = np.interp(imu_time_span, t_imu, imu_df['arduino_timestamp'])
        data['t'] = (data['t'] - np.min(data['t'])) * 1e-3

        dfs[segment] = data
        segment = segment + 1

    df = pd.concat([df.assign(segment=k) for k,df in dfs.items()])

    return df


def add_xy_delta_values(df:pd.DataFrame, segments = False):
        df['dx'] = df['x'] - df['x'].shift(1)
        df['dy'] = df['y'] - df['y'].shift(1)
        # introduces NaN values in the beginning
        df['dx'] = df['dx'].fillna(0)
        df['dy'] = df['dy'].fillna(0)

        if segments:
            idx =  np.nonzero(np.diff(df.segment))[0] + 1
            df.loc[idx, 'dx'] = 0
            df.loc[idx, 'dx'] = 0

        return df

def quaternions(acc, gyr, madgwick, timecolumn):
        print("calculate the quaternion representations...")
        Q = np.zeros((len(timecolumn), 4))
        Q[0] = ahrs.common.orientation.acc2q(acc[0]) 
        #Q[0] = np.array([1.0, 0.0, 0.0, 0.0])
        for i in range(1, len(timecolumn)):
            madgwick.Dt = (timecolumn[i] - timecolumn[i - 1]) 
            Q[i] = madgwick.updateIMU(Q[i - 1], gyr[i], acc[i])

        return Q


def rotation_matrix_body_to_global(q):
    """
    Input:
        Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 

    Output:
        3x3 rotation matrix. 
        This rotation matrix converts a point in the local reference 
        frame to a point in the navigation reference frame.
    """
    # divide by norm so we have a unit quaternion
    q0 = q[0] / norm(q)
    q1 = q[1] / norm(q)
    q2 = q[2] / norm(q)
    q3 = q[3] / norm(q)

    R = np.zeros((3,3))
    
    R[0,0] = 1- 2 * (q2 * q2 + q3 * q3) 
    R[0,1] = 2 * (q1 * q2 - q0 * q3)
    R[0,2] = 2 * (q1 * q3 + q0 * q2)
    
    R[1,0] = 2 * (q1 * q2 + q0 * q3)
    R[1,1] = 1 - 2 * (q1 * q1 + q3 * q3) 
    R[1,2] = 2 * (q2 * q3 - q0 * q1)
    
    R[2,0] = 2 * (q1 * q3 - q0 * q2)
    R[2,1] = 2 * (q2 * q3 + q0 * q1)
    R[2,2] = 1 - 2 * (q1 * q1 + q2 * q2) 

    # inverse of R is body to global, matrix is orthogonal so transpose is equal to inverse                  
    return np.transpose(R)

def get_rotation_matrices(df):
     df['r'] = df.apply(lambda row: rotation_matrix_body_to_global([row.q0, row.q1, row.q2, row.q3]), axis=1)
     return df

def get_navigation_acc(df):
    df[['nav_ax', 'nav_ay', 'nav_az']] = df.apply(lambda row: pd.Series(row.r @ [row.ax, row.ay, row.az]), axis=1) 
    return df

def integrate_1d(t, dx, x0=0):
    (n,) = dx.shape
    x = np.zeros_like(dx)
    x[0] = x0
    for i in range(1, n):
        dt = (t[i] - t[i - 1]) 
        x[i] = (dx[i - 1] + dx[i]) / 2 * dt + x[i - 1]
        
    return x

def integrate(df: pd.DataFrame):
    T = df["t"].values
    vel_x = integrate_1d(T,df['nav_ax'])
    vel_y = integrate_1d(T,df['nav_ay'])
    vel_z = integrate_1d(T,df['nav_az'])

# use starting point tab as intial integration start
    pos_x = integrate_1d(T,vel_x, df.x[0])
    pos_y = integrate_1d(T,vel_y, df.y[0])
    pos_z = integrate_1d(T,vel_z, df.z[0])

    return vel_x, vel_y,vel_z,pos_x,pos_y,pos_z

def get_ith_segment(df: pd.DataFrame, i:int):
    if i in df["segment"].values :
        segment = df[df["segment"] == i]
        return segment
    else:
         print("The dataframe does not contain a segment {}".format(i))


def calculate_STE(df: pd.DataFrame, window_size= 3):
    window_size = window_size

    short_term_energy = []
    for s in set(df["segment"].values):
        segment = df[df["segment"] == s]

        # Calculate squared values for each dimension
        squared_acceleration_x = np.square(segment.ax)
        squared_acceleration_y = np.square(segment.ay)
        squared_acceleration_z = np.square(segment.az)

        acc_total = squared_acceleration_x + squared_acceleration_y + squared_acceleration_z

        padded_acceleration = np.pad(acc_total, (window_size - 1, 0), mode='edge')

        window = np.ones(window_size) 

        result = convolve(padded_acceleration, window, mode='same')[window_size-1:]
        short_term_energy.extend(result)


    return short_term_energy