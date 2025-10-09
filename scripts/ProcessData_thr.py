import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import argparse

"""
Launch with: $ python ProcessData.py <dir_data> <outhistoname>
"""
bars = 6.5
temp = 4

def calculate_charge(Nphe, channel):
    
    # coefficients = {
    # "CH1": {"p0": 3.72e-08, "p1": -3.66e-09},
    # "CH2": {"p0": 3.69e-08, "p1": -4.41e-09},
    # "CH3": {"p0": 3.70e-08, "p1": -4.03e-09},
    # "CH4": {"p0": 4.55e-08, "p1": -2.99e-09},
    # }

    if temp == 2:
        # 2deg calculated at 8.5bar
        coefficients = {
        "CH1": {"p0": 6.16e-08, "p1": -1.00e-08},
        "CH2": {"p0": 7.15e-08, "p1": -7.84e-09},
        "CH3": {"p0": 7.05e-08, "p1": -4.92e-09},
        "CH4": {"p0": 5.93e-08, "p1": -3.74e-09},
        }
    if temp == 4:
        # 4deg calculated at 6.5bar
        coefficients = {
        "CH1": {"p0": 5.82e-08, "p1": -2.92e-09},
        "CH2": {"p0": 7.11e-08, "p1": -5.20e-09},
        "CH3": {"p0": 7.14e-08, "p1": -5.63e-09},
        "CH4": {"p0": 6.00e-08, "p1": -4.14e-09},
        }
    if temp == 9:
        # 9deg calculated at 7.5bar
        coefficients = {
        "CH1": {"p0": 6.17e-08, "p1": -9.48e-09},
        "CH2": {"p0": 7.55e-08, "p1": -1.22e-08},
        "CH3": {"p0": 7.39e-08, "p1": -9.40e-09},
        "CH4": {"p0": 6.47e-08, "p1": -1.25e-08},
        }

    # Calculate the average of p0 and p1 across all channels
    avg_p0 = sum(channel["p0"] for channel in coefficients.values()) / len(coefficients)
    avg_p1 = sum(channel["p1"] for channel in coefficients.values()) / len(coefficients)
    
    # Add CHSum with the averaged coefficients
    coefficients["CHSum"] = {"p0": avg_p0, "p1": avg_p1}

    # Ensure the channel is valid
    if channel not in coefficients:
        raise ValueError(f"Invalid channel '{channel}'. Available channels: {list(coefficients.keys())}")

    # Retrieve the coefficients for the channel
    p0 = coefficients[channel]["p0"]
    p1 = coefficients[channel]["p1"]

    # Calculate and return counts
    return p0 * Nphe + p1
    

def find_x_maximum(moving_avg,time_begin_filtered,thr):

    moving_avg=np.array(moving_avg)
    time_begin_filtered=np.array(time_begin_filtered)
    
    # Condition for the threshold
    above_threshold = moving_avg > thr
    
    # Split into contiguous regions where the condition is valid
    regions = np.split(np.arange(len(above_threshold)), np.where(~above_threshold)[0] + 1)
    
    # Filter valid regions
    valid_regions = [region for region in regions if len(region) > 0 and above_threshold[region[0]]]
    
    # Find the x corresponding to the maximum y in each region
    res = []
    for region in valid_regions:
        region_y = moving_avg[region]
        region_x = time_begin_filtered[region]
        max_index = np.argmax(region_y)
        res.append(region_x[max_index])

    return res


# Set up argument parser
parser = argparse.ArgumentParser(description="Process data and save histograms.")
parser.add_argument("directory_path", help="Directory containing CSV files")
parser.add_argument("outfile", help="Output pickle file to save results")

# Parse arguments
args = parser.parse_args()

# Get the directory path and output file from the arguments
directory_path = args.directory_path
outfile = args.outfile

files_in_directory = os.listdir(directory_path)
channels = ["CH1", "CH2", "CH3", "CH4", "CHSum"]

# Regex pattern to match the number before ".csv" in the filename
pattern = r"(\d+)(?=\.csv)"

# Function to extract the number before .csv and return it as an integer
def extract_number(file_name):
    match = re.search(pattern, file_name)
    if match:
        return int(match.group(0))
    return float('inf')  # In case no number is found, return a large value

files_in_directory = [a_file for a_file in files_in_directory if extract_number(a_file) < 25]

print(files_in_directory)

# Initialize histograms for each channel
bins = np.linspace(0, 10e-6, 1000)  # 300 bins between 0 and 4e-6
hist_counts = {channel: np.zeros(len(bins) - 1) for channel in channels}

"""
DEBUG
"""
#files_in_directory = ["20241213_Kr_8.5bar_Ar_6deg_yes_amp_auto_trigger_1.csv","20241213_Kr_8.5bar_Ar_6deg_yes_amp_auto_trigger_3.csv"]
#UNCOMMENT TO DEBUG

# Parameters for the sliding window

# window_size = 3*0.8e-6  # 1.5bars: tau = 0.8 microseconds
# window_size = 3*0.7e-6  # 2.5bars: tau = 0.7 microseconds
# window_size = 3*3.6e-6  # 3.5bars: tau = 3.6 microseconds
# window_size = 3*2.8e-6  # 4.5bars: tau = 2.8 microseconds
# window_size = 3*1.5e-6  # 5.5bars: tau = 1.5 microseconds
# window_size = 3*1.2e-6  # 6.5bars: tau = 1.2 microseconds
# window_size = 3*0.6e-6  # 7.5bars: tau = 0.6 microseconds
# window_size = 3*0.5e-6  # 8.5bars: tau = 0.5 microseconds

window_size = 8e-6  # 4 microseconds
step_size = 1e-6  # 0.5 microseconds
start_time = 0e-6  # Starting time
end_time = 80e-6  # End time (maximum time in the dataset)
        

# DCCounts = {"CH1":1.25e6*window_size, "CH2": 1.48e6*window_size, "CH3": 1.02e6*window_size, "CH4": 1.1e6*window_size}
if temp == 4:
    # 6.5bars at 4deg
    DCCounts = {"CH1": 9.48E+05*window_size 
                ,"CH2": 9.63E+05*window_size 
                ,"CH3": 7.79E+05*window_size 
                ,"CH4": 1.01E+06*window_size}
if temp == 9:
    # 7.5bars at 9deg
    DCCounts = {"CH1":1.14E+06*window_size
                ,"CH2": 1.12E+06*window_size
                ,"CH3": 9.23E+05*window_size
                ,"CH4": 1.19E+06*window_size}

# amp_factors = {'CH1': 535, 'CH2': 537, 'CH3': 536, 'CH4': 675, 'Ave' : 570.75 }
amp_factors = {'CH1': 348, 'CH2': 373, 'CH3': 347, 'CH4': 361, 'Ave' : 357.25}

DCCounts['CHSum'] = (DCCounts['CH1']/amp_factors['CH1'] + DCCounts['CH2']/amp_factors['CH2'] + DCCounts['CH3']/amp_factors['CH3'] + DCCounts['CH4']/amp_factors['CH4'])*amp_factors['Ave']

sDCCounts={}

sDCCounts['CH1'] = np.sqrt(DCCounts['CH1'])
sDCCounts['CH2'] = np.sqrt(DCCounts['CH2'])
sDCCounts['CH3'] = np.sqrt(DCCounts['CH3'])
sDCCounts['CH4'] = np.sqrt(DCCounts['CH4'])

sDCCounts['CHSum'] = np.sqrt( sDCCounts['CH1']**2+sDCCounts['CH2']**2+sDCCounts['CH3']**2+sDCCounts['CH4']**2 )

threshold = {}

for channel in channels:
    threshold[channel] = calculate_charge( DCCounts[channel]+3*sDCCounts[channel] , channel)
    print(threshold[channel])
    
print(threshold)
    
for file_name in files_in_directory:
    file_path = os.path.join(directory_path, file_name)
    print(file_name)
    
    df = pd.read_csv(file_path)  
    df = df.drop("Unnamed: 0",axis=1)

    events = df['event'].unique()

    """
    DEBUG
    """
    #events = list(range(20))
    #UNCOMMENT TO DEBUG

    
    for event_num in events:

        if(event_num%10==0):
            print(event_num)
        df_event = df[df['event']==event_num]

        # Initialize results dictionary
        results = {channel: {"time_begin": [], "integral_sum": []} for channel in channels}
        
        # Perform the sliding window computation for each channel
        for channel in channels:
            channel_data = df_event[df_event["channel"] == channel]
            
            # Iterate over sliding windows
            current_time = start_time
            while current_time + window_size <= end_time:
                # Filter data within the current window
                window_data = channel_data[
                    (channel_data["time"] >= current_time) & 
                    (channel_data["time"] < current_time + window_size)
                ]
                
                # Calculate the sum of integrals
                integral_sum = window_data["integral"].sum()
                
                # Store results
                results[channel]["time_begin"].append(current_time)
                results[channel]["integral_sum"].append(integral_sum)
                
                # Move the window to the right
                current_time += step_size

            # Apply a moving average filter with kernel size 7
            integral_sum = np.array(results[channel]["integral_sum"])
            moving_avg = np.convolve(integral_sum, np.ones(7) / 7, mode="valid")  # Moving average

            # Adjust time_begin to match the length of the filtered data
            time_begin_filtered = results[channel]["time_begin"][3:-3]  # Trim 3 elements from start and end

            x_max = find_x_maximum(moving_avg,time_begin_filtered,threshold[channel])

            integral_sums = []
            
            for j in range(len(x_max)):
                index = results[channel]['time_begin'].index(x_max[j])
                integral_sums.append(results[channel]['integral_sum'][index])
                
                
            counts, _ = np.histogram(integral_sums, bins=bins)
            hist_counts[channel] += counts


# Plot histograms for each channel
fig, axs = plt.subplots(len(channels), 1, figsize=(10, 12), sharex=True,dpi=300)

for i, channel in enumerate(channels):
    axs[i].bar(bins[:-1], hist_counts[channel], width=np.diff(bins)[0], align='edge', alpha=0.7)
    axs[i].set_title(f"{channel} Histogram")
    axs[i].set_ylabel("Counts")
    axs[i].grid(True)

axs[-1].set_xlabel("Integral Sum")

#plt.tight_layout()
#plt.show()

plt.savefig("../outputs/"+outfile+".png", dpi=300)

# Save the charge_sums data as a pickle file
with open("../outputs/"+outfile+".pkl", "wb") as f:
    pickle.dump(hist_counts, f)


    
