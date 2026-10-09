import sys
import os

import matplotlib.pyplot  as plt
import numpy              as np
import pandas             as pd
import tables             as tb
import scipy

from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
from scipy.special import gammaln

import re
import struct
import glob

font_size = 20
# Configure matplotlib to use Computer Modern fonts and mathtext
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = font_size

# **************************************************************************************************************************************************************************


# path = '/home/investigator/mariandbt/python/data'
path = '/scratch/marian/python/data'
path = path + '/cigar'
runs = {
        # 1.5: '20241011_with_am_1.5bar_N_14.0deg_no_amp', # only CH1
        # 2.5: '20241011_with_am_2.5bar_N_14.0deg_no_amp', 
        # 3.5: '20241011_with_am_3.5bar_N_14.0deg_no_amp', 
        # 4.5: '20241016_with_am_4.5bar_N_14.0deg_no_amp', 
        # 5.5: '20241018_with_am_5.5bar_Ar_14.0deg_no_amp', # only CH1
        # 6.5: '20241021_with_am_6.5bar_Ar_14.0deg_no_amp', 
        # 6.5: '20241022_with_am_6.5bar_Ar_14.0deg_no_amp', 
        # 6.5: '20241022_with_am_6.5bar_Ar_roomtemp_no_amp', 
        # 7.5: '20241023_with_am_7.5bar_Ar_roomtemp_no_amp', 
        # 7.5: '20241024_with_am_7.5bar_Ar_roomtemp_no_amp', 
        # 8.2: '20241028_with_am_8.2bar_Ar_roomtemp_no_amp', 
        # 8.5: '20241025_with_am_8.5bar_Ar_roomtemp_no_amp',
    
        8.5: '20241029_with_am_inside_8.5bar_Ar_roomtemp_no_amp',
        7.5: '20241030_with_am_inside_7.5bar_Ar_roomtemp_no_amp',
        6.5: '20241031_with_am_inside_6.5bar_Ar_roomtemp_no_amp',
        5.5: '20241104_with_am_inside_5.5bar_Ar_roomtemp_no_amp',
        # 5.5: '20241104_with_am_inside_5.5bar_Ar_roomtemp_no_amp_run2', # Hot getter mostly off 
        4.5: '20241105_with_am_inside_4.5bar_Ar_roomtemp_no_amp',
        3.5: '20241106_with_am_inside_3.5bar_Ar_roomtemp_no_amp',
        2.5: '20241107_with_am_inside_2.5bar_Ar_roomtemp_no_amp',
        1.5: '20241108_with_am_inside_1.5bar_Ar_roomtemp_no_amp',
        1.0: '20241111_with_am_inside_1.0bar_Ar_roomtemp_no_amp',
    
        # 8.5: '20241128_no_source_8.5bar_Ar_roomtemp_no_amp'
        # 8.5: '20241128_no_source_8.5bar_Ar_roomtemp_yes_amp'
        # 8.5: '20241128_Kr_8.5bar_Ar_roomtemp_yes_amp'
        # 2.5: '20241204_no_source_2.5bar_Ar_8deg_yes_amp_low_trigger'
        # 2.5: '20241204_no_source_2.5bar_Ar_10deg_yes_amp_low_trigger_only3channels'
        # 2.5: '20241205_Kr_2.5bar_Ar_10deg_yes_amp_low_trigger_only3channels',
        # 8.5: '20241205_Kr_8.5bar_Ar_6deg_yes_amp_only3channels'
        # 8.5: '20241210_no_source_8.5bar_Ar_8deg_yes_amp_only3channels'
        # 8.5: '20241211_no_source_8.5bar_Ar_8deg_yes_amp_only3channels'
        # 8.5: '20241211_Kr_8.5bar_Ar_6deg_yes_amp_only3channels'
        # 8.5: '20241212_Kr_8.5bar_Ar_6deg_yes_amp'
       }

runsSpe = {}
runsSpe['CH1'] = {
                  4.5: '20241016_with_am_4.5bar_Ar_14.0deg_single_photon_CH1_amp' # 3 files
                  }
runsSpe['CH2'] = {
                  4.5: '20241016_with_am_4.5bar_Ar_14.0deg_single_photon_CH2_amp' # 3 files
                  }
runsSpe['CH3'] = {
                  4.5: '20241016_with_am_4.5bar_Ar_14.0deg_single_photon_CH3_amp' # 3 files
                  }
runsSpe['CH4'] = {
                  4.5: '20241016_with_am_4.5bar_Ar_14.0deg_single_photon_CH4_amp' # 3 files
                  }


baseline_th = -0.2e-6 #[s]

def GetRun(bars):
    run = runs[bars]
    return run

def GetSpeRun(channel, bars):
    run = runsSpe[f'CH{channel}'][bars]
    return run

def read_directory(runs_directory, run, file_type = 'h5'):
    run_path = os.path.join(runs_directory, run) 

    # Use glob to find all .h5 files in the directory
    files_list = glob.glob(f"{run_path}/*.{file_type}")

    if file_type == 'h5':
        files_list = sorted(files_list, key=extract_number_h5)
    return files_list

def read_directory_csv(directory):
    csv_path = os.path.join(path, directory) 

    # Use glob to find all .csv files in the directory
    csv_files = glob.glob(f"{csv_path}/*.csv")
    return csv_files

def read_directory_txt(directory):
    txt_path = os.path.join(path, directory) 

    # Use glob to find all .csv files in the directory
    txt_files = glob.glob(f"{txt_path}/*.txt")
    return txt_files

def parse_wf_from_binary(filename):
    data_list = []
    channels = {}  # Dictionary to store {DAQ channel: Real channel}
    
    with open(filename, "rb") as f:
        while True:
            # Read the header
            data = f.read(4)  # Read uint32_t EVID
            if not data:
                break
            EVID = struct.unpack("<I", data)[0]

            data = f.read(8)  # Read uint64_t T
            if not data:
                break
            T = struct.unpack("<Q", data)[0]

            data = f.read(4)  # Read uint32_t size
            if not data:
                break
            size = struct.unpack("<I", data)[0]

            data = f.read(8)  # Read uint64_t sampl_time
            if not data:
                break
            sampl_time = struct.unpack("<Q", data)[0]

            data = f.read(4)  # Read uint32_t ch (number of channels)
            if not data:
                break
            ch = struct.unpack("<I", data)[0]

            waveform_data = {}

            # Read waveforms for each channel
            for _ in range(ch):
                data = f.read(2)  # Read uint16_t numch
                if not data:
                    break
                numch = struct.unpack("<H", data)[0]

                channel_waveforms = []
                for _ in range(size):
                    data = f.read(4)  # Read float w
                    if not data:
                        break
                    w = struct.unpack("<f", data)[0]
                    channel_waveforms.append(w)

                # Store waveform with proper mapping
                real_ch = (numch // 2) + 1  # Convert DAQ channel to real channel
                channels[numch] = real_ch
                waveform_data[real_ch] = channel_waveforms  # Use real channel number as key

            # Create a row per sample point with all channels aligned
            for i in range(size):
                row = {
                    "TIME": (i + 1) * sampl_time / 1e9,  # Convert to seconds
                    "event": EVID,
                    "event_time": T
                }
                
                # Assign waveform values to proper channels
                for real_ch in waveform_data:
                    row[f'CH{real_ch}'] = waveform_data[real_ch][i] / 1e3  # Convert to the same scale

                data_list.append(row)

    df = pd.DataFrame(data_list)
    return df, channels


def daqDecoder(file_path, file_type):
    data_list   = []
    channels    = {}  # Dictionary to store {DAQ channel: Real channel}

    if file_type == 'bin':
        with open(file_path, "rb") as f:
            while True:
                # Read the header
                data = f.read(4)  # Read uint32_t EVID
                if not data:
                    break
                EVID = struct.unpack("<I", data)[0]

                data = f.read(8)  # Read uint64_t T
                if not data:
                    break
                T = struct.unpack("<Q", data)[0]

                data = f.read(4)  # Read uint32_t size
                if not data:
                    break
                size = struct.unpack("<I", data)[0]

                data = f.read(8)  # Read uint64_t sampl_time
                if not data:
                    break
                sampl_time = struct.unpack("<Q", data)[0]

                data = f.read(4)  # Read uint32_t ch (number of channels)
                if not data:
                    break
                ch = struct.unpack("<I", data)[0]

                waveform_data = {}

                # Read waveforms for each channel
                for _ in range(ch):
                    data = f.read(2)  # Read uint16_t numch
                    if not data:
                        break
                    numch = struct.unpack("<H", data)[0]

                    channel_waveforms = []
                    for _ in range(size):
                        data = f.read(4)  # Read float w
                        if not data:
                            break
                        w = struct.unpack("<f", data)[0]
                        channel_waveforms.append(w)

                    # Store waveform with proper mapping
                    real_ch = (numch // 2) + 1  # Convert DAQ channel to real channel
                    channels[numch] = real_ch
                    waveform_data[real_ch] = channel_waveforms  # Use real channel number as key

                # Create a row per sample point with all channels aligned
                for i in range(size):
                    row = {
                        "TIME": (i + 1) * sampl_time / 1e9,  # Convert to seconds
                        "event": EVID,
                        "event_time": T
                    }
                    
                    # Assign waveform values to proper channels
                    for real_ch in waveform_data:
                        row[f'CH{real_ch}'] = waveform_data[real_ch][i] / 1e3  # Convert to the same scale

                    data_list.append(row)


    elif file_type == 'txt':
        with open(file_path, 'r') as f:
            lines = f.readlines()

        event_num = None
        event_time = None
        time_step = None
        headers = []

        for i, line in enumerate(lines):
            line = line.strip()

            # Capture Event number
            if line.startswith("Event n."):
                event_num_match = re.search(r"Event n\.\s*(\d+)", line)
                if event_num_match:
                    event_num = int(event_num_match.group(1))  # Capture the number after "Event n."
                    # print(f"🔹 Found Event number: {event_num}")
                else:
                    print("⚠ Warning: Event number not found.")

            # Capture TimeStamp
            elif line.startswith("TimeStamp:"):
                event_time_match = re.search(r"TimeStamp:\s*(\d+)", line)
                if event_time_match:
                    event_time = int(event_time_match.group(1))  # Capture the number after "TimeStamp:"
                    # print(f"🔹 Found Event time: {event_time}")
                else:
                    print("⚠ Warning: Event time not found.")

            # Capture Sample step value
            elif line.startswith("1 Sample ="):
                time_step_match = re.search(r'0\.[0-9]+', line)
                if time_step_match:
                    time_step = float(time_step_match.group(0))
                    # print(f"🔹 Found Time step: {time_step}")
                else:
                    print("⚠ Warning: Time step not found.")

            # Capture Headers (Channel names) using regex
            elif line.startswith("S") and "CH:" in line:
                headers = re.findall(r'CH:\s*(\d+)', line)  # Find all channels after "CH:"
                # print(f"📌 Headers detected: {headers}")

                # Map DAQ channels to real channels
                for daq_ch in headers:
                    daq_ch = int(daq_ch)  # Convert channel to integer
                    real_ch = (daq_ch // 2) + 1  # Convert DAQ channel to real channel
                    channels[daq_ch] = real_ch
                    # print(f"✅ Mapped DAQ CH {daq_ch} → Real CH {real_ch}")

            # Capture Data values
            elif re.match(r'^\d+\s+', line) and headers:
                values = line.split()
                row = {
                    "TIME": int(values[0]) * time_step / 1e6,  # Convert time to seconds
                    "event": event_num,  # Add event number
                    "event_time": event_time  # Add event time
                }

                for idx, daq_ch in enumerate(headers):
                    daq_ch = int(daq_ch)
                    real_ch = channels[daq_ch]  # Get real channel from map
                    row[f"CH{real_ch}"] = float(values[idx+1]) / 1000  # Assign the data

                # Append row to data
                data_list.append(row)

        # Check if we collected any data
        if not data_list:
            print("⚠ Warning: No data captured.")
        

    # Convert to DataFrame
    df = pd.DataFrame(data_list)

    return df, channels

# Sort the files based on the number that appears right before the '.h5' extension
def extract_number_h5(file):
    # Extract the number before '.h5' using regex
    match = re.search(r'_(\d+)\.h5$', file)  # Matches '_number.h5'
    return int(match.group(1)) if match else float('inf')  # Extract the number or return 'inf'



# Define wrapper for fitting
def crystalball_fit(x, A, beta, m, loc, scale, tail = 'left'):
    from scipy.stats import crystalball
    """
    Adjusted Crystal Ball function with a tail on the right.
    The mean (`loc`) remains positive and unaffected.
    """
    if tail == 'right':
        tail_x = -x + 2 * loc  # Reflect x-values around the mean (loc)
    if tail == 'left':
        tail_x = x  # If the tail is at the left it's okay

    return A*crystalball.pdf(tail_x, beta=beta, m=m, loc=loc, scale=scale)

def gaussian(x, A, mu, sigma):
    """Define a Gaussian function."""
    return A*(1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def poisson_continuous(x, A, mu):
    """
    Continuous generalization of the Poisson pmf (n! -> Gamma(x+1) via
    gammaln), so it can be fit to a histogram of a rescaled count variable
    (e.g. counts/time) that isn't integer-valued, not just raw counts.
    Support is x >= 0, matching the Poisson; returns 0 elsewhere.
    """
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    valid = x >= 0
    out[valid] = A * np.exp(x[valid] * np.log(mu) - mu - gammaln(x[valid] + 1))
    return out

def generalized_poisson_continuous(x, A, theta, lam):
    """
    Continuous generalization (n! -> Gamma(x+1)) of the Generalized Poisson
    / Borel-Poisson pmf (Consul & Jain 1973), used to model correlated
    noise (optical crosstalk / afterpulsing) in SiPM dark-count spectra
    (Vinogradov 2012): a Poisson(theta) number of primary triggers, each
    independently spawning a Borel(lam)-distributed crosstalk/afterpulse
    cascade. lam is the (mean) per-pulse crosstalk probability, 0 <= lam < 1
    for the process to stay subcritical (finite mean/variance); lam = 0
    recovers poisson_continuous(x, A, theta) exactly.

    Mean = theta / (1 - lam), Variance = theta / (1 - lam)**3, so the
    variance-to-mean ratio (excess noise factor) is 1 / (1 - lam)**2 > 1 -
    the distribution is super-Poissonian by construction, with the extra
    spread concentrated in the tail rather than symmetric broadening.
    Support is x >= 0; returns 0 elsewhere.
    """
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    valid = x >= 0
    xv = x[valid]
    shifted = theta + xv * lam   # always > 0 here since theta, lam, xv >= 0
    log_f = (np.log(A) + np.log(theta) + (xv - 1) * np.log(shifted)
             - shifted - gammaln(xv + 1))
    out[valid] = np.exp(log_f)
    return out

def sum_of_gaussians(x, *params):
    """
    Returns the sum of multiple Gaussian functions with normalization.
    
    Parameters:
    x : array-like
        Points at which to evaluate the function.
    *params : list
        Flattened list containing amplitudes, means, and standard deviations.
    
    Returns:
    result : array-like
        The sum of all normalized Gaussians evaluated at points x.
    """
    num_gaussians = len(params) // 3  # Each Gaussian has A, mu, sigma
    A = np.array(params[:num_gaussians])
    mu = np.array(params[num_gaussians:2*num_gaussians])
    sigma = np.array(params[2*num_gaussians:])

    result = np.zeros_like(x)
    for a, m, s in zip(A, mu, sigma):
        # gaussian = (a / (s * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - m) / s) ** 2)
        gaussian = a * np.exp(-0.5 * ((x - m) / s) ** 2)
        result += gaussian
    
    return result


def FingerSeeds(charge, n_peaks, hist_range=None, oversample=4, min_period_bins=4, plot=False, title=None):
    """
    Initial guesses for the finger-plot Gaussian means (pedestal, 1 PE, 2 PE, ...).

    The fingers are (nearly) equally spaced, so instead of hunting for each
    peak with find_peaks the whole histogram is used to find the spacing:

    1. HILL   - the histogram blurred so much that the fingers vanish: the
                smooth envelope the fingers sit on.
    2. RIPPLE - (smoothed histogram - hill) / sqrt(hill): what the fingers
                add on top of the hill, in units of statistical fluctuation.
                Crests (> 0) are fingers, also those that only appear as a
                shoulder in the raw histogram.
    3. GAIN   - a rough period from the ripple's FFT (halved if the spectrum
                also peaks clearly at half that period), then refined by sliding
                a comb mu_1 + k*G (1 PE, 2 PE, 3 PE) over the ripple and keeping
                the (mu_1, G) whose teeth land on the most ripple. The pedestal
                is left out on purpose: it sits ~0.8 G below 1 PE, not 1 G, and
                including it shrinks the gain on poorly resolved channels.
    4. NUDGE  - each PE tooth moves to a ripple crest within +-G/4 if there is
                one; otherwise it stays on the comb. The pedestal is the
                ripple crest between 1.3 G and 0.4 G below 1 PE.

    plot=True draws these steps (hill, ripple + comb, nudges) in a new figure,
    shown at the next plt.show().

    Returns
    -------
    seeds : np.ndarray, shape (n_peaks,)
        Initial mu for each Gaussian, pedestal first.
    gain : float
        Estimated finger spacing (same units as charge).
    """
    q = np.asarray(charge)[np.isfinite(charge)]
    if hist_range is None:
        hist_range = (np.percentile(q, 0.5), np.percentile(q, 99))
    nb   = oversample * int(0.5 * np.sqrt(len(q)))
    h, b = np.histogram(q, bins=nb, range=hist_range)
    x    = 0.5 * (b[1:] + b[:-1])
    bw   = b[1] - b[0]

    # 1-2. hill and ripple
    hs     = gaussian_filter1d(h.astype(float), oversample * 0.75)
    hill   = gaussian_filter1d(h.astype(float), nb / 12)
    ripple = (hs - hill) / np.sqrt(np.maximum(hill, 1.0))
    ripple -= ripple.mean()

    # 3a. rough gain = dominant period of the ripple (zero-padded FFT)
    nfft  = 16 * nb
    P     = np.abs(np.fft.rfft(ripple * np.hanning(nb), nfft))**2
    per   = nfft / np.maximum(np.arange(len(P)), 1e-9)   # period in fine bins
    ok    = (per >= min_period_bins * oversample) & (per <= nb / 2.5)
    i0    = np.flatnonzero(ok)[np.argmax(P[ok])]
    # sub-harmonic check: on poorly resolved channels the big 1 PE bump next to the
    # dip on the pedestal side looks like one slow oscillation, so the FFT can lock
    # onto 2 G. If there is also a clear spectral peak near half that period, that
    # one is the real finger spacing (resolved channels show < 5% there).
    half  = np.flatnonzero(ok & (per > 0.4 * per[i0]) & (per < 0.6 * per[i0]))
    if len(half):
        ih = half[np.argmax(P[half])]
        if ih not in (half[0], half[-1]) and P[ih] > 0.25 * P[i0]:
            i0 = ih
    gain0 = per[i0] * bw

    # rough 1 PE position: FFT phase gives the comb teeth, the tooth nearest 0 is the pedestal
    phase   = np.angle(np.sum(ripple * np.exp(-2j * np.pi * (x - x[0]) / gain0)))
    tooth   = x[0] - phase / (2 * np.pi) * gain0
    mu_ped0 = tooth - gain0 * np.round(tooth / gain0)
    if mu_ped0 < x[0] - 0.5 * gain0:
        mu_ped0 += gain0
    mu1_0 = mu_ped0 + gain0

    # 3b. refine (mu_1, G) on the PE teeth only (1, 2, 3 PE), pedestal excluded
    n_teeth = int(np.clip(n_peaks - 1, 1, 3))
    G_grid  = gain0 * np.linspace(0.75, 1.35, 121)
    m_grid  = mu1_0 + gain0 * np.linspace(-0.5, 0.5, 101)
    k       = np.arange(n_teeth)
    teeth   = m_grid[:, None, None] + G_grid[None, :, None] * k
    score   = np.interp(teeth, x, ripple).sum(axis=2)
    im, ig  = np.unravel_index(np.argmax(score), score.shape)
    mu1, gain = m_grid[im], G_grid[ig]

    def crest(lo, hi, positive=True):
        """Highest ripple crest strictly inside (lo, hi), or None."""
        w = np.flatnonzero((x > lo) & (x < hi))
        if len(w) < 3:
            return None
        i = w[np.argmax(ripple[w])]
        if i in (w[0], w[-1]) or (positive and ripple[i] <= 0):
            return None
        return x[i]

    # 4. nudge each PE tooth to a nearby ripple crest; pedestal searched separately
    comb  = np.r_[mu1 - gain, mu1 + gain * np.arange(n_peaks - 1)]
    seeds = comb.copy()
    found = np.zeros(n_peaks, bool)
    for j in range(1, n_peaks):
        c = crest(comb[j] - gain / 4, comb[j] + gain / 4)
        if c is not None:
            seeds[j], found[j] = c, True
    # the pedestal is often a small bump on the rising side of the hill, so its
    # ripple crest can be negative: accept any local maximum there
    ped = crest(seeds[1] - 1.3 * gain, seeds[1] - 0.4 * gain, positive=False)
    seeds[0] = ped if ped is not None else max(seeds[1] - gain, x[0])

    # returned gain: straight line through the confirmed PE crests, when there are enough
    if found.sum() >= 2:
        g_fit = np.polyfit(np.flatnonzero(found), seeds[found], 1)[0]
        if 0.75 * gain < g_fit < 1.25 * gain:
            gain = g_fit

    if plot:
        fig, axs = plt.subplots(3, 1, figsize=(14, 13), sharex=True, dpi=100)
        if title:
            fig.suptitle(title)
        ax = axs[0]
        ax.plot(x, h, color='gray', alpha=0.4, label='histogram (fine bins)')
        ax.plot(x, hs, color='C0', lw=2, label='lightly smoothed')
        ax.plot(x, hill, color='C2', lw=4, label='hill (fingers blurred away)')
        ax.set_title('1. Hill: the smooth shape the fingers sit on')
        ax.set_ylabel('Counts')
        ax.legend(fontsize=14)

        ax = axs[1]
        ax.plot(x, ripple, color='k', lw=1.5)
        ax.fill_between(x, ripple, 0, where=ripple > 0, color='C1', alpha=0.4, label='crests = fingers')
        ax.fill_between(x, ripple, 0, where=ripple < 0, color='C0', alpha=0.2, label='valleys')
        ax.axhline(0, color='k', lw=0.8)
        for j, c in enumerate(comb):
            ax.axvline(c, color='red', ls='--', lw=1.5, label='comb teeth' if j == 0 else None)
            ax.text(c, ax.get_ylim()[1], f' {j}', color='red', va='top', fontsize=14)
        ax.set_title(f'2. Ripple = (smoothed - hill)/$\\sqrt{{hill}}$, comb with G = {gain:.3g}')
        ax.set_ylabel(r'Ripple [$\sigma$]')
        ax.legend(fontsize=14, loc='upper right')

        ax = axs[2]
        ax.plot(x, hs, color='C0', lw=2)
        y_comb = np.interp(comb, x, hs)
        y_seed = np.interp(seeds, x, hs)
        ax.plot(comb, y_comb, 'v', color='gray', ms=12, label='comb tooth')
        for c, yc, s, ys in zip(comb, y_comb, seeds, y_seed):
            if abs(s - c) > bw:
                ax.annotate('', xy=(s, ys), xytext=(c, yc),
                            arrowprops=dict(arrowstyle='->', color='k', lw=1.5))
        ax.plot(seeds, y_seed, 'o', color='red', ms=12, label='initial guess (seed)')
        ax.set_title('3. Nudge: each tooth moves onto a nearby crest (if there is one)')
        ax.set_ylabel('Counts')
        ax.set_xlabel('Charge')
        ax.legend(fontsize=14)
        plt.tight_layout()   # not shown here: appears at the caller's next plt.show()

    return seeds, gain

# def sum_of_gaussians(x, *params, bin_width=1.0, normalized=False):
#     """
#     Returns the sum of multiple Gaussian functions.

#     Parameters
#     ----------
#     x : array-like
#         Points at which to evaluate the function (e.g., bin centers).
#     *params : list
#         Flattened list containing amplitudes, means, and standard deviations:
#         [A1, A2, ..., mu1, mu2, ..., sigma1, sigma2, ...]
#     bin_width : float, optional
#         Histogram bin width. If fitting to raw counts (density=False in hist),
#         pass the bin width so that the Gaussian is scaled properly.
#         Default is 1.0 (no scaling).
#     normalized : bool, optional
#         If True, each Gaussian is normalized to have area = A.
#         If False, A is the peak amplitude at mu.
#         Default is False.

#     Returns
#     -------
#     result : array-like
#         The sum of all Gaussians evaluated at x.
#     """
#     num_gaussians = len(params) // 3  # Each Gaussian has A, mu, sigma
#     A = np.array(params[:num_gaussians])
#     mu = np.array(params[num_gaussians:2*num_gaussians])
#     sigma = np.array(params[2*num_gaussians:])

#     result = np.zeros_like(x, dtype=float)
#     for a, m, s in zip(A, mu, sigma):
#         if normalized:
#             # A = total area
#             gaussian = (a / (s * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - m) / s) ** 2)
#         else:
#             # A = peak height
#             gaussian = a * np.exp(-0.5 * ((x - m) / s) ** 2)
#         result += gaussian * bin_width

#     return result

def ResponseSiPM(q_in_pes, t, t0, rise_time, decay_time, baseline = 0):

    """
    NOTE: units of t, t0, rise_time and decay_time must be the same
    """
    rise_term   = 1 - np.exp(-(t - t0) / rise_time)
    decay_term  = np.exp(-(t - t0) / decay_time)

    signal          = (rise_term * decay_term)
    signal[t<t0]    = 0
    signal          = signal + baseline
    signal_area     = np.trapz(x = t, y = signal) or 1

    normalized_signal = q_in_pes*signal/signal_area

    return normalized_signal 


def ConvolvedResponseSiPM(t, mu, sigma, t0, rise, tau, wvf_area, baseline = 0):
    """Convolution between ResponseSiPM and a Gaussian."""
    dt = t[1] - t[0]  # Assuming uniform spacing in t
    response = ResponseSiPM(wvf_area, t, t0, rise, tau, baseline)
    gauss = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((t - mu) / sigma) ** 2)
    
    # Perform convolution
    convolved = np.convolve(response, gauss, mode='same')*dt
    return convolved

def double_exponential(t, t0, A0, tau1, A1, tau2, A2):
    
    double_exp = np.zeros_like(t)
    
    exp1 = np.exp(-(t[t>t0] - t0)/tau1)
    exp2 = np.exp(-(t[t>t0] - t0)/tau2)
    
    double_exp[t>t0] = (A1*exp1 + A2*exp2)
    
    double_exp_area        = np.trapz(x = t, y = double_exp) or 1
    normalized_double_exp  = A0*double_exp/double_exp_area

    return normalized_double_exp

def ArXeResponseNEW(t, t0, alpha, tau_s, tau_128, tau_150, r1, r2, r3, A, B, C, D):
    """
    Model for Am alpha decays in ArXe gas (SiPMs)
    
    Parameters:
    - t: Time array (µs).
    - alpha: Fraction of singlet states.
    - tau_s: Singlet lifetime (µs).
    - tau_128: Triplet lifetime (µs).
    - tau_150: ArXe* lifetime (µs).
    - r1, r2, r3: Process rates (µs⁻¹).
    - A, B1, C1: Amplitudes for direct, collisional, radiative components.
    
    Returns:
    - Signal (arbitrary units).
    """
    # Direct Ar₂* de-excitation (singlet + triplet)
    k1 = r1 + 1/tau_128
    f_128 = (alpha / tau_s) * np.exp(-(t[t>t0] - t0) / tau_s) + (1 - alpha) * k1 * np.exp(-k1 * (t[t>t0] - t0))
    
    # Collisional branch (ArXe* formation and decay/Xe₂* conversion)
    k3 = r3 + 1/tau_150
    f_coll = (k1 * k3 / (k1 - k3)) * (np.exp(-k3 * (t[t>t0] - t0)) - np.exp(-k1 * (t[t>t0] - t0)))
    
    # Radiative branch (EUV-induced ArXe*)
    f_rad = (r2 * k3 / (r2 - k3)) * (np.exp(-k3 * (t[t>t0] - t0)) - np.exp(-r2 * (t[t>t0] - t0)))
    
    # Total signal (PDEs omitted since amplitudes are free)
    signal          = np.zeros_like(t) + D
    signal[t>t0]    = A * f_128 + B * f_coll + C * f_rad + D
    
    return signal

def ArXeResponseComponents(t, t0, tau_128, tau_c, tau_r, r1, r2, r3, A, B, C):
    """
    Model for Am alpha decays in ArXe gas (SiPMs)
    
    Parameters:
    - t: Time array (s).
    - t0: Trigger time (s)
    - tau_128: Triplet lifetime, fast component (s).
    - tau_c: ArXe* lifetime for collisional branch, slow component (s).
    - tau_r: ArXe* lifetime for radiative branch, super-slow component (s).
    - r1, r2, r3: Process risings (s).
    - A, B, C: Amplitudes for direct, collisional, radiative components.
    - D: Baseline
    
    Returns:
    - Signal (arbitrary units).
    """
    
    direct          = np.zeros_like(t)
    rise_term       = 1 - np.exp(-(t[t>t0] - (t0))/r1)
    decay_term      = np.exp(-(t[t>t0] - (t0))/tau_128)
    direct[t>t0]    = (rise_term * decay_term)

    collisional         = np.zeros_like(t)
    rise_term           = 1 - np.exp(-(t[t>t0] - (t0))/r2)
    decay_term          = np.exp(-(t[t>t0] - (t0))/tau_c)
    collisional[t>t0]   = (rise_term * decay_term)

    radiative           = np.zeros_like(t)
    rise_term           = 1 - np.exp(-(t[t>t0] - (t0))/r3)
    decay_term          = np.exp(-(t[t>t0] - (t0))/tau_r)
    radiative[t>t0]     = (rise_term * decay_term)
    
    return (A/tau_128)*direct, (B/tau_c)*collisional, (C/tau_r)*radiative

def ArXeResponse(t, t0, tau_128, tau_c, tau_r, r1, r2, r3, A, B, C, D):
    """
    Model for Am alpha decays in ArXe gas (SiPMs)
    
    Parameters:
    - t: Time array (s).
    - t0: Trigger time (s)
    - tau_128: Triplet lifetime, fast component (s).
    - tau_c: ArXe* lifetime for collisional branch, slow component (s).
    - tau_r: ArXe* lifetime for radiative branch, super-slow component (s).
    - r1, r2, r3: Process risings (s).
    - A, B, C: Amplitudes for direct, collisional, radiative components.
    - D: Baseline
    
    Returns:
    - Signal (arbitrary units).
    """
    f_direct, f_collisional, f_radiative = ArXeResponseComponents(t, t0, tau_128, tau_c, tau_r, r1, r2, r3, A, B, C)

    signal    = f_direct + f_collisional + f_radiative  + D

    return signal

def ArResponse(t, t0, A, B, C, tau_fast, rise_slow, tau_slow):
    
    signal      = np.zeros_like(t)
    
    exp_fast    = np.exp(-(t[t>t0] - t0)/tau_fast)
    # fast_area   = np.trapz(x = t[t>t0], y = exp_fast) or 1
    # exp_fast    = exp_fast/fast_area
    # print(fast_area)

    rise_term   = 1 - np.exp(-(t[t>t0] - (t0))/rise_slow)
    decay_term  = np.exp(-(t[t>t0] - (t0))/tau_slow)
    exp_slow    = (rise_term * decay_term)
    # slow_area   = np.trapz(x = t[t>t0], y = exp_slow) or 1
    # exp_slow    = exp_slow/slow_area
    # print(slow_area)

    signal[t>t0]    = (A/tau_fast)*exp_fast + (B/tau_slow)*exp_slow + C
    # signal_area         = np.trapz(x = t, y = signal) or 1
    # print(signal_area)
    # normalized_signal   = A*signal/signal_area

    return signal

def exponential(t, t0, tau, A):

    exp = np.zeros_like(t)

    # exponential function
    exp[t>t0] = np.exp(-t[t>t0] / tau)  # Example of an exponential decay

    exp_area        = np.trapz(x = t, y = exp) or 1
    normalized_exp  = A*exp/exp_area

    return normalized_exp
    

def cigar_response(t, A, t0, rise, tau, A0, tau1, A1, tau2, A2, tau3, A3):
    """Convolution between ResponseSiPM and a double exponential."""
    dt = t[1] - t[0]  # Assuming uniform spacing in t
    
    # effect of the sensor
    response = ResponseSiPM(A, t, t0, rise, tau)
    
    # effect of the gas
    double_exp = double_exponential(t, 0, A0, tau1, A1, tau2, A2)

    # effect of the fiber
    exp = exponential(t, t0/2, tau3, A3)
    
    # First convolution
    convolved = np.convolve(response, exp, mode='same')*dt

    # Second convolution
    final_convolved = np.convolve(convolved, double_exp, mode='same') * dt
    
    return final_convolved

def cigar_response_fix_sipm(t, A0, tau1, A1, tau2, A2, tau3, A3):
    """Response of cigar with a fixed response signal from the SiPM"""
    A       = 1
    t0      = t.mean() # [s]
    rise    = 12e-9 # [s]
    # tau     = 100e-9 # [s]
    # tau     = 28e-9 # [s]
    # tau     = 30e-9 # [s]
    tau     = 40e-9 # [s]
    return cigar_response(t, A, t0, rise, tau, A0, tau1, A1, tau2, A2, tau3, A3)

def cigar_response_fix_sipm_decay_only(t, A0, tau1, A1, tau2, A2, tau3, A3, t_peak, peak_value):
    """
    - Generates response using an extended time axis only for the decay.
    - Aligns the peak using np.roll().
    - Ensures smooth decay by keeping only t >= t_peak.
    """
    # Create extended time axis just for decay (using t[0] and t[-1])
    dt = t[1] - t[0]  # Time step
    t_extended = np.arange(t[0], t[-1] + 1e-6, dt)  # Extend slightly beyond t_max for smooth tail
    
    # Generate response on extended time axis
    response_full = cigar_response_fix_sipm(t_extended, A0, tau1, A1, tau2, A2, tau3, A3)

    # Normalize amplitude to match data peak
    response_scaled = response_full* peak_value / response_full.max()

    # Find the peak in the generated waveform
    peak_index_model    = np.argmax(response_scaled)
    peak_index_data     = np.argmax(t_extended >= t_peak)  # Find index corresponding to t_peak

    # Compute the index shift
    shift = peak_index_data - peak_index_model

    # Shift response using np.roll()
    response_shifted = np.roll(response_scaled, shift)


    # Only slice based on t >= t_peak
    mask_decay = (t_extended >= t_peak)
    response_shifted_slice = response_shifted[mask_decay]

    # Ensure the lengths match by trimming the decay region
    return response_shifted_slice[:len(t[t >= t_peak])]




def PeakCorrection(matrix):
    corrected_matrix = matrix
    
    inf_mask        = np.isinf(matrix)
    neg_inf_mask    = np.isneginf(matrix)

    corrected_matrix[inf_mask]      = matrix[np.isfinite(matrix)].max()
    corrected_matrix[neg_inf_mask]  = matrix[np.isfinite(matrix)].min()
    
    return corrected_matrix


def ShiftWaveformToPeak(t, matrix):
    
    matrix = PeakCorrection(matrix)
    
    # Create a mask for t < 1e-6
    mask = (t > -1e-6) & (t < 1e-6)  # Boolean mask
    
    # Find peak indices for each row using the mask
    peak_indices = np.argmax(np.where(mask, matrix, -np.inf), axis=1)  # Shape: (n_rows,)
    
    # Get the t values corresponding to the peaks
    t_peak = t[peak_indices]  # Shape: (n_rows,)
    
    # Compute the shifted t for all rows
    t_shifted = t - t_peak[:, np.newaxis]  # Shape: (n_rows, n_columns)
    
    # Perform vectorized interpolation
    # Broadcast t_shifted along axis 1 and interpolate using np.interp
    left_values = matrix[:, 0][:, np.newaxis]  # Left boundary values
    right_values = matrix[:, -1][:, np.newaxis]  # Right boundary values
    
    # Vectorized interpolation (manually apply np.interp across rows)
    aligned_matrix = np.empty_like(matrix)
    for i in range(matrix.shape[0]):
        aligned_matrix[i, :] = np.interp(
            t,                      # Target x-coordinates (1D)
            t_shifted[i, :],        # Source x-coordinates (1D for each row)
            matrix[i, :],           # Source y-values (1D for each row)
            left=left_values[i],    # Left boundary value for the row
            right=right_values[i]   # Right boundary value for the row
        )
    
    return aligned_matrix


# def ChargeToPes(charge_in_Vs, channel, temp = '8deg', amplified = False):

#     # integral is in V*us
#     CHAmp={
#     "CH1":535,
#     "CH2":537,
#     "CH3":684,
#     "CH4":675}

#     if temp == 'roomDAQ':
#         # Samuele's (20250220) RoomTemp
#         ConvPar={
#         "CH1":(6.81e-8,-1.26e-8), # V*s
#         "CH2":(7.06e-8,-1.56e-8), # V*s
#         "CH3":(6.36e-8,-1.23e-8), # V*s
#         "CH4":(6.53e-8,-1.38e-8)  # V*s
#         }

#     elif temp == 'room':
#         # Samuele's (20241025) RoomTemp
#         ConvPar={
#         "CH1":(2.75e-8,-6.32e-9), # V*s
#         "CH2":(3.18e-8,-5.60e-9), # V*s
#         "CH3":(3.58e-8,-3.49e-9), # V*s
#         "CH4":(3.87e-8,-4.68e-9)  # V*s
#         }
        
#     elif temp == '8deg':
#         # Samuele's (20241205) 8deg
#         ConvPar={
#         "CH1":(3.72e-8,-3.66e-9), # V*s
#         "CH2":(3.69e-8,-4.41e-9), # V*s
#         "CH3":(4.52e-8,-1.68e-9), # V*s
#         "CH4":(4.55e-8,-2.99e-9)  # V*s
#         }


#     elif temp == '14deg':
#     # Marian's (20241022) 14deg
#         ConvPar={
#         "CH1":(3.26e-8,-7.32e-9), # V*s
#         "CH2":(3.79e-8,-1.06e-8), # V*s
#         "CH3":(4.44e-8,-1.03e-8), # V*s
#         "CH4":(4.13e-8,-1.07e-8)  # V*s
#         }

#     else:
#         print('Sorry sweetie, we don\'t have callibration for that temperature yet :(')

#     if amplified:
#         integral = charge_in_Vs[f'charge_CH{channel}'].copy()
#     else:
#         integral = charge_in_Vs[f'charge_CH{channel}'].copy()*CHAmp[f'CH{channel}']

#     p0, p1 = ConvPar[f'CH{channel}']

#     photoelectrons = (integral - p1) / p0
#     return photoelectrons

def ChargeToPes(charge_in_Vs, channel, temp, gas, amplified = False, CHAmp=None, version='v2'):

    # # integral is in V*us
    # CHAmp={
    # "CH1":535,
    # "CH2":537,
    # "CH3":684,
    # "CH4":675}

    # CALLIBRATION TAKEN AMPLIFIED

    ConvPar = None

    if temp == 'room':
        if version == 'v1':
            # Samuele's (20250220) RoomTemp
            ConvPar={
            "CH1":(6.81e-8,-1.26e-8), # V*s
            "CH2":(7.06e-8,-1.56e-8), # V*s
            "CH3":(6.36e-8,-1.23e-8), # V*s
            "CH4":(6.53e-8,-1.38e-8)  # V*s
            }

        elif version == 'v2':
            # TODO: fill in the v2 calibration
            ConvPar = None

    # Runs 70-115 (Xe) *******************************************************************************
    if gas == 'Xe':
    
        if temp == '8deg':
            # WITH AMPLIFICATION
            # 8degs measured at 8.5bar
            if version == 'v1':
                # Run73 in mV*s
                ConvPar={
                "CH1":(5.08e-5,-3.46e-5), # mV*s
                "CH2":(6.68e-5,-3.47e-5), # mV*s
                "CH3":(6.66e-5,-4.75e-5), # mV*s
                "CH4":(4.84e-5,-1.84e-5)  # mV*s
                }

            elif version == 'v2':
                # Run73 in mV*s
                # Per-run baseline correction + removing pedestal
                ConvPar={
                "CH1":(6.06e-5,-3.30e-6), # mV*s
                "CH2":(7.38e-5,-6.74e-6), # mV*s
                "CH3":(7.39e-5,-3.04e-6), # mV*s
                "CH4":(5.59e-5,-3.82e-6)  # mV*s
                }

        elif temp == '10deg':
            # WITH AMPLIFICATION
            # 10degs measured at 7.5bar
            if version == 'v1':
                # Run76 in mV*s
                ConvPar={
                "CH1":(5.21e-5,-3.84e-5), # mV*s
                "CH2":(6.70e-5,-3.70e-5), # mV*s
                "CH3":(6.59e-5,-4.50e-5), # mV*s
                "CH4":(4.90e-5,-1.90e-5)  # mV*s
                }

            elif version == 'v2':
                # Run76 in mV*s
                # Per-run baseline correction + removing pedestal
                ConvPar={
                "CH1":(5.94e-5,-3.61e-7), # mV*s
                "CH2":(7.74e-5,-1.15e-5), # mV*s
                "CH3":(8.23e-5,-1.34e-5), # mV*s
                "CH4":(5.84e-5,-2.72e-6)  # mV*s
                }

        elif temp == '11.5deg':
            # WITH AMPLIFICATION
            # 11.5degs measured at 4.5bar
            if version == 'v1':
                # Run104 in mV*s
                ConvPar={
                "CH1":(5.63e-5,1.83e-5), # mV*s
                "CH2":(7.18e-5,1.97e-5), # mV*s
                "CH3":(7.27e-5,1.22e-5), # mV*s
                "CH4":(5.23e-5,1.45e-5)  # mV*s
                }

            elif version == 'v2':
                # Run104 in mV*s
                # Per-run baseline correction + removing pedestal
                ConvPar={
                "CH1":(6.33e-5,-2.97e-6), # mV*s
                "CH2":(7.25e-5,5.12e-6), # mV*s
                "CH3":(7.86e-5,-3.56e-6), # mV*s
                "CH4":(5.95e-5,-5.34e-6)  # mV*s
                }

        elif temp == '12deg':
            # WITH AMPLIFICATION
            # 12degs measured at 6.5bar
            if version == 'v1':
                # Run81 in mV*s
                ConvPar={
                "CH1":(5.29e-5,-4.16e-5), # mV*s
                "CH2":(6.47e-5,-2.68e-5), # mV*s
                "CH3":(7.36e-5,1.04e-5), # mV*s
                "CH4":(5.71e-5,6.24e-5)  # mV*s
                }

            elif version == 'v2':
                # Run91 in mV*s
                # Per-run baseline correction + removing pedestal
                ConvPar={
                "CH1":(6.62e-5,-5.89e-6), # mV*s
                "CH2":(7.94e-5,-9.21e-6), # mV*s
                "CH3":(7.80e-5,-7.94e-9), # mV*s
                "CH4":(6.62e-5,-1.53e-5)  # mV*s
                }

        elif temp == '13deg':
            # WITH AMPLIFICATION
            # 13degs measured at 5.5bar
            if version == 'v1':
                # Run84 in mV*s
                ConvPar={
                "CH1":(5.60e-5,3.37e-6), # mV*s
                "CH2":(7.10e-5,1.99e-5), # mV*s
                "CH3":(7.60e-5,-5.34e-6), # mV*s
                "CH4":(5.04e-5,1.79e-5)  # mV*s
                }

            elif version == 'v2':
                # Run84 in mV*s
                # Per-run baseline correction + removing pedestal
                ConvPar={
                "CH1":(6.55e-5,-5.76e-6), # mV*s
                "CH2":(7.79e-5,-5.61e-6), # mV*s
                "CH3":(8.14e-5,-5.47e-6), # mV*s
                "CH4":(6.57e-5,-1.23e-5)  # mV*s
                }

        elif temp == '13v2deg':
            # WITH AMPLIFICATION
            # 13degs measured at 3.5bar
            if version == 'v1':
                # Run107 in mV*s
                ConvPar={
                "CH1":(5.72e-5,-2.24e-6), # mV*s
                "CH2":(7.01e-5,1.82e-5), # mV*s
                "CH3":(7.33e-5,-3.05e-6), # mV*s
                "CH4":(5.15e-5,1.41e-5)  # mV*s
                }

            elif version == 'v2':
                # Run107 in mV*s
                # Per-run baseline correction + removing pedestal
                ConvPar={
                "CH1":(6.19e-5,8.40e-7), # mV*s
                "CH2":(7.68e-5,-3.65e-6), # mV*s
                "CH3":(8.24e-5,-5.52e-6), # mV*s
                "CH4":(5.77e-5,3.33e-6)  # mV*s
                }

        elif temp == '13v3deg':
            # WITH AMPLIFICATION
            # 13degs measured at 2.5bar
            if version == 'v1':
                # Run109 in mV*s
                ConvPar={
                "CH1":(5.60e-5,2.89e-7), # mV*s
                "CH2":(6.61e-5,-3.61e-5), # mV*s
                "CH3":(4.92e-5,1.99e-5), # mV*s
                "CH4":(7.12e-5,-2.23e-6)  # mV*s
                }

            elif version == 'v2':
                # Run109 in mV*s
                # Per-run baseline correction + removing pedestal
                ConvPar={
                "CH1":(6.21e-5,2.89e-6), # mV*s
                "CH2":(7.58e-5,-3.75e-6), # mV*s
                "CH3":(6.89e-5,2.13e-5), # mV*s
                "CH4":(5.60e-5,-5.61e-6)  # mV*s
                }

        elif temp == '14deg':
            # WITH AMPLIFICATION
            # 14degs measured at 1.5bar
            if version == 'v1':
                # Run115 in mV*s
                ConvPar={
                "CH1":(5.93e-5,-1.14e-5), # mV*s
                "CH2":(6.64e-5,-4.25e-5), # mV*s
                "CH3":(4.96e-5,1.74e-5), # mV*s
                "CH4":(7.33e-5,-1.25e-5)  # mV*s
                }

            elif version == 'v2':
                # Run115 in mV*s
                # Per-run baseline correction + removing pedestal
                ConvPar={
                "CH1":(6.05e-5,-1.14e-7), # mV*s
                "CH2":(7.29e-5,3.58e-6), # mV*s
                "CH3":(5.91e-5,-9.78e-7), # mV*s
                "CH4":(7.89e-5,-6.35e-6)  # mV*s
                }


    # Runs 70-115 (Xe) *******************************************************************************


    # Runs 172-186 (Ar) *******************************************************************************
    elif gas == 'Ar':
        if temp == '6deg':
            # WITH AMPLIFICATION
            # 6degs measured at 8.5bar
            if version == 'v1':
                # Run172 in mV*s
                ConvPar={
                "CH1":(6.05e-5,-3.70e-5), # mV*s
                "CH2":(7.28e-5,-2.37e-5), # mV*s
                "CH3":(5.54e-5,-1.41e-5), # mV*s
                "CH4":(7.45e-5,-4.51e-5)  # mV*s
                }

            elif version == 'v2':
                # Run172 in mV*s
                # Per-run baseline correction + removing pedestal
                ConvPar={
                "CH1":(6.59e-5,2.05e-7), # mV*s
                "CH2":(7.91e-5,7.26e-7), # mV*s
                "CH3":(6.99e-5,-1.24e-5), # mV*s
                "CH4":(8.03e-5,9.75e-6)  # mV*s
                }

        elif temp == '7deg':
            # WITH AMPLIFICATION
            # 7degs measured at 7.5bar
            if version == 'v1':
                # Run174 in mV*s
                ConvPar={
                "CH1":(6.11e-5,-3.92e-5), # mV*s
                "CH2":(7.23e-5,-2.43e-5), # mV*s
                "CH3":(5.34e-5,-1.02e-5), # mV*s
                "CH4":(7.53e-5,-4.85e-5)  # mV*s
                }

            elif version == 'v2':
                # Run174 in mV*s
                # Per-run baseline correction + removing pedestal
                ConvPar={
                "CH1":(6.88e-5,-2.93e-6), # mV*s
                "CH2":(8.06e-5,-3.61e-7), # mV*s
                "CH3":(6.39e-5,-1.87e-6), # mV*s
                "CH4":(8.50e-5,1.10e-6)  # mV*s
                }

        elif temp == '8deg':
            # WITH AMPLIFICATION
            # 8degs measured at 6.5bar
            if version == 'v1':
                # Run176 in mV*s
                ConvPar={
                "CH1":(6.27e-5,-4.16e-5), # mV*s
                "CH2":(7.38e-5,-2.79e-5), # mV*s
                "CH3":(5.52e-5,-1.26e-5), # mV*s
                "CH4":(7.49e-5,-4.83e-5)  # mV*s
                }

            elif version == 'v2':
                # Run176 in mV*s
                # Per-run baseline correction + removing pedestal
                ConvPar={
                "CH1":(6.93e-5,1.56e-7), # mV*s
                "CH2":(8.22e-5,-7.20e-7), # mV*s
                "CH3":(6.44e-5,3.62e-7), # mV*s
                "CH4":(8.71e-5,-2.67e-6)  # mV*s
                }

        elif temp == '9deg':
            # WITH AMPLIFICATION
            # 9degs measured at 5.5bar
            if version == 'v1':
                # Run178 in mV*s
                ConvPar={
                "CH1":(6.30e-5,-4.27e-5), # mV*s
                "CH2":(7.37e-5,-2.94e-5), # mV*s
                "CH3":(5.51e-5,-1.34e-5), # mV*s
                "CH4":(7.53e-5,-5.09e-5)  # mV*s
                }

            elif version == 'v2':
                # Run178 in mV*s
                # Per-run baseline correction + removing pedestal
                ConvPar={
                "CH1":(6.94e-5,1.53e-6), # mV*s
                "CH2":(8.08e-5,3.07e-6), # mV*s
                "CH3":(6.84e-5,-5.39e-6), # mV*s
                "CH4":(8.44e-5,4.34e-6)  # mV*s
                }

        elif temp == '10deg':
            # WITH AMPLIFICATION
            # 10degs measured at 4.5bar
            if version == 'v1':
                # Run180 in mV*s
                ConvPar={
                "CH1":(6.38e-5,-4.49e-5), # mV*s
                "CH2":(7.46e-5,-3.31e-5), # mV*s
                "CH3":(5.67e-5,-1.81e-5), # mV*s
                "CH4":(7.57e-5,-5.43e-5)  # mV*s
                }

            elif version == 'v2':
                # Run180 in mV*s
                # Per-run baseline correction + removing pedestal
                ConvPar={
                "CH1":(7.19e-5,-2.07e-6), # mV*s
                "CH2":(7.98e-5,6.32e-6), # mV*s
                "CH3":(6.76e-5,-1.62e-6), # mV*s
                "CH4":(8.49e-5,4.40e-6)  # mV*s
                }

        elif temp == '11deg':
            # WITH AMPLIFICATION
            # 11degs measured at 3.5bar
            if version == 'v1':
                # Run182 in mV*s
                ConvPar={
                "CH1":(6.47e-5,-4.94e-5), # mV*s
                "CH2":(7.48e-5,-3.54e-5), # mV*s
                "CH3":(6.03e-5,-2.66e-5), # mV*s
                "CH4":(7.71e-5,-5.90e-5)  # mV*s
                }

            elif version == 'v2':
                # Run182 in mV*s
                # Per-run baseline correction + removing pedestal
                ConvPar={
                "CH1":(6.95e-5,3.78e-6), # mV*s
                "CH2":(8.14e-5,3.30e-6), # mV*s
                "CH3":(6.71e-5,1.98e-6), # mV*s
                "CH4":(8.54e-5,5.22e-6)  # mV*s
                }

        elif temp == '12deg':
            # WITH AMPLIFICATION
            # 12degs measured at 2.5bar
            if version == 'v1':
                # Run184 in mV*s
                # Fix baseline + no pedestal removing
                ConvPar={
                "CH1":(6.44e-5,-4.88e-5), # mV*s
                "CH2":(7.47e-5,-3.75e-5), # mV*s
                "CH3":(5.63e-5,-1.99e-5), # mV*s
                "CH4":(7.54e-5,-5.84e-5)  # mV*s
                }

            elif version == 'v2':
                # Run184 in mV*s
                # Per-run baseline correction + removing pedestal
                ConvPar={
                "CH1":(7.02e-5,2.84e-6), # mV*s
                "CH2":(8.23e-5,2.39e-6), # mV*s
                "CH3":(6.75e-5,8.62e-8), # mV*s
                "CH4":(8.54e-5,3.83e-6)  # mV*s
                }

        elif temp == '13.5deg':
            # WITH AMPLIFICATION
            # 13.5degs measured at 1.5bar
            if version == 'v1':
                # Run186 in mV*s
                # Fix baseline + no pedestal removing
                ConvPar={
                "CH1":(6.45e-5,-5.38e-5), # mV*s
                "CH2":(7.38e-5,-4.20e-5), # mV*s
                "CH3":(6.01e-5,-3.09e-5), # mV*s
                "CH4":(7.46e-5,-6.20e-5)  # mV*s
                }

            elif version == 'v2':
                # Run186 in mV*s
                # Per-run baseline correction + removing pedestal
                ConvPar={
                "CH1":(7.14e-5,1.97e-6), # mV*s
                "CH2":(7.96e-5,6.49e-6), # mV*s
                "CH3":(6.78e-5,2.09e-6), # mV*s
                "CH4":(8.62e-5,4.86e-6)  # mV*s
                }
    # Runs 172-186 (Ar) *******************************************************************************



    # # Runs 118-131 (Ar) *******************************************************************************
    # elif temp == '6deg':
    #     # WITH AMPLIFICATION
    #     # 6degs measured at 8.5bar
    #     # Run118 in mV*s
    #         ConvPar={
    #         "CH1":(5.08e-5,5.52e-6), # mV*s
    #         "CH2":(6.14e-5,5.56e-6), # mV*s
    #         "CH3":(4.98e-5,9.28e-6), # mV*s
    #         "CH4":(6.7e-5,-8.48e-7)  # mV*s
    #         }

    # elif temp == '7deg':
    #     # WITH AMPLIFICATION
    #     # 7degs measured at 7.5bar
    #     # Run124 in mV*s
    #         ConvPar={
    #         "CH1":(6.27e-5,1.08e-5), # mV*s
    #         "CH2":(7.58e-5,5.93e-6), # mV*s
    #         "CH3":(6.25e-5,1.67e-5), # mV*s
    #         "CH4":(7.83e-5,-1.80e-7)  # mV*s
    #         }

    # elif temp == '8deg':
    #     # WITH AMPLIFICATION
    #     # 8degs measured at 6.5bar
    #     # Run128 in mV*s
    #         ConvPar={
    #         "CH1":(6.21e-5,1.43e-5), # mV*s
    #         "CH2":(7.63e-5,6.72e-6), # mV*s
    #         "CH3":(5.75e-5,1.91e-5), # mV*s
    #         "CH4":(7.77e-5,7.65e-7)  # mV*s
    #         }

    # elif temp == '9deg':
    #     # WITH AMPLIFICATION
    #     # 9degs measured at 5.5bar
    #     # Run131 in mV*s
    #         ConvPar={
    #         "CH1":(6.43e-5,7.62e-6), # mV*s
    #         "CH2":(7.57e-5,3.92e-6), # mV*s
    #         "CH3":(6.27e-5,1.30e-5), # mV*s
    #         "CH4":(7.76e-5,2.70e-7)  # mV*s
    #         }
    # # Runs 118-131 (Ar) *******************************************************************************


    # elif temp == '2deg':
    #     # 2degs measured at 8.5bars
    #         ConvPar={
    #         "CH1":(6.16e-8,-1.00e-8), # V*s
    #         "CH2":(7.15e-8,-7.84e-9), # V*s
    #         "CH3":(7.05e-8,-4.92e-8), # V*s
    #         "CH4":(5.93e-8,-3.74e-9)  # V*s
    #         }

    # elif temp == '4deg':
    #     # 4degs measured at 6.5bars
    #         ConvPar={
    #         "CH1":(5.82e-8,-2.92e-9), # V*s
    #         "CH2":(7.11e-8,-5.20e-9), # V*s
    #         "CH3":(7.14e-8,-5.63e-8), # V*s
    #         "CH4":(6.00e-8,-4.14e-9)  # V*s
    #         }

    # elif temp == '9deg':
    #     # 9degs measured at 7.5bars
    #         ConvPar={
    #         "CH1":(6.17e-8,-9.48e-9), # V*s
    #         "CH2":(7.55e-8,-1.22e-8), # V*s
    #         "CH3":(7.39e-8,-9.40e-9), # V*s
    #         "CH4":(6.47e-8,-1.25e-8)  # V*s
    #         }


    # elif temp == '10deg':
    #     # 10degs measured at atmospheric pressure
    #         ConvPar={
    #         "CH1":(7.14e-8,-1.38e-8), # V*s
    #         "CH2":(5.08e-8,-6.66e-9), # V*s
    #         "CH3":(4.93e-8,-5.81e-9), # V*s
    #         "CH4":(4.26e-8,-6.83e-9)  # V*s
    #         }

    # elif temp == '8deg':
    #     # WITH AMPLIFICATION
    #     # 8degs measured at 6.5bar
    #         ConvPar={
    #         "CH1":(4.43e-8,1.36e-8), # V*s
    #         "CH2":(6.08e-8,1.88e-9), # V*s
    #         "CH3":(5.88e-8,8.24e-9), # V*s
    #         "CH4":(4.59e-8,1.29e-8)  # V*s
    #         }


    # elif temp == '9deg':
    #     # WITH AMPLIFICATION
    #     # 9degs measured at 5.5bar
    #         ConvPar={
    #         "CH1":(4.54e-8,1.34e-8), # V*s
    #         "CH2":(6.21e-8,3.06e-10), # V*s
    #         "CH3":(5.98e-8,6.91e-9), # V*s
    #         "CH4":(4.77e-8,1.06e-8)  # V*s
    #         }

    # elif temp == '13deg':
    #     # 13degs measured at atmospheric pressure
    #         ConvPar={
    #         "CH1":(7.38e-8,-1.64e-8), # V*s
    #         "CH2":(5.21e-8,-7.85e-9), # V*s
    #         "CH3":(5.00e-8,-6.14e-9), # V*s
    #         "CH4":(4.36e-8,-6.72e-9)  # V*s
    #         }

    # elif temp == '16deg':
    #     # 16degs measured at atmospheric pressure
    #         ConvPar={
    #         "CH1":(7.37e-8,-1.41e-8), # V*s
    #         "CH2":(5.26e-8,-8.31e-9), # V*s
    #         "CH3":(5.03e-8,-6.29e-9), # V*s
    #         "CH4":(4.43e-8,-6.94e-9)  # V*s
    #         }

    # elif temp == '19deg':
    #     # 19degs measured at atmospheric pressure
    #         ConvPar={
    #         "CH1":(7.41e-8,-1.64e-8), # V*s
    #         "CH2":(5.26e-8,-8.90e-9), # V*s
    #         "CH3":(5.01e-8,-7.61e-9), # V*s
    #         "CH4":(4.45e-8,-7.99e-9)  # V*s
    #         }

    # elif temp == '22deg':
    #     # 22degs measured at atmospheric pressure
    #         ConvPar={
    #         "CH1":(7.38e-8,-1.71e-8), # V*s
    #         "CH2":(5.25e-8,-1.06e-8), # V*s
    #         "CH3":(4.99e-8,-8.83e-9), # V*s
    #         "CH4":(4.46e-8,-8.95e-9)  # V*s
    #         }

    # elif temp == '25deg':
    #     # 22degs measured at atmospheric pressure
    #         ConvPar={
    #         "CH1":(7.39e-8,-2.02e-8), # V*s
    #         "CH2":(5.19e-8,-1.18e-8), # V*s
    #         "CH3":(4.96e-8,-1.02e-8), # V*s
    #         "CH4":(4.50e-8,-1.14e-8)  # V*s
    #         }

            
    else:
        print('Sorry sweetie, we don\'t have callibration for that temperature yet :(')

    if ConvPar is None:
        raise ValueError(f'No calibration for gas={gas}, temp={temp}, version={version}')


    # Calculate averages 
    avg_p0 = sum(p[0] for p in ConvPar.values()) / len(ConvPar)
    avg_p1 = sum(p[1] for p in ConvPar.values()) / len(ConvPar)
    ConvPar['CHSum'] = (avg_p0, avg_p1)

    if amplified:
        integral = charge_in_Vs
    else:
        integral = charge_in_Vs*CHAmp[f'CH{channel}'][0] + CHAmp[f'CH{channel}'][1]

    p0, p1 = ConvPar[f'CH{channel}']

    photoelectrons = (integral - p1) / p0
    return photoelectrons

def _fmt_sci(x):
    # 6.97e-05 -> 6.97e-5, matching the existing calibration entries
    if not np.isfinite(x):
        return 'np.nan'
    m, e = f'{x:.2e}'.split('e')
    return f'{m}e{int(e)}'

def _WriteCalibBlock(func_name, conditions, body, file_path=None):
    """
    Replace the body of a nested `if <var> == '<value>':` branch inside
    function `func_name` of this file. `conditions` is the nesting path,
    e.g. [('gas', 'Ar'), ('temp', '12deg'), ('version', 'v2')]; `body` is a
    list of unindented lines, indented here to match the branch.
    Returns (old_lines, new_lines).
    """
    if file_path is None:
        file_path = os.path.abspath(__file__)

    with open(file_path) as f:
        lines = f.read().split('\n')

    def indent(line):
        return len(line) - len(line.lstrip())

    def find(pattern, start, end):
        rx = re.compile(pattern)
        for k in range(start, end):
            if rx.match(lines[k]):
                return k
        return None

    def block_end(k):
        # first non-blank, non-comment line at or below the indentation of line k
        level = indent(lines[k])
        for j in range(k + 1, len(lines)):
            s = lines[j].strip()
            if s and not s.startswith('#') and indent(lines[j]) <= level:
                return j
        return len(lines)

    start = find(rf'def {func_name}\(', 0, len(lines))
    end   = find(r'def ', start + 1, len(lines)) or len(lines)
    path  = []
    for var, value in conditions:
        path.append(f"{var} == '{value}'")
        line = find(rf"\s*(el)?if {var} == '{re.escape(value)}':", start, end)
        if line is None:
            raise ValueError(f"No `{' -> '.join(path)}` branch in {func_name}")
        start, end = line, block_end(line)

    # body = lines after the branch line up to the next statement at its level,
    # minus trailing blank lines / section comments that belong to what follows
    branch   = start
    body_end = end
    while body_end > branch + 1 and (not lines[body_end - 1].strip()
                                     or indent(lines[body_end - 1]) <= indent(lines[branch])):
        body_end -= 1

    pad = ' ' * (indent(lines[branch]) + 4)
    new = [pad + l for l in body]
    old = lines[branch + 1:body_end]
    lines[branch + 1:body_end] = new

    with open(file_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Updated {func_name} [{', '.join(v for _, v in conditions)}] in {file_path}")
    print('--- old ---\n' + '\n'.join(old))
    print('--- new ---\n' + '\n'.join(new))
    return old, new

def WriteConvPar(ConvPar, gas, temp, run, version='v2',
                 comment='Per-run baseline correction + removing pedestal',
                 file_path=None):
    """
    Write a {'CH1': (slope, intercept), ...} calibration into the
    `elif version == '<version>':` branch of ChargeToPes (in this same file),
    under the given gas and temp, in the same format as the existing entries.
    Whatever that branch held before (e.g. the `ConvPar = None` TODO) is replaced.

    Reload the module afterwards (importlib.reload(cig)) for ChargeToPes to see it.
    """
    chs  = list(ConvPar.keys())
    body = [f'# {run} in mV*s', f'# {comment}', 'ConvPar={']
    for k, ch in enumerate(chs):
        slope, intercept = ConvPar[ch]
        sep = ',' if k < len(chs) - 1 else ' '
        body.append(f'"{ch}":({_fmt_sci(slope)},{_fmt_sci(intercept)}){sep} # mV*s')
    body.append('}')
    _WriteCalibBlock('ChargeToPes', [('gas', gas), ('temp', temp), ('version', version)], body, file_path)

def GetDCRPar(temp, gas):
    """
    Dark-noise calibration per channel, from FingerPlot4CHs.ipynb's
    calculate_DCR (Generalized/Borel-Poisson fit to the finger-plot peak areas).

    Returns
    -------
    DCRPar : dict of {'CH1': (DCR, DCR_err, lam, lam_err), ...}
        DCR [Hz] counts PRIMARY dark avalanches only; lam is the total
        correlated noise (crosstalk + afterpulses) per avalanche, as seen
        inside the calibration integration window. lam = np.nan where the
        fit fell back to a plain Poisson (too few peaks).
    calib_window : float
        Width [s] of the integration window lam was measured in.

    Fill/overwrite entries from the notebook with WriteDCRPar.
    """
    DCRPar, calib_window = None, None

    # Runs 70-115 (Xe) *******************************************************************************
    if gas == 'Xe':
        if temp == '8deg':
            # Run73, Generalized Poisson fit
            DCRPar={
            "CH1":(1.77e6,3.29e5,0.067,0.098), # (DCR [Hz], err, lam, err)
            "CH2":(1.55e6,5.41e5,0.042,0.158), # (DCR [Hz], err, lam, err)
            "CH3":(1.93e6,1.36e6,0.067,0.441), # (DCR [Hz], err, lam, err)
            "CH4":(1.90e6,3.28e5,0.000,0.083)  # (DCR [Hz], err, lam, err)
            }
            calib_window = 6.16e-7 # s

        elif temp == '10deg':
            # Run76, Generalized Poisson fit
            DCRPar={
            "CH1":(2.09e6,8.85e5,0.031,0.272), # (DCR [Hz], err, lam, err)
            "CH2":(1.35e6,3.09e4,0.113,0.009), # (DCR [Hz], err, lam, err)
            "CH3":(2.19e6,2.68e5,0.012,0.078), # (DCR [Hz], err, lam, err)
            "CH4":(1.76e6,5.93e5,0.062,0.159)  # (DCR [Hz], err, lam, err)
            }
            calib_window = 6.16e-7 # s

        elif temp == '11.5deg':
            # Run104, Generalized Poisson fit
            DCRPar={
            "CH1":(2.25e6,2.03e5,0.000,0.049), # (DCR [Hz], err, lam, err)
            "CH2":(2.01e6,6.59e5,0.002,0.166), # (DCR [Hz], err, lam, err)
            "CH3":(1.74e6,1.29e6,0.098,0.326), # (DCR [Hz], err, lam, err)
            "CH4":(2.50e6,9.17e5,0.000,0.275)  # (DCR [Hz], err, lam, err)
            }
            calib_window = 6.16e-7 # s

        elif temp == '12deg':
            # Run91, Generalized Poisson fit
            DCRPar={
            "CH1":(2.08e6,1.28e5,0.007,0.030), # (DCR [Hz], err, lam, err)
            "CH2":(1.86e6,6.67e5,0.032,0.176), # (DCR [Hz], err, lam, err)
            "CH3":(1.70e6,1.01e6,0.126,0.335), # (DCR [Hz], err, lam, err)
            "CH4":(2.20e6,1.13e6,0.000,0.291)  # (DCR [Hz], err, lam, err)
            }
            calib_window = 6.16e-7 # s

        elif temp == '13deg':
            # Run84, Generalized Poisson fit
            DCRPar={
            "CH1":(2.14e6,6.49e5,0.034,0.162), # (DCR [Hz], err, lam, err)
            "CH2":(2.06e6,2.52e5,0.037,0.063), # (DCR [Hz], err, lam, err)
            "CH3":(1.39e6,1.30e5,0.252,0.038), # (DCR [Hz], err, lam, err)
            "CH4":(2.19e6,3.15e5,0.032,0.090)  # (DCR [Hz], err, lam, err)
            }
            calib_window = 6.16e-7 # s

        elif temp == '13v2deg':
            # Run107, Generalized Poisson fit
            DCRPar={
            "CH1":(2.49e6,5.97e5,0.000,0.146), # (DCR [Hz], err, lam, err)
            "CH2":(2.08e6,5.00e5,0.051,0.129), # (DCR [Hz], err, lam, err)
            "CH3":(1.91e6,1.79e4,0.130,0.005), # (DCR [Hz], err, lam, err)
            "CH4":(2.28e6,2.84e5,0.000,0.070)  # (DCR [Hz], err, lam, err)
            }
            calib_window = 6.16e-7 # s

        elif temp == '13v3deg':
            # Run109, Generalized Poisson fit
            DCRPar={
            "CH1":(2.23e6,2.63e5,0.000,0.064), # (DCR [Hz], err, lam, err)
            "CH2":(2.09e6,1.08e6,0.000,0.296), # (DCR [Hz], err, lam, err)
            "CH3":(2.29e6,9.94e5,0.000,0.242), # (DCR [Hz], err, lam, err)
            "CH4":(1.71e6,8.43e5,0.097,0.234)  # (DCR [Hz], err, lam, err)
            }
            calib_window = 6.16e-7 # s

        elif temp == '14deg':
            # Run115, Generalized Poisson fit
            DCRPar={
            "CH1":(2.56e6,1.05e6,0.006,0.268), # (DCR [Hz], err, lam, err)
            "CH2":(2.12e6,4.39e5,0.057,0.114), # (DCR [Hz], err, lam, err)
            "CH3":(1.91e6,4.03e5,0.091,0.100), # (DCR [Hz], err, lam, err)
            "CH4":(2.44e6,1.82e5,0.069,0.045)  # (DCR [Hz], err, lam, err)
            }
            calib_window = 6.16e-7 # s
    # Runs 70-115 (Xe) *******************************************************************************

    # Runs 172-186 (Ar) *******************************************************************************
    elif gas == 'Ar':
        if temp == '6deg':
            # Run172, Generalized Poisson fit
            DCRPar={
            "CH1":(1.34e6,4.21e5,0.043,0.108), # (DCR [Hz], err, lam, err)
            "CH2":(1.01e6,1.05e5,0.153,0.028), # (DCR [Hz], err, lam, err)
            "CH3":(7.59e5,5.67e5,0.158,0.162), # (DCR [Hz], err, lam, err)
            "CH4":(1.10e6,4.49e5,0.117,0.120)  # (DCR [Hz], err, lam, err)
            }
            calib_window = 6.16e-7 # s

        elif temp == '7deg':
            # Run174, Generalized Poisson fit
            DCRPar={
            "CH1":(1.35e6,1.18e5,0.071,0.030), # (DCR [Hz], err, lam, err)
            "CH2":(1.17e6,1.88e5,0.137,0.050), # (DCR [Hz], err, lam, err)
            "CH3":(1.26e6,1.11e5,0.081,0.032), # (DCR [Hz], err, lam, err)
            "CH4":(9.85e5,1.57e5,0.179,0.042)  # (DCR [Hz], err, lam, err)
            }
            calib_window = 6.16e-7 # s

        elif temp == '8deg':
            # Run176, Generalized Poisson fit
            DCRPar={
            "CH1":(1.34e6,2.33e5,0.078,0.060), # (DCR [Hz], err, lam, err)
            "CH2":(1.13e6,6.04e4,0.154,0.016), # (DCR [Hz], err, lam, err)
            "CH3":(9.60e5,1.42e4,0.173,0.004), # (DCR [Hz], err, lam, err)
            "CH4":(1.28e6,3.59e5,0.123,0.095)  # (DCR [Hz], err, lam, err)
            }
            calib_window = 6.16e-7 # s

        elif temp == '9deg':
            # Run178, Generalized Poisson fit
            DCRPar={
            "CH1":(1.35e6,4.83e4,0.074,0.012), # (DCR [Hz], err, lam, err)
            "CH2":(1.21e6,1.71e5,0.147,0.045), # (DCR [Hz], err, lam, err)
            "CH3":(1.44e6,9.76e4,0.051,0.028), # (DCR [Hz], err, lam, err)
            "CH4":(1.13e6,3.20e5,0.185,0.088)  # (DCR [Hz], err, lam, err)
            }
            calib_window = 6.16e-7 # s

        elif temp == '10deg':
            # Run180, Generalized Poisson fit
            DCRPar={
            "CH1":(1.25e6,2.42e5,0.117,0.063), # (DCR [Hz], err, lam, err)
            "CH2":(1.20e6,1.72e5,0.158,0.046), # (DCR [Hz], err, lam, err)
            "CH3":(1.42e6,5.42e5,0.037,0.157), # (DCR [Hz], err, lam, err)
            "CH4":(1.35e6,3.24e5,0.137,0.087)  # (DCR [Hz], err, lam, err)
            }
            calib_window = 6.16e-7 # s

        elif temp == '11deg':
            # Run182, Generalized Poisson fit
            DCRPar={
            "CH1":(1.51e6,2.36e5,0.066,0.061), # (DCR [Hz], err, lam, err)
            "CH2":(1.32e6,1.59e5,0.151,0.043), # (DCR [Hz], err, lam, err)
            "CH3":(1.45e6,2.09e5,0.064,0.060), # (DCR [Hz], err, lam, err)
            "CH4":(1.41e6,6.30e5,0.129,0.166)  # (DCR [Hz], err, lam, err)
            }
            calib_window = 6.16e-7 # s

        elif temp == '12deg':
            # Run184, Generalized Poisson fit
            DCRPar={
            "CH1":(1.45e6,3.54e5,0.102,0.094), # (DCR [Hz], err, lam, err)
            "CH2":(1.57e6,1.94e5,0.088,0.050), # (DCR [Hz], err, lam, err)
            "CH3":(1.20e6,3.58e5,0.113,0.102), # (DCR [Hz], err, lam, err)
            "CH4":(1.01e6,3.32e5,0.272,0.095)  # (DCR [Hz], err, lam, err)
            }
            calib_window = 6.16e-7 # s

        elif temp == '13.5deg':
            # Run186, Generalized Poisson fit
            DCRPar={
            "CH1":(1.77e6,7.69e4,0.043,0.019), # (DCR [Hz], err, lam, err)
            "CH2":(1.59e6,2.64e5,0.124,0.070), # (DCR [Hz], err, lam, err)
            "CH3":(1.55e6,2.16e5,0.076,0.063), # (DCR [Hz], err, lam, err)
            "CH4":(1.44e6,3.79e5,0.160,0.101)  # (DCR [Hz], err, lam, err)
            }
            calib_window = 6.16e-7 # s
    # Runs 172-186 (Ar) *******************************************************************************

    if DCRPar is None:
        raise ValueError(f'No DCR calibration for gas={gas}, temp={temp}')

    return DCRPar, calib_window

def DarkCountsInWindow(window, channel, temp, gas, correlated=True):
    """
    Mean number of non-signal (dark) p.e. expected inside an integration
    window of width `window` [s], for a channel (1-4, or 'Sum' for the sum
    of the 4 channels), at the given gas/temp. Meant to be subtracted from
    the measured mean N_pe (from ChargeToPes) before computing a collection
    efficiency.

    What is taken into account
    --------------------------
    * Primary dark counts: thermal avalanches are a Poisson process with
      constant rate DCR, uncorrelated with the signal. So the expected
      number in ANY window of width T is DCR*T, independent of where the
      window sits relative to the trigger, and it scales linearly with T.
    * Correlated noise of those dark counts: each primary spawns a
      Borel(lam) cascade of crosstalk/afterpulse avalanches, giving
      1/(1-lam) avalanches on average. ChargeToPes converts charge to p.e.
      with the single-avalanche gain, so each of those secondaries is
      counted as one more p.e. in the integral. The dark charge in p.e. is
      therefore DCR*T/(1-lam), not DCR*T. Leaving lam out would undercount
      it by a factor (1-lam), e.g. 25% for lam = 0.2. Set correlated=False
      to get the primaries alone (DCR*T), e.g. to count how many windows
      have a dark count rather than how much charge they add.
    * 'Sum': the channels are independent SiPMs, so their means add and
      their errors add in quadrature.
    * Uncertainty: DCR_err and lam_err from the fit, propagated through
      DCR*T/(1-lam).

    What is ignored, and why
    ------------------------
    * Correlated noise of the SIGNAL p.e.: real photons also trigger
      crosstalk/afterpulses, so the measured signal is inflated by
      1/(1-lam) as well. That is not a dark count (it is proportional to
      the signal, not additive background), so it isn't returned here.
      For a collection efficiency, correct it separately. With
      lam = GetDCRPar(temp, gas)[0][ch][2],
          N_detected = (N_measured - DarkCountsInWindow(...)) * (1 - lam)
    * Change of lam with window length: lam is measured in calib_window
      and lumps prompt crosstalk (independent of T) with delayed crosstalk
      and afterpulses that happen to land inside the window (these grow
      with T). The peak areas alone can't separate the two, so the same
      lam is used for any T. That is fine for windows similar to
      calib_window; a warning is printed when T differs from it by more
      than a factor 2, since lam is then under- (longer T) or over-
      (shorter T) estimated.
    * Dark pulses just before the window whose tail (or afterpulses) leaks
      in, and those near the end that are only partly integrated: edge
      effects of order DCR*pulse_width, which on average roughly cancel
      and are small next to DCR*T for windows much longer than one pulse.
    * DCR-lam covariance from the fit: not stored, so errors are added in
      quadrature. The fit pins the mean number of p.e. (theta/(1-lam))
      better than theta or lam separately, so the two are anticorrelated
      and the quadrature error is conservative (an overestimate).
    * Event-by-event fluctuations: only the mean is returned (the variance,
      theta/(1-lam)^3, is super-Poissonian). For a collection efficiency
      built from mean N_pe, the mean is what's needed.
    * Temperature/overvoltage dependence: the DCR depends strongly on
      temperature, so values are only valid at the calibration point
      (no interpolation between temperatures).
    * lam = NaN (plain-Poisson fallback for channels with < 4 peaks):
      treated as 0, with a warning, since then the correlated part is
      missing and the result is a lower bound.

    Returns
    -------
    n_dark, n_dark_err : float
        Mean dark p.e. (or primaries, if correlated=False) in the window,
        and its uncertainty.
    """
    DCRPar, calib_window = GetDCRPar(temp, gas)

    if calib_window is not None and not (0.5 <= window / calib_window <= 2):
        print(f'Warning: window = {window:.2e} s differs from the calibration window '
              f'({calib_window:.2e} s) by more than x2; lam (correlated noise) is less reliable')

    chs = list(DCRPar.keys()) if channel == 'Sum' else [f'CH{channel}']

    n_dark, var_dark = 0., 0.
    for ch in chs:
        DCR, DCR_err, lam, lam_err = DCRPar[ch]
        if not correlated:
            lam, lam_err = 0., 0.
        elif np.isnan(lam):
            print(f'Warning: {ch} has no correlated-noise fit (lam = NaN), using lam = 0 (lower bound)')
            lam, lam_err = 0., 0.

        n_dark   += DCR * window / (1 - lam)
        var_dark += ((window / (1 - lam) * DCR_err)**2
                     + (DCR * window / (1 - lam)**2 * lam_err)**2)

    return n_dark, np.sqrt(var_dark)

def WriteDCRPar(DCR, DCR_err, XT, XT_err, gas, temp, run, calib_window, file_path=None):
    """
    Write calculate_DCR's outputs (dicts of {channel: value}) into the
    `temp` branch of GetDCRPar (in this same file) under the given gas, in
    the same format as the existing entries. Whatever that branch held
    before is replaced.

    Reload the module afterwards (importlib.reload(cig)) for GetDCRPar to see it.
    """
    def fmt_lam(x):
        return f'{x:.3f}' if np.isfinite(x) else 'np.nan'

    chs  = list(DCR.keys())
    body = [f'# {run}, Generalized Poisson fit', 'DCRPar={']
    for k, ch in enumerate(chs):
        sep = ',' if k < len(chs) - 1 else ' '
        body.append(f'"{ch}":({_fmt_sci(DCR[ch])},{_fmt_sci(DCR_err[ch])},'
                    f'{fmt_lam(XT[ch])},{fmt_lam(XT_err[ch])}){sep} # (DCR [Hz], err, lam, err)')
    body += ['}', f'calib_window = {_fmt_sci(calib_window)} # s']
    _WriteCalibBlock('GetDCRPar', [('gas', gas), ('temp', temp)], body, file_path)

def GetEpsTot(gas, pressure=None):
    """
    Light correction eps_tot: fraction of the scintillation photons of a
    full-energy Am-241 alpha that are NOT absorbed by the teflon or by the
    source, i.e. that can reach the SiPMs. From GeometricCorrections.ipynb
    (2D Gaussian fits to the nexus photon summaries in data/MarcFactors),
    eps_tot = 1 - mu_teflon/mu_photons - mu_source/mu_photons.

    Returns
    -------
    {pressure [bar]: (eps_tot, err, syst_down, syst_up), ...} for the gas,
    or the tuple of one pressure if `pressure` is given. err is statistical;
    syst_down/syst_up (both >= 0) are the asymmetric systematic errors from
    the optical model (GeometricCorrections_SystError.ipynb variations:
    steel/PTFE reflectivity, PTFE polishing, UNIFIED model), so
    eps_tot is in [eps_tot - syst_down, eps_tot + syst_up].

    Fill/overwrite entries from the notebook with WriteEpsTot.
    """
    EpsTot = None

    if gas == 'Ar':
        # GeometricCorrections.ipynb, 2D Gaussian fit, photon_threshold = 80
        EpsTot={
        1.0:(0.7069,0.0022,0.0000,0.0000), # (eps_tot, err, syst_down, syst_up)
        1.5:(0.6453,0.0014,0.0045,0.0068), # (eps_tot, err, syst_down, syst_up)
        2.5:(0.5726,0.0009,0.0026,0.0094), # (eps_tot, err, syst_down, syst_up)
        3.5:(0.5366,0.0010,0.0000,0.0147), # (eps_tot, err, syst_down, syst_up)
        4.5:(0.5195,0.0011,0.0003,0.0185), # (eps_tot, err, syst_down, syst_up)
        5.5:(0.5097,0.0013,0.0032,0.0187), # (eps_tot, err, syst_down, syst_up)
        6.5:(0.5019,0.0014,0.0000,0.0257), # (eps_tot, err, syst_down, syst_up)
        7.5:(0.4918,0.0014,0.0005,0.0253), # (eps_tot, err, syst_down, syst_up)
        8.5:(0.4818,0.0014,0.0007,0.0275)  # (eps_tot, err, syst_down, syst_up)
        }

    elif gas == 'Xe':
        # GeometricCorrections.ipynb, 2D Gaussian fit, photon_threshold = 80
        EpsTot={
        1.0:(0.6119,0.0010,0.0000,0.0000), # (eps_tot, err, syst_down, syst_up)
        1.5:(0.5623,0.0010,0.0001,0.0122), # (eps_tot, err, syst_down, syst_up)
        2.5:(0.5231,0.0012,0.0020,0.0180), # (eps_tot, err, syst_down, syst_up)
        3.5:(0.5042,0.0014,0.0009,0.0233), # (eps_tot, err, syst_down, syst_up)
        4.5:(0.4857,0.0014,0.0003,0.0298), # (eps_tot, err, syst_down, syst_up)
        5.5:(0.4778,0.0013,0.0006,0.0319), # (eps_tot, err, syst_down, syst_up)
        6.5:(0.4683,0.0012,0.0015,0.0315), # (eps_tot, err, syst_down, syst_up)
        7.5:(0.4668,0.0011,0.0024,0.0335), # (eps_tot, err, syst_down, syst_up)
        8.5:(0.4582,0.0010,0.0000,0.0367)  # (eps_tot, err, syst_down, syst_up)
        }

    if EpsTot is None:
        raise ValueError(f'No eps_tot for {gas}: run GeometricCorrections.ipynb with write_eps = True')
    if pressure is None:
        return EpsTot
    return EpsTot[pressure]

def WriteEpsTot(eps_tot, eps_tot_err, gas, photon_threshold, syst_down=None, syst_up=None, file_path=None):
    """
    Write GeometricCorrections.ipynb's eps_tot, statistical error and
    systematic errors (dicts of {pressure: value}; missing systematics are
    written as 0) into the `gas` branch of GetEpsTot (in this same file).
    Whatever that branch held before is replaced.

    Reload the module afterwards (importlib.reload(cig)) for GetEpsTot to see it.
    """
    syst_down = syst_down or {}
    syst_up   = syst_up or {}
    pressures = sorted(eps_tot.keys())
    body = [f'# GeometricCorrections.ipynb, 2D Gaussian fit, photon_threshold = {photon_threshold}', 'EpsTot={']
    for k, p in enumerate(pressures):
        sep = ',' if k < len(pressures) - 1 else ' '
        body.append(f'{float(p)}:({eps_tot[p]:.4f},{eps_tot_err[p]:.4f},'
                    f'{syst_down.get(p, 0.):.4f},{syst_up.get(p, 0.):.4f}){sep} # (eps_tot, err, syst_down, syst_up)')
    body.append('}')
    _WriteCalibBlock('GetEpsTot', [('gas', gas)], body, file_path)

def CreateWfSum(wf, channels, params):

    ChList = wf.columns[1:-3].tolist() 
    print(ChList)

    amp_factors = params['amp_factors']
    avg_amp_factor = np.mean(list(amp_factors.values()))  # Average amplification factor
    
    if params["is_amplified"] == True:
        # Calculate CHSum with amplification adjustments
        wf['CHSum'] = sum(  wf[channels[i]] / amp_factors[ch] * avg_amp_factor for i, ch in enumerate(amp_factors.keys())  )
    else:
        # Simple sum of the channels
        wf['CHSum'] = sum(wf[ch] for ch in channels)
            
    # Return the dataframe with the required columns
    return wf[['TIME']+ ChList +['CHSum', 'event', 'event_time', 'file_idx']]


# def BaselineCorrection(voltage_matrix: np.ndarray, window: int):
    # """
    # Baseline correction for waveforms using minimum standard deviation window.

    # Parameters
    # ----------
    # voltage_matrix : np.ndarray
    #     2D array of shape (nevents, nsamples), each row is a waveform.
    # window : int
    #     Sliding window size for mean/std calculation.

    # Returns
    # -------
    # voltage_corrected : np.ndarray
    #     Baseline-corrected waveforms, same shape as voltage_matrix.
    # baselines : np.ndarray
    #     Estimated baseline per waveform, shape (nevents,).
    # min_std_indices : np.ndarray
    #     Starting index of the lowest-variance window per waveform.
    # """

    # # (1) Moving mean
    # mean_profile = uniform_filter1d(voltage_matrix, size=window, axis=1, mode="reflect")

    # # (2) Moving variance (E[x^2] - (E[x])^2)
    # mean_sq_profile = uniform_filter1d(voltage_matrix**2, size=window, axis=1, mode="reflect")
    # var_profile = mean_sq_profile - mean_profile**2
    # std_profile = np.sqrt(np.maximum(var_profile, 0))  # numerical stability

    # # (3) Index of minimum std per row
    # min_idx = np.argmin(std_profile, axis=1)

    # # (4) Baseline value from mean_profile
    # row_idx = np.arange(voltage_matrix.shape[0])
    # baselines = mean_profile[row_idx, min_idx]

    # # (5) Subtract baseline from each waveform
    # voltage_corrected = voltage_matrix - baselines[:, None]

    # # (6) Compute starting index of window (accounting for centered window)
    # min_std_indices = np.clip(min_idx - window // 2, 0, voltage_matrix.shape[1] - 1)

    # return voltage_corrected, baselines, min_std_indices

def BaselineCorrection(voltage_matrix: np.ndarray, pretrigger=124, window: int = None):
    """
    Baseline correction using the median of each waveform.

    Parameters
    ----------
    voltage_matrix : np.ndarray
        2D array of shape (nevents, nsamples), each row is a waveform.
    window : (ignored)
        Included only for compatibility.

    Returns
    -------
    voltage_corrected : np.ndarray
        Baseline-corrected waveforms.
    baselines : np.ndarray
        Median baseline per waveform.
    median_indices : np.ndarray
        Dummy index array (all zeros, kept for compatibility).
    """


    # (1) Baseline = median of the first pretrigger samples in each waveform
    baselines = np.median(voltage_matrix[:, :pretrigger], axis=1)

    # (2) Subtract baseline
    voltage_corrected = voltage_matrix - baselines[:, None]

    # (3) No sliding window concept here, but return zeros to keep API identical
    median_indices = np.zeros(voltage_matrix.shape[0], dtype=int)

    return voltage_corrected, baselines, median_indices

def FixBaselineCorrection(voltage_matrix: np.ndarray, amplified: bool, channel: int):

    fixBaselines_noAmp = {'CH1': -369
                        ,'CH2': -410
                        ,'CH3': -350
                        ,'CH4': -408
                        }
    amp_offset = 600
    fixBaselines_Amp = {'CH1': -amp_offset
                        ,'CH2': -amp_offset
                        ,'CH3': -amp_offset
                        ,'CH4': -amp_offset
                        }

    # (1) Baseline = median of the first pretrigger samples in each waveform
    if amplified:
        baseline = fixBaselines_Amp[f'CH{channel}']
    else:
        baseline = fixBaselines_noAmp[f'CH{channel}']
    # (2) Subtract baseline
    voltage_corrected = voltage_matrix - baseline

    return voltage_corrected, baseline


def BaselinePeakCorrection(t, matrix, baseline_th_in_s = -0.2e-6, baseline_tolerance_in_V = 0.01e-3):
    # Peak correction
    matrix = PeakCorrection(matrix)

    # Baseline calculation 
    matrix_shifted = ShiftWaveformToPeak(t, matrix)
    
    # matrix_mean = np.average(matrix_shifted,axis=0)
    # matrix_baseline = matrix_mean[t < baseline_th_in_s].mean()

    waveform_before_trigger = matrix[:, t < baseline_th_in_s].flatten()

    binin = np.arange(waveform_before_trigger.min() - baseline_tolerance_in_V, 
                      waveform_before_trigger.max() + baseline_tolerance_in_V, 
                      baseline_tolerance_in_V)
    counts, bin_edges = np.histogram(waveform_before_trigger, bins=binin)
    # Find the index of the highest bin
    max_bin_idx = np.argmax(counts)

    # Get the bin range
    bin_start, bin_end = bin_edges[max_bin_idx], bin_edges[max_bin_idx + 1]
    mode = (bin_start + bin_end)/2

    matrix_baseline = mode
    # print(f'baseline = {mode*1e3:.2f} [mV]')
    
    # Baseline correction
    matrix_corrected = matrix - matrix_baseline

    return matrix_corrected


def ReadWaveform(run, file, event): 
    h5_files = read_directory(run)
    waveform = pd.read_hdf(h5_files[file], where=f'(event == {event})')
    return waveform


def PrintWaveform(time, waveform, run, file, event, label = '', time_units = 'us', title = None):

    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(12,5), constrained_layout=True)

    font_size = 20

    if time_units == 'us':
        time        = time*1e6
        time_units  = r'$\mu$s'
    if time_units == 'ns':
        time = time*1e9
    
    ax.plot(time, waveform[event], label = f'{label}')
        
    if title == None:
        title = f'Event nº {event}'

    ax.set_title(f'{title}; run {run}_{file}', fontsize = font_size);
    ax.set_xlabel(f'Time [{time_units}]', fontsize = font_size);
    ax.set_ylabel('Signal [V]', fontsize = font_size);
    
    return fig, ax


def PrintSpectrumPerChannel(run, charge_df, channels = [1, 2, 3, 4], pes = True, amplified = True, temp = 'room'):
    fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize=(10,7), constrained_layout=True)
    font_size = 20
    nbins = 100

    for i, ii in enumerate(channels):
        axx = ax[(i // 2) % 2, i%2]
        
        if pes:
            charge_in_pes = ChargeToPes(charge_df, ii, temp, amplified)
            events, bins, image = axx.hist(charge_in_pes, bins = nbins, label = f'CH{ii}', alpha = 1)
    #         axx.set_xlim(0, 500)        
        else:
            channel_charge = charge_df[f'charge_CH{ii}']
            events, bins, image = axx.hist(channel_charge*1e6, bins = nbins, label = f'CH{ii}', alpha = 1)
    #         axx.set_xlim(0, 0.02)        

        if pes:
            axx.set_xlabel(r'Charge [pes]', fontsize = font_size);
        else:
            axx.set_xlabel(r'Charge [V $\cdot$ $\mu$ s]', fontsize = font_size);
            
        axx.set_ylabel(r'Counts', fontsize = font_size);
        
        peaks, _ = find_peaks(events, height = events.max()*0.6, distance=15)  # Adjust 'height' as needed to filter smaller peaks
        peak_values = [(bins[i] + bins[i + 1]) / 2 for i in peaks]

        axx.plot(peak_values, events[peaks], 'o', label = f'Peak value = {max(peak_values):.2f}')
        
        # axx.set_yscale('log')
        axx.legend(loc = 'best', fontsize = 0.5*font_size)
        
    # Set a general title for the entire figure
    fig.suptitle(f'{run}', fontsize = font_size)

    return ax


def PrintTotalSpectrum(run, charge_df, channels = [1, 2, 3, 4], pes = True, amplified = False, temp = 'room', fit = None, new_figure = True, 
                       bins = 100, alpha = 1, density = False, label = None):

    if new_figure:
            fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(10, 7), constrained_layout=True) # Create a new figure

    else:
        # Check if there's an existing figure and create it if there's none
        if plt.gcf().get_axes():
            ax = plt.gcf().get_axes()[0]
        else:
            fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(10, 7), constrained_layout=True)

    font_size = 20

    charge_in_pes = pd.DataFrame([])

    for i, ii in enumerate(channels):
        charge_in_pes[f'charge_CH{ii}'] = ChargeToPes(charge_df, ii, temp, amplified)

    total_charge_in_pes = charge_in_pes.sum(axis = 1)
    total_charge_in_Vus = charge_df.sum(axis = 1)*1e6

    if label == None:
        label = f'{len(channels)} channels sum ({len(charge_df)} waveforms)'
        
    if pes:
        events, bins, image = ax.hist(total_charge_in_pes, bins = bins, density = density,
                                    label = label, 
                                    alpha = alpha)
        ax.set_xlabel(r'Charge [pes]', fontsize = font_size);
        charge_data = total_charge_in_pes
        # ax.set_xlim(0, 500)
    else:
        
        events, bins, image = ax.hist(total_charge_in_Vus, bins = bins*2, density = density,
                                    label = label, 
                                    alpha = alpha)
        ax.set_xlabel(r'Charge [V $\cdot$ $\mu$ s]', fontsize = font_size);
        charge_data = total_charge_in_Vus
        # ax.set_xlim(0, 0.04)

    # Find all peaks in the histogram
    peaks, _ = find_peaks(events, height=events.max()*0.6, distance=30)  # Adjust 'height' as needed to filter smaller peaks
    peak_values = [(bins[i] + bins[i + 1]) / 2 for i in peaks]
    peak_values = np.array(peak_values)

    ax.plot(peak_values, events[peaks], 'o', 
            label =  f"Peak value(s) = {', '.join(f'{peak:.2f}' for peak in peak_values)}")

    
        
    # ax.set_title(f'Multiphoton spectrum', fontsize = font_size);
    ax.set_title(f'{run}', fontsize = font_size);
    ax.set_ylabel(r'Counts', fontsize = font_size);

    # ax.set_yscale('log')
    ax.legend(loc = 'best', fontsize = font_size*0.5)
    ax.set_ylim(0, events.max()*4/3)

    if fit is None:
        return ax, bins, events
    
    else:
        if fit == 'gaussian':
            # Fit data
            x_data = (bins[1:] + bins[:-1]) / 2
            y_data = events
            mask   = (x_data > (max(peak_values) - 300)) & (x_data < (max(peak_values) + 300))
            x_data = x_data[mask]
            y_data = y_data[mask]
            # Compute weighted mean
            # weighted_mean = np.sum(x_data * y_data) / np.sum(y_data)

            A     = 10*y_data.sum()
            mu    = 1000 # Mean
            sigma = 60   # Standard deviation

            initial_guess = [A, mu, sigma]

            bounds = (0, np.inf)
            popt, pcov = curve_fit(gaussian, x_data, y_data, p0 = initial_guess, bounds = bounds)

            ax.plot(x_data, gaussian(x_data, *popt), '--r',
                label = fr'A = {popt[0]:.2f}; $\mu$ = {popt[1]:.2f}; $\sigma$ = {popt[2]:.2f}')

        if fit == 'crystalball':
            # Fit data
            x_data = (bins[1:] + bins[:-1]) / 2
            y_data = events
            mask   = x_data > (max(peak_values) - 500)
            x_data = x_data[mask]
            y_data = y_data[mask]
            # Compute weighted mean
            # weighted_mean = np.sum(x_data * y_data) / np.sum(y_data)

            A     = 10*y_data.sum()
            # beta  = 0.02  # Tail parameter
            beta  = 0.2  # Tail parameter
            # m     = 2 # Shape parameter
            m     = 1.5 # Shape parameter
            # loc   = max(peak_values) # Mean
            # loc   = (y_data*x_data).sum()/y_data.sum() # Mean
            loc   = 1000 # Mean
            scale = 60   # Standard deviation

            initial_guess = [A, beta, m, loc, scale]

            bounds = (0, np.inf)
            popt, pcov = curve_fit(crystalball_fit, x_data, y_data, p0 = initial_guess, bounds = bounds)

            ax.plot(x_data, crystalball_fit(x_data, *popt), '--r',
                label = fr'A = {popt[0]:.2f}; $\beta$ = {popt[1]:.2f}; m = {popt[2]:.2f}; $\mu$ = {popt[3]:.2f}; scale = {popt[4]:.2f}')
    
        ax.legend(loc = 'best', fontsize = font_size*0.5)

        return ax, bins, events, popt, pcov


def CalculateEfficiency(detected_pes_dict, correction_file = '20250210_LightCorrection.csv', poisson_aprox = False):
    
    reference_plot = {2.004: 1.075e5,
                        2.512: 1.088e5,
                        2.985: 1.122e5,
                        3.493: 1.180e5,
                        4.000: 1.233e5,
                        4.507: 1.285e5,
                        4.997: 1.325e5,
                        6.011: 1.526e5,
                        6.974: 1.727e5,
                        8.004: 1.903e5,
                        8.967: 2.072e5,
                        9.965: 2.183e5}

    # Extract X and Y values and sort them
    x = np.array(sorted(reference_plot.keys()))
    y = np.array([reference_plot[k] for k in sorted(reference_plot.keys())])

    # Create an interpolation function (linear by default)
    reference_func = interp1d(x, y, kind='linear', fill_value="extrapolate")

    path = '/scratch/marian/python/data'
    path = path + '/cigar'    
    correction_path = os.path.join(path, correction_file) 
    correction_df   = pd.read_csv(correction_path)

    # Create dictionary
    created_photons_dict = dict(
        zip(correction_df['pressure'], 
            correction_df['max_photon_counts']
        ))
    plate_correction_dict = dict(
        zip(correction_df['pressure'], 
            correction_df['max_source_hits']/correction_df['max_photon_counts']
        ))
    plate_correction_error_dict = dict(
        zip(correction_df['pressure'], 
            correction_df['max_source_hits_err']/correction_df['max_source_hits']
        ))
    teflon_correction_dict = dict(
        zip(correction_df['pressure'], 
            correction_df['max_teflon_hits']/correction_df['max_photon_counts']
        ))
    teflon_correction_error_dict = dict(
        zip(correction_df['pressure'], 
            correction_df['max_teflon_hits_err']/correction_df['max_teflon_hits']
        ))

    eff_dict = {}
    eff_error_dict = {}
    corrections_dict = {}
    detectable_photons_dict = {}

    for pressure, detected_pes in detected_pes_dict.items():
    
        corrections_dict[pressure] = {'created':[], 'plate':[], 'teflon':[]}
        
        plate_correction        = plate_correction_dict[f'{pressure}bar']
        plate_correction_err    = plate_correction_error_dict[f'{pressure}bar']
        teflon_correction       = teflon_correction_dict[f'{pressure}bar']
        teflon_correction_err   = teflon_correction_error_dict[f'{pressure}bar']
        
        photons_created = reference_func(pressure)
        # photons_created = created_photons_dict[f'{pressure}bar']
        corrections_dict[pressure]['created'].append(photons_created)
                                        
        photons_plate       = photons_created*plate_correction
        photons_plate_err   = photons_created*plate_correction_err
        corrections_dict[pressure]['plate'].append(photons_plate)
        # photons_teflon = (photons_created - photons_plate)*teflon_correction
        photons_teflon      = photons_created*teflon_correction
        photons_teflon_err  = photons_created*teflon_correction_err
        corrections_dict[pressure]['teflon'].append(photons_teflon)
        
        corrections = photons_plate + photons_teflon
        
        detectable_photons                  = photons_created - corrections
        detectable_photons_dict[pressure]   = detectable_photons
        
        k       = detected_pes
        N       = detectable_photons
        N_err   = np.sqrt(photons_plate_err**2 + photons_teflon_err**2)

        eff_dict[pressure] = k/N
        if poisson_aprox:
            k_err                       = np.sqrt(k)
            eff_error_dict[pressure]    = (k/N)*np.sqrt((k_err/k)**2 + (N_err/N)**2)
        else:
            # Binomial aprox
            eff_error_dict[pressure] = np.sqrt(k*(1 - k/N))*(1/N)

    # Transform dictionary into DataFrame
    corrected_values_df = pd.DataFrame.from_dict({
                                                    pressure: {
                                                    'pressure': pressure,
                                                    'created': int(values['created'][0]),
                                                    'plate'  : int(values['plate'][0]),
                                                    'teflon' : int(values['teflon'][0])
                                                }
                                                for pressure, values in corrections_dict.items()
                                            }, orient='index')
    
    return eff_dict, eff_error_dict, detectable_photons_dict, corrected_values_df


def PlotEfficiency(detected_pes_dict, 
                   detected_pes_error_dict, 
                   detectable_photons_dict, 
                   eff_dict,
                   eff_error_dict
                   ):
    # Create the figure and gridspec with different heights
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})

    # Example data provided
    charge_pressure     = np.array(list(detected_pes_dict.keys()))
    charge              = np.array(list(detected_pes_dict.values()))
    charge_error        = np.array(list(detected_pes_error_dict.values()))

    # light_pressure    = np.array(list(reference_plot.keys()))
    # light             = np.array(list(reference_plot.values()))
    light_pressure  = np.array(list(detectable_photons_dict.keys()))
    light           = np.array(list(detectable_photons_dict.values()))


    efficiency_pressure     = np.array(list(eff_dict.keys()))
    efficiency              = np.array(list(eff_dict.values())) * 100
    efficiency_error        = np.array(list(eff_error_dict.values())) * 100


    # Top plot for charge and light data
    ax1.errorbar(charge_pressure[:], charge[:], yerr=charge_error[:], fmt='o-', label=r'Am-241 peak', color='black',
                markerfacecolor='none', markeredgecolor='black')
    ax1.set_ylabel(r'Charge [# p.e.]')
    ax1.grid(True)

    # Plot light data without error bars
    ax1_right = ax1.twinx()
    ax1_right.plot(light_pressure, light, 's--', label=r'doi:10.1109/TNS.2002.801700 (right axis)', color='gray',
                markerfacecolor='none', markeredgecolor='gray')
    ax1_right.set_ylabel(r'Light [# photons]')

    # Make the right y-axis gray
    ax1_right.spines['right'].set_color('gray')  # Change the spine color
    ax1_right.tick_params(axis='y', colors='gray')  # Change the tick color
    ax1_right.yaxis.label.set_color('gray')  # Change the label color

    # Legends
    ax1.legend(loc= (0.1, 0.9), fontsize = 0.7*font_size)
    ax1_right.legend(loc= (0.1, 0.8), fontsize = 0.7*font_size)

    # Bottom plot for efficiency
    ax2.errorbar(efficiency_pressure[:], efficiency[:], yerr=efficiency_error[:], fmt='o--', color='black',
                markerfacecolor='none', markeredgecolor='black')
    ax2.set_xlabel(r'Pressure [bar]')
    ax2.set_ylabel(r'Efficiency [%]')
    ax2.grid(True)

    # Adjust layout
    plt.tight_layout()
    plt.show()

    return fig, (ax1, ax2)


def MeanWaveformOLD(h5_files, channels = [1, 2, 3, 4], start_files = 0, top_files = -1,):
   
    mean_dict = {}
    for ii in channels:
        mean_dict[f'mean_CH{ii}']    = []
        mean_dict[f'mean_err_CH{ii}']    = []

    time_ticks   = None

    for file in h5_files[start_files:top_files]:
        
        print(f'Processing... {file}' + ' '*20, end = '\r')
        
        for i in range(10):
            print(f'Processing... {file} (batch {i})' + ' '*20, end = '\r')
            df = pd.read_hdf(file, where=f'(event >= {200*i}) & (event < {200*(i+1)})')
            
            if time_ticks is None:
                # Calculate it only during the first iteration
                time_ticks = len(df.groupby('event').get_group(1))

                time = df["TIME"].to_numpy()[:time_ticks] # [s]

            for ii in channels:
                ch          = df[f"CH{ii}"].to_numpy().reshape(-1, time_ticks)  # [V]
                # Baseline and Peak correction
                ch          = BaselinePeakCorrection(time, ch, baseline_th)
                ch          = ShiftWaveformToPeak(time, ch)
                ch_mean     = np.average(ch,axis=0)

                ch_var      = np.var(ch, axis=0, ddof=1)
                ch_mean_err = np.sqrt(ch_var / np.shape(ch)[0]) # sqrt(var) = sigma; mu_err = sigma/sqrt(N)

                mean_dict[f'mean_CH{ii}'].append(ch_mean)
                mean_dict[f'mean_err_CH{ii}'].append(ch_mean_err)

    mean_dict['TIME'] = time
    for ii in channels:
        mean_dict[f'mean_CH{ii}']       = np.average(np.array(mean_dict[f'mean_CH{ii}']),axis=0)
        mean_dict[f'mean_err_CH{ii}']   = np.average(np.array(mean_dict[f'mean_err_CH{ii}']),axis=0)

    mean_df = pd.DataFrame(mean_dict)
        
    print(f'DONE! Last file processed: {file}' + ' '*20, end = '\r')

    return mean_df


def MeanWaveform(h5_files, channels=[1, 2, 3, 4], start_files=0, top_files=-1):

    num_channels        = len(channels)
    total_waveforms     = 2000  # Total waveforms across all batches
    num_batches         = 1  # Fixed number of batches
    num_files           = len(h5_files[start_files:top_files])
    waveforms_per_batch = total_waveforms//num_batches  # Each batch contains exactly 200 waveforms
    print(f'Waveforms per batch: {waveforms_per_batch}')

    # First pass: Determine time_ticks (only needed once) and time axis (same for all waveforms)
    df          = pd.read_hdf(h5_files[0], where='event == 0')  # Get time length from the first event
    time_ticks  = len(df)  # Get time length from the first event
    time        = df["TIME"].to_numpy()[:time_ticks]  # Extract time values

    # Preallocate 3D arrays: Shape (num_channels, num_batches, time_ticks)
    batch_means     = np.zeros((num_channels, num_batches*num_files, time_ticks), dtype=np.float32)  # Mean per batch
    batch_vars      = np.zeros((num_channels, num_batches*num_files, time_ticks), dtype=np.float32)  # Variance per batch
        
    file_batch_idx = 0

    for file in h5_files[start_files:top_files]:
        print(f'Processing... {file}' + ' '*20, end='\r')

        for batch_idx in range(num_batches):
            print(f'Processing... {file} (batch {batch_idx})' + ' '*20, end='\r')
            df = pd.read_hdf(file, where=f'(event >= {waveforms_per_batch*batch_idx}) & (event < {waveforms_per_batch*(batch_idx+1)})')

            for ch_idx, ii in enumerate(channels):
                ch = df[f"CH{ii}"].to_numpy().reshape(-1, time_ticks)  # [V]
                ch = BaselinePeakCorrection(time, ch, baseline_th)
                ch = ShiftWaveformToPeak(time, ch)

                # Store directly in preallocated arrays (CHANNEL first)
                batch_means[ch_idx, file_batch_idx]  = np.average(ch, axis=0)
                batch_vars[ch_idx, file_batch_idx]   = np.var(ch, axis=0, ddof=1)

            file_batch_idx = file_batch_idx + 1

    # Compute overall mean waveform across all batches (simple mean)
    mean_waveform = np.average(batch_means, axis=1)

    # Compute error on the mean
    mean_variance = np.average(batch_vars, axis=1)  # Since all batches have equal waveforms
    error_on_mean = np.sqrt(mean_variance / total_waveforms) # sqrt(var) = sigma; mu_err = sigma/sqrt(N)

    # Create DataFrame for output
    mean_dict = {'TIME': time}
    for ch_idx, ii in enumerate(channels):
        mean_dict[f'mean_CH{ii}']       = mean_waveform[ch_idx]
        mean_dict[f'mean_err_CH{ii}']   = error_on_mean[ch_idx]

    mean_df = pd.DataFrame(mean_dict)

    print(f'DONE! Last file processed: {file}' + ' '*20, end='\r')

    return mean_df


def MeanWaveformDAQ(files_list, files_type = 'bin', channels=[1, 2, 3, 4], start_files=0, top_files=-1):

    num_channels        = len(channels)
    waveforms_per_file  = 2000  # Total waveforms per file
    num_files           = len(files_list[start_files:top_files])  # Fixed number of batches
    total_waveforms     = waveforms_per_file*num_files
    time_ticks          = None


    for file_idx, file in enumerate(files_list[start_files:top_files]):
        print(f'Processing... {file}' + ' '*20, end='\r')

        df, _ = daqDecoder(file, files_type)

        if time_ticks is None:
            # First pass: Determine time_ticks (only needed once) and time axis (same for all waveforms)
            first_event = int(df['event'].unique()[0])
            time_ticks  = len(df.groupby('event').get_group(first_event)) # Get time length from the first event
            time        = df["TIME"].to_numpy()[:time_ticks]  # Extract time values
            # Preallocate 3D arrays: Shape (num_channels, num_batches, time_ticks)
            file_means     = np.zeros((num_channels, num_files, time_ticks), dtype=np.float32)  # Mean per file
            file_vars      = np.zeros((num_channels, num_files, time_ticks), dtype=np.float32)  # Variance per file

        for ch_idx, ii in enumerate(channels):
            ch = df[f"CH{ii}"].to_numpy().reshape(-1, time_ticks)  # [V]
            # ch = BaselinePeakCorrection(time, ch, baseline_th)
            # ch = ShiftWaveformToPeak(time, ch)

            # Store directly in preallocated arrays (CHANNEL first)
            file_means[ch_idx, file_idx]  = np.average(ch, axis=0)
            file_vars[ch_idx, file_idx]   = np.var(ch, axis=0, ddof=1)

    # Compute overall mean waveform across all batches (simple mean)
    mean_waveform = np.average(file_means, axis=1)

    # Compute error on the mean
    mean_variance = np.average(file_vars, axis=1)  # Since all batches have equal waveforms
    error_on_mean = np.sqrt(mean_variance / total_waveforms) # sqrt(var) = sigma; mu_err = sigma/sqrt(N)

    # Create DataFrame for output
    mean_dict = {'TIME': time}
    for ch_idx, ii in enumerate(channels):
        mean_dict[f'mean_CH{ii}']       = mean_waveform[ch_idx]
        mean_dict[f'mean_err_CH{ii}']   = error_on_mean[ch_idx]

    mean_df = pd.DataFrame(mean_dict)

    print(f'DONE! Last file processed: {file}' + ' '*20, end='\r')

    return mean_df


def MeanWaveformDAQnew(files_list, files_type='bin', channels=[1, 2, 3, 4], start_files=0, top_files=-1):

    num_channels        = len(channels)
    waveforms_per_file  = 2000  # Total waveforms per file
    num_files           = len(files_list[start_files:top_files])  # Fixed number of batches
    total_waveforms     = waveforms_per_file*num_files

    # Step 1: Determine the global time range
    all_time_data = []
    global_time_set = set()  # Use a set for fast uniqueness check

    for file in files_list[start_files:top_files]:
        df, _ = daqDecoder(file, files_type)
        unique_time_values = np.unique(df["TIME"].to_numpy())  # Get unique times directly
        global_time_set.update(unique_time_values)  # Add to global set

    # Step 2: Create a global time axis (merged unique times)
    global_time = np.array(sorted(global_time_set))  # Convert set to sorted array
    num_time_points = len(global_time)
    print(num_time_points)
    # Preallocate 3D arrays: Shape (num_channels, num_batches, time_ticks)
    file_means     = np.full((num_channels, num_files, num_time_points), np.nan)  # Mean per file
    file_vars      = np.full((num_channels, num_files, num_time_points), np.nan)  # Variance per file

    # Step 3: Process each file and store waveforms in their actual time positions
    for file_idx, file in enumerate(files_list[start_files:top_files]):
        print(f'Processing... {file}' + ' ' * 20, end='\r')

        df, _ = daqDecoder(file, files_type)
        time_values = df["TIME"].to_numpy()

        for ch_idx, ch in enumerate(channels):
            ch_waveforms = df[f"CH{ch}"].to_numpy().reshape(-1, len(np.unique(time_values)))  # [V]
            print(len(time_values))

            average_waveform    = np.average(ch_waveforms, axis=0)
            waveform_variance   = np.var(ch_waveforms, axis=0, ddof=1)

            mask = np.isin(global_time, time_values)  # Find matching indices
            print(len(average_waveform))
            print(len(file_means[ch_idx, file_idx][mask]))

            file_means[ch_idx, file_idx][mask]  = average_waveform
            file_vars[ch_idx, file_idx][mask]   = waveform_variance


    # Step 4: Compute mean and error while ignoring NaNs
    mean_waveform = np.nanmean(file_means, axis=1)

    mean_variance = np.nanvar(file_vars, axis=1)  # Since all batches have equal waveforms
    error_on_mean = np.sqrt(mean_variance / np.sum(~np.isnan(file_means), axis=1)) # sqrt(var) = sigma; mu_err = sigma/sqrt(N)

    # Step 5: Convert to DataFrame
    mean_dict = {'TIME': global_time}
    for ch_idx, ii in enumerate(channels):
        mean_dict[f'mean_CH{ii}']       = mean_waveform[ch_idx]
        mean_dict[f'mean_err_CH{ii}']   = error_on_mean[ch_idx]

    mean_df = pd.DataFrame(mean_dict)

    print(f'DONE! Last file processed: {file}' + ' ' * 20, end='\r')

    return mean_df



def CreateChargeDataFrame(h5_files, channels = [1, 2, 3, 4], start_files = 0, top_files = -1):

    charge_dict = {}
    for ii in channels:
        charge_dict[f'charge_CH{ii}']    = []

    time_ticks   = None

    for file in h5_files[start_files:top_files]:
        
        print(f'Processing... {file}' + ' '*20, end = '\r')
        
        for i in range(10):
            print(f'Processing... {file} (batch {i})' + ' '*20, end = '\r')
            df = pd.read_hdf(file, where=f'(event >= {200*i}) & (event < {200*(i+1)})')
            
            if time_ticks is None:
                # Calculate it only during the first iteration
                time_ticks = len(df.groupby('event').get_group(1))

            time = df["TIME"].to_numpy()[:time_ticks] # [s]
            integration_window = (time > -3e-6) & (time < 9e-6)

            for ii in channels:
                ch          = df[f"CH{ii}"].to_numpy().reshape(-1, time_ticks)  # [V]
                # Baseline and Peak correction
                ch          = BaselinePeakCorrection(time, ch, baseline_th)
                ch_area     = np.trapz(ch[:, integration_window], time[integration_window], axis=1)
                charge_dict[f'charge_CH{ii}'].append(ch_area)

                  
    for ii in channels:
        charge_dict[f'charge_CH{ii}']    = np.array(charge_dict[f'charge_CH{ii}'])
        charge_dict[f'charge_CH{ii}']    = charge_dict[f'charge_CH{ii}'].flatten()

    charge_df = pd.DataFrame(charge_dict)
        
    print(f'DONE! Last file processed: {file}' + ' '*20, end = '\r')

    return charge_df

def CreateChargeDataFrameDAQ(directory_files, channels = [1, 2, 3, 4], start_files = 0, top_files = -1, file_type = 'bin'):

    charge_dict         = {}
    num_files           = len(directory_files[start_files:top_files])  # Fixed number of batches
    waveforms_per_file  = 2000  # Total waveforms per file

    for ii in channels:
        charge_dict[f'charge_CH{ii}'] = np.zeros((num_files, waveforms_per_file), dtype=np.float32)

    # for ii in channels:
    #     charge_dict[f'charge_CH{ii}']    = []

    time_ticks   = None

    for file_idx, file in enumerate(directory_files[start_files:top_files]):
        print(f'Processing... {file}' + ' '*20, end='\r')

        df, _ = daqDecoder(file, file_type)

        if time_ticks is None:
            # First pass: Determine time_ticks (only needed once) and time axis (same for all waveforms)
            first_event = int(df['event'].unique()[0])
            time_ticks  = len(df.groupby('event').get_group(first_event)) # Get time length from the first event
            time        = df["TIME"].to_numpy()[:time_ticks] # [s]  # Extract time values

        
        integration_window = (time > -1e-6) & (time < 90e-6)

        for ii in channels:
            ch          = df[f"CH{ii}"].to_numpy().reshape(-1, time_ticks)  # [V]
            # Baseline correction
            ch_mean     = np.average(ch,axis=0)
            # baseline    = ch_mean[time < 0.5e-6].mean()
            baseline    = ch_mean[ch_mean < 0].mean()
            ch          = ch - baseline

            ch_area     = np.trapz(ch[:, integration_window], time[integration_window], axis=1)
            print(ii)
            print(np.shape(ch), np.shape(ch_area))
            charge_dict[f'charge_CH{ii}'][file_idx, :] = ch_area

                  
    for ii in channels:
        charge_dict[f'charge_CH{ii}']    = charge_dict[f'charge_CH{ii}'].flatten()

    charge_df = pd.DataFrame(charge_dict)
        
    print(f'DONE! Last file processed: {file}' + ' '*20, end = '\r')

    return charge_df


def find_peaks_in_matrix(matrix, output = 'position', **kwargs):
    """
    Apply scipy.signal.find_peaks row-wise to a 2D matrix.

    Parameters:
        matrix (np.ndarray): A 2D array where each row is a waveform.
        **kwargs: Additional keyword arguments for scipy.signal.find_peaks.

    Returns:
        np.ndarray: An array of arrays, where each sub-array contains the peak indices of a row.
    """
    if matrix.ndim != 2:
        raise ValueError("Input matrix must be a 2D array.")

    # Using list comprehension for clarity and efficiency
    peaks_list = [find_peaks(row, **kwargs)[0] for row in matrix]

    # Converting list of arrays to an array of arrays
    peaks_pos =  np.array(peaks_list, dtype=object)

    if output == 'position':
        return peaks_pos
    
    else:
        peaks_binary = np.zeros_like(matrix)
        for row in range(np.shape(matrix)[0]):

            peaks_binary[row][peaks_pos[row]] = 1
    
        if output == 'binary':
            return peaks_binary
        if output == 'boolean':
            return peaks_binary.astype(bool)
            

def MovingSum(matrix, time, time_window_width, dt, **kwargs):

    moving_sum= []

    nwindows = int(np.ceil((time.max() - time.min())/dt))
    t0 = time.min()

    peaks_boolean   = find_peaks_in_matrix(matrix, output = 'boolean', **kwargs)
    peaks_values    = peaks_boolean*matrix

    for nwindow in range(nwindows):
        t0              = t0 + dt*nwindow
        time_window     = (time >= t0) & (time < (t0 + time_window_width))
        window_voltage  = peaks_values[:, time_window].sum(axis = 1) # [V] total voltage inside the time window for ALL waveforms
        moving_sum.append(window_voltage)

    return np.concatenate(moving_sum)


def MovingPeakDistribution(h5_files, channels = [1, 2, 3, 4], start_files = 0, top_files = -1, time_window_width = 4*1e-6, dt = 1*1e-9):

    moving_peak_dist = {}
    for ch in channels:
        moving_peak_dist[f'CH{ch}'] = []

    time_ticks      = None
    batch_width     = 2000

    for file in h5_files[start_files:top_files]:
        for batch in range(1):
            print(f'Processing... {file} (batch {batch})' + ' '*20, end = '\r')

            df = pd.read_hdf(file, where=f'(event >= {batch_width*batch}) & (event < {batch_width*(batch+1)})')
            
            if time_ticks is None:
                # Calculate it only during the first iteration
                time_ticks = len(df.groupby('event').get_group(batch_width*batch + 1))

            time = df["TIME"].to_numpy()[:time_ticks] # [s]

            for ch in channels:
                print(f'Processing... {file} (batch {batch} channel {ch})' + ' '*20, end = '\r')
                wvf = df[f"CH{ch}"].to_numpy().reshape(-1, time_ticks)  # [V]
                wvf = BaselinePeakCorrection(time, wvf, baseline_th)

                if ch == 4:
                    peak_th = .15 # [V] 
                else:
                    peak_th = 0.07

                # Create a dictionary for the keyword arguments
                kwargs = {'height': peak_th, 'prominence': 0.1, 'distance': 15}

                moving_sum = MovingSum(wvf, time, time_window_width, dt, **kwargs)
                moving_peak_dist[f'CH{ch}'].append(moving_sum)

    for ch in channels:
        moving_peak_dist[f'CH{ch}'] = np.concatenate(moving_peak_dist[f'CH{ch}'])

    return moving_peak_dist



# 2D MAPS

def Read2Dmap(csv_file):
    # Read the data into a DataFrame
    df = pd.read_csv(csv_file, index_col=0)
    # Convert the DataFrame to a numpy array for the histogram values
    hist = df.to_numpy()
    # Check what pressure the file is for
    pressure = csv_file.split('_')[2]
    # Remove 'test' string from pressure
    pressure = pressure.replace('test', '')

    if 'teflon' in csv_file:
        string = 'Teflon Photons'
    elif 'source' in csv_file:
        string = 'Source Photons'

    # Extract bin edges from the DataFrame indices and columns
    ycenters = df.index.to_numpy().astype(float)  # Teflon/source photons (rows of DataFrame)
    xcenters = df.columns.to_numpy().astype(float)  # Total photons (columns of DataFrame)

    # Remove the first bin (0,0 bin) if it exists
    if ycenters[0] == 0 and xcenters[0] == 0:
        hist    = hist[1:, 1:]  # Remove the first row and column
        ycenters  = ycenters[1:]  # Remove the first y edge (total photons)
        xcenters  = xcenters[1:]  # Remove the first x edge (Teflon photons)

    return pressure, string, hist, ycenters, xcenters


def Slice2Dmap(hist):
    # Calculate the sum of each column
    column_sums = np.sum(hist, axis=0)

    # Find the index of the column with the maximum sum
    column_index = np.argmax(column_sums)

    # Slice the column
    selected_column = hist[:, column_index]

    return column_index, selected_column


def DefineCustomMap(colormap = plt.cm.viridis, color_for_zero = 0):

    from matplotlib.colors import LinearSegmentedColormap
    # Define the custom colormaps
    # Custom Viridis with white for 0
    cmap_custom_zero    = colormap(np.arange(colormap.N))
    cmap_custom_zero[0] = np.array([1, 1, 1, color_for_zero])  # 1 for White, 0 for Transparent
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', cmap_custom_zero)

    return custom_cmap


def Print2Dmaps(hist, ycenters, xcenters, column_index, selected_column, y_label = 'Lost photons', gas_name = 'Ar', pressure = '0bar', **kwargs):

    filtered_hist                   = np.zeros_like(hist)
    filtered_hist[:, column_index]  = selected_column

    # Plot the 2D histogram using matplotlib
    custom_viridis_cmap = DefineCustomMap(colormap = plt.cm.viridis, color_for_zero = 0) # transparent background
    custom_inferno_cmap = DefineCustomMap(colormap = plt.cm.inferno, color_for_zero = 1) # white background

    # Create subplots
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5), constrained_layout=True)

    # First subplot
    im1 = ax[0].imshow(
        hist,
        origin='lower',
        aspect='auto',
        extent=[xcenters[0], xcenters[-1], ycenters[0], ycenters[-1]],
        cmap=custom_viridis_cmap
    )
    fig.colorbar(im1, ax=ax[0], label='Counts')
    ax[0].set_ylabel(y_label)
    ax[0].set_xlabel('Total Photons')
    ax[0].set_title('2D Histogram (Viridis)')

    # Second subplot with two overlaid histograms
    im2 = ax[1].imshow(
        hist,
        origin='lower',
        aspect='auto',
        extent=[xcenters[0], xcenters[-1], ycenters[0], ycenters[-1]],
        cmap=custom_viridis_cmap
    )
    im3 = ax[1].imshow(
        filtered_hist,
        origin='lower',
        aspect='auto',
        extent=[xcenters[0], xcenters[-1], ycenters[0], ycenters[-1]],
        cmap=custom_inferno_cmap,
        **kwargs
    )
    fig.colorbar(im2, ax=ax[1], label='Counts')  # Colorbar for the base histogram
    ax[1].set_ylabel(y_label)
    ax[1].set_xlabel('Total Photons')
    ax[1].set_title('Overlayed 2D Histograms (Inferno)')

    # Show the plots
    plt.suptitle('2D Histograms for ' + gas_name + ' at ' + pressure)
    plt.show()

    return fig, ax


def PrintSlicedHist(x_label, ycenters, hist, **kwargs):

    column_index, selected_column = Slice2Dmap(hist)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7), constrained_layout=True)

    events, bins, image = ax.hist(ycenters, bins = len(ycenters), weights = selected_column, **kwargs)

    # Gaussian fit
    x_data, y_data = ycenters, selected_column

    A       = np.trapz(x = x_data, y = y_data)
    mu      = ycenters[column_index]
    sigma   = y_data.std()

    initial_guess = [A, mu, sigma]
    bounds = (0, np.inf)
    popt, pcov = curve_fit(gaussian, x_data, y_data, p0 = initial_guess, bounds = bounds)

    ax.plot(x_data, gaussian(x_data, *popt), '--r', label = fr'Gaussian fit: A = {popt[0]:.2f}; $\mu$ = {popt[1]:.2f}; $\sigma$ = {popt[2]:.2f}')

    ax.set_xlabel(f'{x_label}');
    ax.set_ylabel(r'Counts');

    return fig, ax, events, bins, popt, pcov


def CalculateCorrections(csv_file, fit = False):

    pressure, string, hist, ycenters, xcenters = Read2Dmap(csv_file)

    column_index, selected_column = Slice2Dmap(hist) 

    total_photons = xcenters[column_index]
    
    if fit:
        # Gaussian fit
        x_data, y_data = ycenters, selected_column

        A       = np.trapz(x = x_data, y = y_data)
        mu      = ycenters[column_index]
        sigma   = y_data.std()

        initial_guess = [A, mu, sigma]
        bounds = (0, np.inf)
        popt, pcov = curve_fit(gaussian, x_data, y_data, p0 = initial_guess, bounds = bounds)

        return pressure, string, total_photons, popt, pcov
    else:
        frequencies     = selected_column
        bin_midpoints   = ycenters
        # Compute weighted mean
        weighted_mean = np.sum(bin_midpoints * frequencies) / np.sum(frequencies)
        # Compute weighted standard deviation
        weighted_variance = np.sum(frequencies * (bin_midpoints - weighted_mean) ** 2) / np.sum(frequencies)
        weighted_std = np.sqrt(weighted_variance)
        # Compute standard error of the mean (SEM)
        total_samples = np.sum(frequencies)
        sem = weighted_std / np.sqrt(total_samples)

        return pressure, string, total_photons, weighted_mean, sem



def CreateCorrectionsDataFrame(csv_files, fit = False):

    df = pd.DataFrame(columns=['pressure', 
                               'max_photon_counts', 
                               'max_teflon_hits', 
                               'max_teflon_hits_err', 
                               'max_source_hits', 
                               'max_source_hits_err'
                               ])
    for file in csv_files:

        if fit:
            pressure, string, total_photons, popt, pcov = CalculateCorrections(file, fit)

            # Standard errors (square root of diagonal elements of the covariance matrix)
            perr = np.sqrt(np.diag(pcov))

            lost_photons      = popt[1]
            lost_photons_err  = perr[1]
        else:
            pressure, string, total_photons, lost_photons, lost_photons_err = CalculateCorrections(file, fit)


        # Check if a row with the same pressure already exists
        if pressure in df['pressure'].values:
            # Update the existing row
            idx = df[df['pressure'] == pressure].index[0]
            df.loc[idx, 'max_photon_counts'] = total_photons
            if string == 'Teflon Photons': 
                df.loc[idx, 'max_teflon_hits']      = lost_photons  # Your teflon hits value
                df.loc[idx, 'max_teflon_hits_err']  = lost_photons_err  # Your teflon hits error
            if string == 'Source Photons':
                df.loc[idx, 'max_source_hits']      = lost_photons  # Your source hits value
                df.loc[idx, 'max_source_hits_err']  = lost_photons_err  # Your source hits error
        else:
            # Create a new row
            if string == 'Teflon Photons':
                new_row = pd.DataFrame([{
                                        'pressure': pressure,
                                        'max_photon_counts': total_photons,
                                        'max_teflon_hits': lost_photons,
                                        'max_teflon_hits_err': lost_photons_err,
                                        'max_source_hits': None,
                                        'max_source_hits_err': None
                                    }])
            if string == 'Source Photons':
                new_row = pd.DataFrame([{
                                        'pressure': pressure,
                                        'max_photon_counts': total_photons,
                                        'max_teflon_hits': None,
                                        'max_teflon_hits_err': None,
                                        'max_source_hits': lost_photons,
                                        'max_source_hits_err': lost_photons_err
                                    }])
            # Concatenate the new row to the DataFrame
            df = pd.concat([df, new_row], ignore_index=True)

    return df



# Gas Time Constants







        





