
import pandas as pd
import re
import struct
import numpy as np
from pathlib import Path

def parse_txt_to_dataframe(file_path):
    data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    event_num = None
    event_time = None
    time_step = None
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Capture Event number
        if line.startswith("Event n."):
            event_num = int(re.search(r'\d+', line).group())
        
        # Capture TimeStamp
        elif line.startswith("TimeStamp:"):
            event_time = int(re.search(r'\d+', line).group())
        
        # Capture 1 Sample step value
        elif line.startswith("1 Sample ="):
            time_step = float(re.search(r'0\.[0-9]+', line).group())
            
        # Capture data values
        elif re.match(r'^\d+\s+-?\d+\.\d+', line):
            s, ch2 = line.split()
            s = int(s)
            ch2 = float(ch2)/1000
            time = s * time_step/1e6
            data.append((time, ch2, event_num, event_time))
            
    # Create DataFrame
    df = pd.DataFrame(data, columns=["TIME", "CH2", "event", "event_time"])
    return df
 
def parse_txt_to_dataframe_multich(file_path):
    data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()

    event_num = None
    event_time = None
    time_step = None
    channels = []
    
    for i, line in enumerate(lines):
        line = line.strip()

        # Capture Event number
        if line.startswith("Event n."):
            event_num = int(re.search(r'\d+', line).group())
                        
        # Capture TimeStamp
        elif line.startswith("TimeStamp:"):
            event_time = int(re.search(r'\d+', line).group())

        # Capture 1 Sample step value
        elif line.startswith("1 Sample ="):
            time_step = float(re.search(r'0\.[0-9]+', line).group())

        # Capture channels
        elif line.startswith("S") and "CH:" in line :
            channels = [f"CH{i+1}" for i in range(line.count("CH:"))]

        # Capture data values
        elif re.match(r'^\d+', line):
            values = line.split()
            s = int(values[0])
            time = s * time_step / 1e6
            ch_values = [float(v) / 1000 for v in values[1:]]
            data.append((time, *ch_values, event_num, event_time))
        
    # Create DataFrame
    column_names = ["TIME"] + channels + ["event", "event_time"]
    df = pd.DataFrame(data, columns=column_names)
    return df


def parse_wf_from_binary(
    path,
    *,
    channels=4,
    n_events=500,
    file_idx = 0,
    dtype="<f4",          # little-endian float32
    event_header_bytes=28, # size of per-event header in bytes
    sample_binning=8e-9    # seconds per sample (default 8 ns)
):
    """
    Reads binary waveform data and returns a tidy pandas DataFrame.

    Each event layout: [event_header_bytes] + [channels * samples_per_waveform * dtype]

    Args:
        path (str or Path): Path to the binary file
        channels (int): Number of waveform channels
        n_events (int or None): Number of events to read. If None, read all events in file.
        dtype (str): Data type of waveform samples
        event_header_bytes (int): Size of per-event header in bytes (0 if none)
        sample_binning (float): Sample spacing in seconds

    Returns:
        df (pd.DataFrame): Columns = TIME, CH1..CHn, event, event_time
    """
    path = Path(path)
    file_size = path.stat().st_size
    sample_bytes = np.dtype(dtype).itemsize

    if n_events is None:
        raise ValueError("You must provide n_events (no default samples_per_waveform available).")

    # Compute number of samples per waveform
    bytes_per_event = file_size // n_events
    data_bytes_per_event = bytes_per_event - event_header_bytes
    if data_bytes_per_event % (channels * sample_bytes) != 0:
        raise ValueError("File size does not match given n_events and channels.")
    samples_per_waveform = data_bytes_per_event // (channels * sample_bytes)

    records = []

    with path.open("rb") as f:
        for evt in range(n_events):
            # ---- Read header ----
            event_id = None
            event_time = None
            if event_header_bytes:
                h = f.read(event_header_bytes)
                if len(h) < event_header_bytes:
                    raise RuntimeError(f"Unexpected EOF while reading header for event {evt}")
                # interpret header minimally like SECOND: EVID + T
                if event_header_bytes >= 12:
                    event_id = struct.unpack("<I", h[0:4])[0]
                    event_time = struct.unpack("<Q", h[4:12])[0]

            # ---- Read waveform ----
            buf = f.read(data_bytes_per_event)
            if len(buf) < data_bytes_per_event:
                raise RuntimeError(f"Unexpected EOF while reading waveform for event {evt}")

            arr = np.frombuffer(buf, dtype=dtype).reshape(channels, samples_per_waveform)

            # ---- Compute unique event ID ----
            global_event_id = file_idx * n_events + evt

            # ---- Build tidy rows ----
            for i in range(samples_per_waveform):
                row = {
                        "TIME": (i + 1) * sample_binning,  # TIME first
                        **{f"CH{ch+1}": float(arr[ch, i]) for ch in range(channels)},  # channel values
                        "event": global_event_id,
                        "event_time": event_time if event_time is not None else 0,
                        'file_idx':file_idx
                    }
                records.append(row)

    df = pd.DataFrame(records)
    return df

def load_waveforms_until_eof(
    path,
    *,
    channels=4,
    samples_per_waveform=752,
    dtype="<f4",          # little-endian float32
    event_header_bytes=28 # set to 0 if there's no per-event header
):
    """
    Reads events until EOF.
    Each event layout: [event_header_bytes] + [channels * samples_per_waveform * dtype]
    Returns:
      waveforms: (num_events, channels, samples_per_waveform) array
      headers:   (num_events, event_header_bytes//4) <u4 array (or None if header_bytes==0)
    """
    path = Path(path)
    sample_bytes = np.dtype(dtype).itemsize
    data_bytes_per_event = channels * samples_per_waveform * sample_bytes

    wfs = []
    hdrs = [] if event_header_bytes else None

    with path.open("rb") as f:
        evt = 0
        while True:
            # Read per-event header (if any)
            if event_header_bytes:
                h = f.read(event_header_bytes)
                if not h:
                    break  # clean EOF at boundary
                if len(h) != event_header_bytes:
                    print(f"Warning: partial header at event {evt} — stopping.")
                    break
                hdrs.append(np.frombuffer(h, dtype="<u4"))
            else:
                # If no header, peek one byte to see if we're at EOF
                p = f.peek(1) if hasattr(f, "peek") else f.read(1)
                if p == b"":
                    break
                if not hasattr(f, "peek"):
                    # we consumed 1 byte; seek back
                    f.seek(-1, 1)

            # Read waveform payload
            buf = f.read(data_bytes_per_event)
            if len(buf) != data_bytes_per_event:
                print(f"Warning: partial data payload at event {evt} — stopping.")
                break

            arr = np.frombuffer(buf, dtype=dtype).reshape(channels, samples_per_waveform)
            wfs.append(arr)
            evt += 1

    if not wfs:
        raise RuntimeError("No complete events found.")

    waveforms = np.stack(wfs, axis=0)  # (E, C, N)
    headers = (np.stack(hdrs, axis=0) if hdrs is not None else None)
    return waveforms, headers

# def parse_wf_from_binary(filename):
#     data_list = []
#     nlines=0
#     nevents=300
#     with open(filename, "rb") as f:
#         while True:
#             # Read the header
#             data = f.read(4)  # Read uint32_t EVID
#             if not data:
#                 break
#             EVID = struct.unpack("<I", data)[0]
#             data = f.read(8)  # Read uint64_t T
#             if not data:
#                 break
#             T = struct.unpack("<Q", data)[0]
#             data = f.read(4)  # Read uint32_t size
#             if not data:
#                 break
#             size = struct.unpack("<I", data)[0]
#             data = f.read(8)  # Read uint64_t sampl_time
#             if not data:
#                 break
#             sampl_time = struct.unpack("<Q", data)[0]
#             data = f.read(4)  # Read uint32_t ch (number of channels)
#             if not data:
#                 break
#             ch = struct.unpack("<I", data)[0]
#             waveform_data = {}
#             # Read waveforms for each channel
#             for channel in range(ch):
#                 data = f.read(2)  # Read uint16_t numch
#                 if not data:
#                     break
#                 numch = struct.unpack("<H", data)[0]
#                 channel_waveforms = []
#                 for _ in range(size):
#                     data = f.read(4)  # Read float w
#                     if not data:
#                         break
#                     w = struct.unpack("<f", data)[0]
#                     channel_waveforms.append(w)
#                 waveform_data[f'{numch}'] = channel_waveforms
#             # Create a row per sample point with all channels aligned
#             for i in range(size):
#                 row = {}
                
#                 row.update({f'CH{j+1}': waveform_data[f'{numch}'][i]/1e3 for j,numch in enumerate(waveform_data)})
#                 row.update({"event": EVID})
#                 row.update({"event_time": T})

#                 data_list.append(row)

#     print(nlines,nevents)
#     df = pd.DataFrame(data_list)
#     df.insert(0, 'TIME', (df.index % size + 1) * sampl_time/1e9)  # Time in microseconds

#     return df
