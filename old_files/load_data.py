import librosa
import matplotlib.pyplot as plt
import soundfile as sf
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torchaudio
import torchaudio.transforms as T
import numpy as np
import torch.nn.functional as F


show_plot = True
denoise_data = False
percentile_of_timesteps = 99

# Load the audio file at a fixed sampling rate (e.g., 8000 Hz)
data_len = 1

sampling_rate_list = []
channels_list = []
duration_list = []
time_steps_list = []

encoded_labels = []


recordings_path = '/home/pavan/Music/spectrum/Innatera/recordings/'
recordings_list = [os.path.join(recordings_path, f) for f in os.listdir(recordings_path) if f.endswith('.wav')]
print(len(recordings_list))

def apply_bandpass_filter(waveform, sample_rate):
    """
    Applies bandpass filter (300–3000 Hz) and gain normalization to the waveform.
    Returns the processed waveform.
    """
    effects = [
        ['bandpass', '300', '3000'],  # speech frequency range
        ['gain', '-n']                # normalize to 0 dB
    ]

    processed_waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
        waveform, sample_rate, effects
    )

    return processed_waveform


def pad_or_trim_mfcc(mfcc, max_len=36):
    """
    Pads or trims the MFCC array along the time axis to make its shape (n_mfcc, max_len).
    
    Parameters:
        mfcc (np.ndarray): MFCC feature array of shape (n_mfcc, time_steps)
        max_len (int): Desired number of time steps

    Returns:
        np.ndarray: MFCC array of shape (n_mfcc, max_len)
    """
    _, _, time_steps = mfcc.shape

    pad_width = max_len - time_steps

    if time_steps < max_len:
        # # Pad with zeros at the end
        padded_mfcc = F.pad(mfcc, (0, pad_width))
        return padded_mfcc
    
    elif time_steps > max_len:
        # # Trim to max_len
        trimmed_mfcc = mfcc[:, :, :max_len]
        return trimmed_mfcc
    
    return mfcc  # Already the correct size


# ## Obtain the the raw temporal structure of the sound.
for recording in recordings_list[:]:

    filename = os.path.basename(recording)
    label = int(filename.split('_')[0])  
    encoded_labels.append(label)

    ## Waveform represents air pressure variations over time
    waveform, samplerate = torchaudio.load(recording)

    # Apply noise reduction preprocessing
    if denoise_data:
        waveform = apply_bandpass_filter(waveform, samplerate)

    waveform = waveform / waveform.abs().max()

    # print(waveform.min(), waveform.max())

    # print(waveform.shape[1], len(waveform[0]))

    channels_list.append(waveform.shape[0])
    sampling_rate_list.append(samplerate)
    duration_list.append(waveform.shape[1] / samplerate)

    mfcc_transform = T.MFCC(
        sample_rate=samplerate,         # or 8000, depending on your waveform
        n_mfcc=13,                 # typically 13–40 coefficients
        melkwargs={
            "n_fft": 400,
            "hop_length": 160,
            "n_mels": 40
        }
    )

    mfcc_features = mfcc_transform(waveform)
    _, _, t_steps = mfcc_features.shape

    # print(mfcc_features.shape)

    updated_mfcc_features = pad_or_trim_mfcc(mfcc_features)

    # print(updated_mfcc_features)
    _, _, updated_t_steps = updated_mfcc_features.shape
    
    
    if updated_t_steps == 36:
        time_steps_list.append(updated_t_steps)
    else:
        print(updated_mfcc_features.shape)



unique_labels = sorted(set(encoded_labels))
print("Unique labels found:", unique_labels)








































# Percentiles to calculate
percentiles = [50, 75, 85, 90, 95, 99]

'''

If you set max_len = 23, you won’t need to pad, but you’ll trim 50% of your samples.
If you set max_len = 29, you only trim 10% of samples and pad the rest.
If you set max_len = 36, you keep 99% of your data untrimmed, but pad many of the shorter samples.

'''
percentile_values = np.percentile(time_steps_list, percentiles)

print("Percentiles of MFCC sequence lengths:", percentile_values)

# Create subplots: 1 row, 2 columns
fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=("MFCC Sequence Length Histogram", "Percentiles of Sequence Lengths"),
    horizontal_spacing=0.15
)


hist, bin_edges = np.histogram(time_steps_list, bins=np.arange(min(time_steps_list), max(time_steps_list)+5))  # from 8 to 116
bin_labels = [f"[{bin_edges[i]}, {bin_edges[i+1]})" for i in range(len(hist))]

print("Maximum MFCC time-steps:", max(time_steps_list))
print("Minimum MFCC time-steps:", min(time_steps_list))

fig.add_trace(
    go.Bar(
        x=bin_labels,
        y=hist,
        name='Histogram',
        # marker=dict(color='skyblue'),
        hovertemplate='Bin Range: %{x}<br>Count: %{y}<extra></extra>',
    ),
    row=1, col=1
)

# Percentile line plot (right)
fig.add_trace(
    go.Scatter(
        x=[str(p) + "%" for p in percentiles],
        y=percentile_values,
        mode='lines+markers',
        name='Percentiles',
        marker=dict(size=8, color='crimson'),
        hovertemplate='Percentile: %{x}<br>Length: %{y}<extra></extra>',
    ),
    row=2, col=1
)

fig.update_layout(
    title="Histogram of MFCC sequence lengths",
    xaxis_title="Time Step Bin",
    yaxis_title="Sample Count",
    xaxis_tickangle=-45,
    # width=1200, height=400
)

if show_plot:
    fig.show()




















































# x_labels = [f"{i+1}" for i in range(len(recordings_list))]


# # Create 3-row subplot figure
# fig = make_subplots(
#     rows=3, cols=1,
#     subplot_titles=("Number of Channels", "Sampling Rate (Hz)", "Duration (seconds)"),
#     shared_xaxes=True,
#     vertical_spacing=0.07
# )

# # Scatter plot instead of bar or line
# fig.add_trace(
#     go.Scatter(x=x_labels, y=channels_list, mode='markers', marker=dict(size=5), name="Channels"),
#     row=1, col=1
# )

# fig.add_trace(
#     go.Scatter(x=x_labels, y=sampling_rate_list, mode='markers', marker=dict(size=5), name="Sampling Rate"),
#     row=2, col=1
# )

# fig.add_trace(
#     go.Scatter(x=x_labels, y=duration_list, mode='markers', marker=dict(size=5), name="Duration"),
#     row=3, col=1
# )

# # Layout for full screen
# fig.update_layout(
#     title_text="FSDD Audio Files Metadata Overview",
#     showlegend=False,
#     # height=1000,  # Large height for full screen
#     # width=1800,   # Wide enough for large screens
#     margin=dict(l=50, r=50, t=80, b=50)
# )

# fig.update_xaxes(title_text="Audio Files", row=3, col=1)
# fig.update_yaxes(title_text="Count", row=1, col=1)
# fig.update_yaxes(title_text="Hz", row=2, col=1)
# fig.update_yaxes(title_text="Seconds", row=3, col=1)

# fig.show()