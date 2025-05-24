import librosa
import matplotlib.pyplot as plt
import soundfile as sf
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Load the audio file at a fixed sampling rate (e.g., 8000 Hz)
data_len = 1

sampling_rate_list = []
channels_list = []
duration_list = []

recordings_path = '/home/pavan/Music/spectrum/Innatera/recordings/'
recordings_list = [os.path.join(recordings_path, f) for f in os.listdir(recordings_path) if f.endswith('.wav')]

print(len(recordings_list))

# ## Obtain the the raw temporal structure of the sound.
for recording in recordings_list:
    # y, sr = librosa.load(audio_path, sr=None)

    f = sf.SoundFile(recording)

    channels_list.append(f.channels)
    sampling_rate_list.append(f.samplerate)
    duration_list.append(len(f) / f.samplerate)

    # print(f"Channels: {f.channels}")
    # print(f"Sample Rate: {f.samplerate}")
    # print(f"Duration: {len(f) / f.samplerate:.2f} seconds")

    # print(y.shape, sr, len(y)/sr)  # Length of the audio signal

x_labels = [f"{i+1}" for i in range(len(recordings_list))]


# Create 3-row subplot figure
fig = make_subplots(
    rows=3, cols=1,
    subplot_titles=("Number of Channels", "Sampling Rate (Hz)", "Duration (seconds)"),
    shared_xaxes=True,
    vertical_spacing=0.07
)

# Scatter plot instead of bar or line
fig.add_trace(
    go.Scatter(x=x_labels, y=channels_list, mode='markers', marker=dict(size=6), name="Channels"),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=x_labels, y=sampling_rate_list, mode='markers', marker=dict(size=6), name="Sampling Rate"),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=x_labels, y=duration_list, mode='markers', marker=dict(size=6), name="Duration"),
    row=3, col=1
)

# Layout for full screen
fig.update_layout(
    title_text="FSDD Audio Files Metadata Overview",
    showlegend=False,
    # height=1000,  # Large height for full screen
    # width=1800,   # Wide enough for large screens
    margin=dict(l=50, r=50, t=80, b=50)
)

fig.update_xaxes(title_text="Audio Files", row=3, col=1)
fig.update_yaxes(title_text="Count", row=1, col=1)
fig.update_yaxes(title_text="Hz", row=2, col=1)
fig.update_yaxes(title_text="Seconds", row=3, col=1)

fig.show()