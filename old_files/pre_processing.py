import librosa
import matplotlib.pyplot as plt
import soundfile as sf
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torchaudio


class KnowYourData:
    def __init__(self, recordings_path, **kwargs):
        self.recordings_path = recordings_path
        self.recordings_list = [os.path.join(recordings_path, f) for f in os.listdir(recordings_path) if f.endswith('.wav')]
        self.sampling_rate_list = []
        self.channels_list = []
        self.duration_list = []

    def analyze(self):
        for recording in self.recordings_list:
            waveform, samplerate = torchaudio.load(recording)

            waveform = waveform / waveform.abs().max()
            print(waveform.min(), waveform.max())

            self.channels_list.append(waveform.shape[0])
            self.sampling_rate_list.append(samplerate)
            self.duration_list.append(waveform.shape[1] / samplerate)

    def plot(self):
        x_labels = [f"{i+1}" for i in range(len(self.recordings_list))]

        self.fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=("Number of Channels", "Sampling Rate (Hz)", "Duration (seconds)"),
            shared_xaxes=True,
            vertical_spacing=0.07
        )

        self.fig.add_trace(
            go.Scatter(x=x_labels, y=self.channels_list, mode='markers', marker=dict(size=6), name="Channels"),
            row=1, col=1
        )

        self.fig.add_trace(
            go.Scatter(x=x_labels, y=self.sampling_rate_list, mode='markers', marker=dict(size=6), name="Sampling Rate"),
            row=2, col=1
        )

        self.fig.add_trace(
            go.Scatter(x=x_labels, y=self.duration_list, mode='markers', marker=dict(size=6), name="Duration"),
            row=3, col=1
        )

        self.fig.update_layout(
            title_text="FSDD Audio Files Metadata Overview",
            showlegend=False,
            margin=dict(l=50, r=50, t=80, b=50)
        )

        self.fig.update_xaxes(title_text="Audio Files", row=3, col=1)
        self.fig.update_yaxes(title_text="Count", row=1, col=1)
        self.fig.update_yaxes(title_text="Hz", row=2, col=1)
        self.fig.update_yaxes(title_text="Seconds", row=3, col=1)

    def show_figure(self):
        self.fig.show()

    def save_figure(self, filename):
        self.fig.write_html(filename)
        print(f"Figure saved as {filename}")


class FSDDPreProcessing(KnowYourData):
    def __init__(self, recordings_path, **kwargs):
        super().__init__(recordings_path, **kwargs)
        self.analyze()

    def plot(self):
        super().plot()

    def show_figure(self):
        super().show_figure()

    def save_figure(self, filename):
        super().save_figure(filename)