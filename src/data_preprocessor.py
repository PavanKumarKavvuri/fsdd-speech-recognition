import torch
import torchaudio
import torch.nn.functional as F
import torchaudio.transforms as T

# 2. Dataset class
class SpokenDigitDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None, denoise_data=False, sample_rate=8000):
        self.data = data
        self.transform = transform
        self.denoise_data = denoise_data
        self.sample_rate = sample_rate
        self.mfcc_transform = T.MFCC(
            sample_rate=self.sample_rate,         
            n_mfcc=13,                 
            melkwargs={
                "n_fft": 400,
                "hop_length": 160,
                "n_mels": 40
            }
        )

    def __len__(self):
        return len(self.data)

    def apply_bandpass_filter(self, waveform, sample_rate):
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

    def pad_or_trim_mfcc(self, mfcc, max_len=36):
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

    def __getitem__(self, idx):
        audio_path, label = self.data[idx]

        waveform, samplerate = torchaudio.load(audio_path)
        if self.denoise_data:
            waveform = self.apply_bandpass_filter(waveform, samplerate)
        waveform = waveform / waveform.abs().max()

        mfcc_features = self.mfcc_transform(waveform)
        updated_mfcc_features = self.pad_or_trim_mfcc(mfcc_features)

        return updated_mfcc_features, label

# --