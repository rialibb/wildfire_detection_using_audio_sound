from typing import Optional

import torch
from torchaudio import transforms

from features.mel_spectrogram import MelSpectrogram


class MFCC(torch.nn.Module):
    def __init__(
        self,
        n_mels: int = 128,
        num_frequencies: int = 201,
        window_length: int = 200,
        hop_length: Optional[int] = None,
    ) -> None:
        """Mel-Frequency Cepstral-Coefficient feature-extractor.

        Parameters
        ----------
        n_mels: int, optional
            the number of mel filterbanks
        num_frequencies: int, optional
            the number of frequencies to sample on in the Spectrogram. Defaults to 201.
        window_length: int, optional
            the length of the spectrogram window in the Spectrogram, when using forward, in frames. Defaults to 10.
        hop_size: int, optional
            the distance between two spectrogram windows, when using forward, in ms. Defaults to win_length/2.
        """
        super(MFCC, self).__init__()
        self.mel_spectrogram = MelSpectrogram(
            n_mels=n_mels,
            num_frequencies=num_frequencies,
            window_size=window_length,
            hop_size=hop_length,
        )

        TOP_DB = 80.0
        self.amplitude_db = transforms.AmplitudeToDB("power", TOP_DB)
        self.discrete_cosine_transform = torch.F.create_dct(
            self.n_mfcc, self.MelSpectrogram.n_mels, self.norm
        )

    def extract(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """Computes Mel-Frequency Cepstral Coefficients from a mel-spectrogram

        Parameters
        ----------
        mel_spectrogram: torch.Tensor
            the Mel-Spectrogram of the waveform

        Returns
        -------
        torch.Tensor
            Mel-Frequency Cepstral Coefficients
        """

        amplitude_mel_spectrogram = self.amplitude_to_DB(mel_spectrogram)

        # Apply discrete cosine transform
        mfcc = torch.matmul(
            amplitude_mel_spectrogram.transpose(-1, -2), self.discrete_consine_transform
        ).transpose(-1, -2)
        return mfcc

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Computes MFCC from a waveform.


        Parameters
        ----------
        waveform: torch.Tensor
            the waveform whose MFCC should be computed

        Returns
        -------
        torch.Tensor
            Mel-Frequency Cepstral Coefficients
        """

        mel_spec = self.mel_spectrogram(waveform)
        return self.extract(mel_spectrogram=mel_spec)
