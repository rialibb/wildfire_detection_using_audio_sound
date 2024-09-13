import torch
from transformers import ASTFeatureExtractor
import torchaudio

# Useful documentation can be found at: https://huggingface.co/docs/transformers/main/en/model_doc/audio-spectrogram-transformer#audio-spectrogram-transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AstFeatureZero(torch.nn.Module):

    output_shape =  None

    def __init__(self, initial_sampling_rate: int = 44100, ast_sampling_rate: int = 16000):
        """ Initializes the AstEncoder instance

        Parameters:
            initial_sampling_rate (`int`, *optional*):
                            The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
                            `sampling_rate` at the forward call to prevent silent errors.
            ast_sampling_rate ('int', *optional*)
                            The sampling rate needed for AST

        """
        super(AstFeatureZero, self).__init__()
        self.feature_extractor = ASTFeatureExtractor()

        self.initial_sampling_rate      = initial_sampling_rate
        self.ast_sampling_rate          = ast_sampling_rate
        self.resampler                  = torchaudio.transforms.Resample(orig_freq=self.initial_sampling_rate, new_freq=self.ast_sampling_rate)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Computes the AST model encoding.

        Parameters
        ----------
        waveform: torch.Tensor
            the waveform as a torch tensor

        Returns
        -------
        torch.Tensor
            the output of AST Feature Extractor
        """
        waveform_resampled                = self.resampler(waveform)
        # AstFeatureExtractor expects numpy input. Keeping a tensor throws an error
        waveform_np             = waveform_resampled.cpu().numpy()
        # use built-in feature extractor specific to AST. Returns torch tensors
        input_ast_encoder       = self.feature_extractor(waveform_np.squeeze(1), sampling_rate=self.ast_sampling_rate, return_tensors="pt")
        # forward pass through the AST model

        return input_ast_encoder
