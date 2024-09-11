import torch
from transformers import ASTFeatureExtractor, ASTModel
import torchaudio
# Load the pre-trained AST model and feature extractor
HUGGING_FACE_AST_MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
# Useful documentation can be found at: https://huggingface.co/docs/transformers/main/en/model_doc/audio-spectrogram-transformer#audio-spectrogram-transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AstEncoder(torch.nn.Module):

    output_shape = (1, 768)

    def __init__(self, initial_sampling_rate: int = 44100, ast_sampling_rate: int = 16000):
        """ Initializes the AstEncoder instance

        Parameters:
            initial_sampling_rate (`int`, *optional*):
                            The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
                            `sampling_rate` at the forward call to prevent silent errors.
            ast_sampling_rate ('int', *optional*)
                            The sampling rate needed for AST

        """
        super(AstEncoder, self).__init__()
        self.feature_extractor          = ASTFeatureExtractor.from_pretrained(HUGGING_FACE_AST_MODEL_NAME)
        self.ast_model                  = ASTModel.from_pretrained(HUGGING_FACE_AST_MODEL_NAME)

        self.initial_sampling_rate      = initial_sampling_rate
        self.ast_sampling_rate          = ast_sampling_rate
        self.resampler                  = torchaudio.transforms.Resample(orig_freq=self.initial_sampling_rate, new_freq=self.ast_sampling_rate)

        # The AST Model should not be retrained
        self.ast_model.training         = False
        # Freezing every weight of AST
        for p in self.ast_model.parameters():
            p.requires_grad = False

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Computes the AST model encoding.

        Parameters
        ----------
        waveform: torch.Tensor
            the waveform as a torch tensor

        Returns
        -------
        torch.Tensor
            the output of AST model's encoder
        """
        waveform_resampled                = self.resampler(waveform)
        # AstFeatureExtractor expects numpy input. Keeping a tensor throws an error
        waveform_np             = waveform_resampled.cpu().numpy()
        input_ast_encoder       = self.feature_extractor(waveform_np.squeeze(1), sampling_rate=self.ast_sampling_rate, return_tensors="pt")
        output_model            = self.ast_model(input_ast_encoder['input_values'].to(device),
                                                 output_attentions=False,
                                                 output_hidden_states=False,
                                                 return_dict=True)

        encoder_output      = output_model.last_hidden_state
        # Retrieve only first dimension that corresponds to the dimension of the special <CLS> token
        audio_encoding      = encoder_output[:, 0, :]

        return audio_encoding

# TODO test code : to delete when pipeline is working
#ast_encoder = AstEncoder()
#audio =  torchaudio.load("../data/esc50/audio/1-137-A-32.wav")
# audio_resampled = transform(audio[0])
# output_encoder = ast_encoder(audio[0])
# print(output_encoder.shape)
