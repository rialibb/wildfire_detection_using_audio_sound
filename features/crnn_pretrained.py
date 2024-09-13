"""
The code in this file is split into two parts.

The first part, coded by ourselves, is the  CRNNPretrainedFeature class that makes a call to the pretrained model.
The second part, taken from Hugging Face, is the structure of the pretrained model.

"""

import os

import torch
from torchaudio import transforms
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.utils.hub import cached_file
import torchaudio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CRNNPretrainedFeature(torch.nn.Module):

    output_shape = (1, 447)

    def __init__(self, initial_sampling_rate: int = 44100):
        """ Initializes the AstEncoder instance

        Parameters:
            initial_sampling_rate (`int`, *optional*):
                            The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
                            `sampling_rate` at the forward call to prevent silent errors.

        """
        super(CRNNPretrainedFeature, self).__init__()
        self.crnn_pretrained            = Cnn8RnnSoundEventDetection.from_pretrained("wsntxxn/cnn8rnn-audioset-sed",
                                                                                     trust_remote_code=True)
        self.initial_sampling_rate      = initial_sampling_rate
        self.crnn_sampling_rate         = self.crnn_pretrained.config.sample_rate
        self.resampler                  = torchaudio.transforms.Resample(orig_freq=self.initial_sampling_rate, new_freq=self.crnn_sampling_rate)

        # The CRNN Model should not be retrained
        self.crnn_pretrained.training         = False
        # Freezing every weight of CRNN
        for p in self.crnn_pretrained.parameters():
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
        waveform_resampled      = self.resampler(waveform)
        waveform_resampled.to(device)
        # Remove one dimension so that the shapes matches the CRNN input requirements
        waveform_resampled = waveform_resampled.squeeze(1)
        # forward pass through the CRNN model
        output_model            = self.crnn_pretrained(waveform_resampled)

        return output_model



""" 

    The following code comes from https://huggingface.co/wsntxxn/cnn8rnn-audioset-sed/tree/main
    and is put there so that we can see the model's structure.
    The weighs are not stored here, they are downloaded from Hugging face's hub in CRNNPretrainedFeature.__init__
    
"""

class Cnn8RnnConfig(PretrainedConfig):

    def __init__(
            self,
            classes_num: int = 447,
            sample_rate: int = 32000,
            **kwargs
    ):
        self.classes_num = classes_num
        self.sample_rate = sample_rate
        super().__init__(**kwargs)


class Cnn8RnnSoundEventDetection(PreTrainedModel):
    config_class = Cnn8RnnConfig

    def __init__(self, config: Cnn8RnnConfig):
        super().__init__(config)
        self.config = config
        self.time_resolution = 0.01
        self.interpolate_ratio = 4  # Downsampled ratio

        # Logmel spectrogram extractor
        self.melspec_extractor = transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=1024,
            win_length=1024,
            hop_length=320,
            f_min=50,
            f_max=14000,
            n_mels=64,
            norm="slaney",
            mel_scale="slaney"
        )
        self.db_transform = transforms.AmplitudeToDB()

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.rnn = nn.GRU(512, 256, bidirectional=True, batch_first=True)
        self.fc_audioset = nn.Linear(512, config.classes_num, bias=True)
        self.temporal_pooling = LinearSoftmax()

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, waveform):
        x = self.melspec_extractor(waveform)
        x = self.db_transform(x)  # (batch_size, mel_bins, time_steps)
        x = x.transpose(1, 2)
        x = x.unsqueeze(1)

        frames_num = x.shape[2]

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg+max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg+max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(1, 2), pool_type='avg+max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(1, 2), pool_type='avg+max')
        x = F.dropout(x, p=0.2, training=self.training)  # (batch_size, 256, time_steps / 4, mel_bins / 16)
        x = torch.mean(x, dim=3)

        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        x, _ = self.rnn(x)
        segmentwise_output = torch.sigmoid(self.fc_audioset(x)).clamp(1e-7, 1.)
        clipwise_output = self.temporal_pooling(segmentwise_output)

        # # Get framewise output
        # framewise_output = interpolate(segmentwise_output,
        #                                self.interpolate_ratio)
        # framewise_output = pad_framewise_output(framewise_output, frames_num)

        return clipwise_output

    def save_pretrained(self, save_directory, *args, **kwargs):
        super().save_pretrained(save_directory, *args, **kwargs)
        with open(os.path.join(save_directory, "classes.txt"), "w") as f:
            for class_name in self.classes:
                f.write(class_name + "\n")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                        **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path,
                                        *model_args, **kwargs)
        class_file = cached_file(pretrained_model_name_or_path, "classes.txt")
        with open(class_file, "r") as f:
            model.classes = [l.strip() for l in f]
        return model




def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


def interpolate(x, ratio):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.

    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output, frames_num):
    """Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    pad = framewise_output[:, -1:, :].repeat(1, frames_num - framewise_output.shape[1], 1)
    """tensor for padding"""

    output = torch.cat((framewise_output, pad), dim=1)
    """(batch_size, frames_num, classes_num)"""

    return output


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x


class LinearSoftmax(nn.Module):
    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim

    def forward(self, time_decision):
        return (time_decision ** 2).sum(self.pooldim) / time_decision.sum(
            self.pooldim)


