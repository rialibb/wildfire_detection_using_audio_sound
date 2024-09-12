import torch

from classifiers.nn_utils import CNNLayer,ESC

class ConvolutionalRNNZhang(torch.nn.Module):
    def __init__(self, input_size: tuple[int, int],approach:ESC) -> tuple[int, int]:
        """
        ConvolutionalRNN (Convolutional Reccurrent Neural Networks model from Zhang paper
        (Attention based Convolutional Recurrent Neural Network for Environmental Sound
        Classification)) class constructor.

        Parameters
        ----------
        input_size : tuple
            -- Input size (length, height)
        approach: ESC
            the type of approach (ESC2, ESC10 or ESC50)
        """
        super(ConvolutionalRNNZhang, self).__init__()

        # define the first Conv layer
        self.conv_layer1 = CNNLayer(
            in_channels=1,
            out_channels=32,
            cnn_kernel_size=(3, 5),
            cnn_stride=(1, 1),
            mp_kernel_size=(1, 1),
            mp_stride=(1, 1),
        )

        # define the second Conv layer
        self.conv_layer2 = CNNLayer(
            in_channels=32,
            out_channels=32,
            cnn_kernel_size=(3, 5),
            cnn_stride=(1, 1),
            mp_kernel_size=(4, 3),
            mp_stride=(4, 3),
        )

        # define the third Conv layer
        self.conv_layer3 = CNNLayer(
            in_channels=32,
            out_channels=64,
            cnn_kernel_size=(3, 1),
            cnn_stride=(1, 1),
            mp_kernel_size=(1, 1),
            mp_stride=(1, 1),
        )

        # define the forth Conv layer
        self.conv_layer4 = CNNLayer(
            in_channels=64,
            out_channels=64,
            cnn_kernel_size=(3, 1),
            cnn_stride=(1, 1),
            mp_kernel_size=(4, 1),
            mp_stride=(4, 1),
        )

        # define the fifth Conv layer
        self.conv_layer5 = CNNLayer(
            in_channels=64,
            out_channels=128,
            cnn_kernel_size=(1, 5),
            cnn_stride=(1, 1),
            mp_kernel_size=(1, 1),
            mp_stride=(1, 1),
        )

        # define the sixth Conv layer
        self.conv_layer6 = CNNLayer(
            in_channels=128,
            out_channels=128,
            cnn_kernel_size=(1, 5),
            cnn_stride=(1, 1),
            mp_kernel_size=(1, 3),
            mp_stride=(1, 3),
        )

        # define the seventh Conv layer
        self.conv_layer7 = CNNLayer(
            in_channels=128,
            out_channels=256,
            cnn_kernel_size=(3, 3),
            cnn_stride=(1, 1),
            mp_kernel_size=(1, 1),
            mp_stride=(1, 1),
        )

        # define the eighth Conv layer
        self.conv_layer8 = CNNLayer(
            in_channels=256,
            out_channels=256,
            cnn_kernel_size=(3, 3),
            cnn_stride=(1, 1),
            mp_kernel_size=(2, 2),
            mp_stride=(2, 2),
        )

        # find the output size of the Conv layers 
        input_size = self.conv_layer1.get_output_size(input_size=input_size)
        input_size = self.conv_layer2.get_output_size(input_size=input_size)
        input_size = self.conv_layer3.get_output_size(input_size=input_size)
        input_size = self.conv_layer4.get_output_size(input_size=input_size)
        input_size = self.conv_layer5.get_output_size(input_size=input_size)
        input_size = self.conv_layer6.get_output_size(input_size=input_size)
        input_size = self.conv_layer7.get_output_size(input_size=input_size)
        input_size = self.conv_layer8.get_output_size(input_size=input_size)

        # define a GRU layer
        self.gru_layer = torch.nn.GRU(
            256, 256, num_layers=2, bidirectional=True, batch_first=True
        )

        # define a Dropout layer
        self.dropout_layer = torch.nn.Dropout(p=0.5)

        # define a Tanh activation layer
        self.tanh_layer = torch.nn.Tanh()

        # define a Linear layer
        self.linear_layers = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(256, approach.value),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method for the CRNN model from Zhang.
        Computes and returns prediction of the model for the data x.

        Parameters
        ----------
        x : torch.Tensor
            -- Input data tensor
        """
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = self.conv_layer5(x)
        x = self.conv_layer6(x)
        x = self.conv_layer7(x)
        x = self.conv_layer8(x)
        x = x.permute(0, 2, 3, 1)
        x = x.view(x.size(0), -1, x.size(3))
        x, _ = self.gru_layer(x)
        x = self.dropout_layer(x)
        x = self.tanh_layer(x)
        x = x.mean(dim=1)
        x = self.linear_layers(x)
        return x
