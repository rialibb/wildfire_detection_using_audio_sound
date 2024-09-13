import torch

from classifiers.nn_utils import LocalResponseNorm, CNNLayer,ESC


class MlpPostCrnnPetrained(torch.nn.Module):

    def __init__(self, input_size:tuple[int,int],approach:ESC) -> None:
        """
        MlpPostCRnnPetrained model(MLP layers after the pretrained CRNN model)
        These layers takes the output of the pre-trained AST model and returns the logits for each class

        Parameters
        ----------
        input_size : tuple
            -- Input size (length, height)
        approach: ESC
            the type of approach (ESC2, ESC10 or ESC50)
        """

        super(MlpPostCrnnPetrained,self).__init__()


        self.linear_layers=torch.nn.Sequential(
            torch.nn.Linear(in_features=input_size[0]*input_size[1],out_features=512),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=512,out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256,out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64,out_features=approach.value)
        )


    def forward(self,x:torch.Tensor) -> torch.Tensor:
        """
        Forward method for the MLPostAst model for AST
        Computes and returns prediction of the model for the data x.

        Parameters
        ----------
        x : torch.Tensor
            -- Input data tensor
        """
        x=self.linear_layers(x)

        return x