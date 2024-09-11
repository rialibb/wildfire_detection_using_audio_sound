import torch

from classifiers.nn_utils import Flattening, LocalResponseNorm, CNNLayer


class MLPPostAst(torch.nn.Module):
    def __init__(self, input_size:tuple[int,int]) -> tuple[int,int]:


        super(MLPPostAst,self).__init__()

        self.flatten=Flattening()


        self.linear_layers=torch.nn.Sequential(
            torch.nn.Linear(in_features=input_size[0]*input_size[1],out_features=512),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=512,out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256,out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64,out_features=2)
        )


    def forward(self,x:torch.Tensor) -> torch.Tensor:

        x=self.flatten(x)
        x=self.linear_layers(x)

        return x