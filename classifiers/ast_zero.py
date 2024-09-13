import torch

from classifiers.nn_utils import ESC
from transformers import ASTModel, ASTConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AstZero(torch.nn.Module):
    def __init__(self, input_size:tuple[int,int],approach:ESC) -> tuple[int,int]:
        """
        MLPostAst model(MLP layers after the AST model) 
        These layers takes the output of the pre-trained AST model and returns the logits for each class

        Parameters
        ----------
        input_size : tuple
            -- Input size (length, height)
        approach: ESC
            the type of approach (ESC2, ESC10 or ESC50)
        """

        super(AstZero,self).__init__()

        config = ASTConfig()
        self.ast_model = ASTModel(config=config)

        self.linear_layers=torch.nn.Sequential(
            torch.nn.Linear(self.ast_model.config.hidden_size ,out_features=512),
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
        
        # forward pass through the AST model
        output_model            = self.ast_model(x['input_values'].to(device),
                                                 output_attentions=False,
                                                 output_hidden_states=False,
                                                 return_dict=True)

        encoder_output      = output_model.last_hidden_state
        # Retrieve only first dimension that corresponds to the dimension of the special <CLS> token
        audio_encoding      = encoder_output[:, 0, :]
        
        output =self.linear_layers(audio_encoding)

        return output