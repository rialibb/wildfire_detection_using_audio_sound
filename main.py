from classifiers.cnn_bardou import CNNBardou
from classifiers.crnn_zhang import ConvolutionalRNNZhang
from classifiers.mlp_post_ast import MLPPostAst
from classifiers.ast_zero import AstZero
from datasets import ESCDataset, ESCDatasetBin, ESC
from features import Spectrogram, MelSpectrogram, Cochleagram, AstEncoder, AstFeatureZero
from models import train, FEATURES, generate_models

BATCH_SIZE = 32

MODEL_KWARGS = {"input_size": (256, 2206)} #size of a spectrogram/mel-spectrogram

APPROACH = ESC.TEN  # change the approach based on 2, 10 or 50 classes

#load de the dataset based on the approach selected
esc_dataset = ESCDataset(download=False,categories=APPROACH) if APPROACH.value!=2 else ESCDatasetBin(download=False, train_index= list(range(2744)))


print(f"We are using {esc_dataset} for the dataset")
print(f"Type of approach : {APPROACH}")

#Uncomment the model to use

#bardou_models = generate_models({"spectrogram": Spectrogram}, CNNBardou, "cnn_bardou",approach=APPROACH)
#zhang_models = generate_models(FEATURES, ConvolutionalRNNZhang, "crnn_zhang", approach=APPROACH)
#ast_model = generate_models({"ast_Encoder": AstEncoder}, MLPPostAst, "MLP_post_AST",approach=APPROACH)
ast_zero=generate_models({"spectrogram":AstFeatureZero},AstZero,"ast_zero",approach=APPROACH)

#creating the train/test dataset
loaders = esc_dataset.train_test_split().into_loaders(batch_size=BATCH_SIZE)

#run the training process for each featuress
for model in ast_zero:
    train(model=model, loaders=loaders)
