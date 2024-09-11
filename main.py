from classifiers.cnn_bardou import CNNBardou
from classifiers.crnn_zhang import ConvolutionalRNNZhang
from datasets import ESCDataset, ESCDatasetBin
from features import Spectrogram, MelSpectrogram, Cochleagram
from models import train, FEATURES, generate_models

BATCH_SIZE = 32

MODEL_KWARGS = {"input_size": (256, 2206)} #size of a spectrogram/mel-spectrogram

APPROACH = 2  # change the approach based on 2, 10 or 50 classes

esc_dataset = ESCDataset(download=False) if APPROACH!=2 else ESCDatasetBin(download=False)

bardou_models = generate_models({"spectrogram": Spectrogram}, CNNBardou, "cnn_bardou")
#zhang_models = generate_models(FEATURES, ConvolutionalRNNZhang, "crnn_zhang", classifier_kwargs=MODEL_KWARGS)

loaders = esc_dataset.train_test_split().into_loaders(batch_size=BATCH_SIZE)

for model in bardou_models:
    train(model=model, loaders=loaders)
