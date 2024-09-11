import logging
import torch
from torch.optim.lr_scheduler import OneCycleLR

from classifiers.nn_utils import SequentialSaveableModel, SaveableModel
from datasets import TrainValidTestDataLoader
from features import Spectrogram, MelSpectrogram, AstEncoder


FEATURES = {
    "spectrogram": Spectrogram,
    "mel_spectrogram": MelSpectrogram,
    #"audio_spectrogram_transformers" : AstEncoder,
    # "cochleagram": Cochleagram
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def notify(*msg: str):
    message = " ".join(msg)
    print(message)

notify("Job started!")

def train(
    model: SaveableModel,
    loaders: TrainValidTestDataLoader,
    epochs: int = 100,
    learning_rate: float = 0.001,
    logging_file: str = "classifiers.log",
):
    """Train a neural network on the ESC-50 dataset.

    Parameters
    ----------
    model: torch.nn.Module
        the model to train on the ESC-50 dataset
    train_percentage: float
        the percentage data to use for training
    test_percentage: float
        the percentage data to use for testing
    learning_rate: float
        the learning rate to use for training
    """

    logging.basicConfig(filename=logging_file)

    # We use cross-entropy as it is well-known for performing well in classification problems
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = OneCycleLR(optimizer,max_lr=learning_rate, epochs = 100, steps_per_epoch= len(loaders.train), pct_start = 0.1)

    softmax   = torch.nn.Softmax()
    for epoch in range(epochs):
        notify(f"----------------------- EPOCH {epoch} -----------------------")

        train_loss = 0.0
        model.train()
        train_accuracy = 0.0
        for batch_num, data in enumerate(loaders.train):
            waveforms, labels = data
            waveforms, labels = waveforms.to(device), labels.to(device)

            # set optimizer params to zero
            optimizer.zero_grad()
            predictions         = model(waveforms)

            loss                = loss_func(predictions, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            class_prediction = softmax(predictions).argmax(dim=-1)
            train_accuracy += ((class_prediction == labels).sum())/class_prediction.shape[0]

            train_loss += loss.item()

        # get the mean values over all the batches
        train_accuracy  = train_accuracy / len(loaders.train)
        train_loss      = train_loss / len(loaders.train)

        notify(f"Train loss: {train_loss:.2f} Train accuracy :{train_accuracy*100:.2f}%")
        notify(f"Last value of learning rate for this epoch: {scheduler._last_lr}")


        # validation step
        model.eval()

        with torch.no_grad():
            
            val_loss        = 0
            val_accuracy    = 0
            for batch_num, data in enumerate(loaders.valid):
                waveforms, labels   = data
                waveforms, labels   = waveforms.to(device), labels.to(device)
                predictions         = model(waveforms)
                val_loss            += loss_func(predictions, labels)
                class_prediction    = softmax(predictions).argmax(dim=-1)
                val_accuracy        += ((class_prediction == labels).sum()) / class_prediction.shape[0]

            # get the mean values over the all the validation batches
            val_accuracy = val_accuracy /  len(loaders.valid)
            val_loss     = val_loss / len(loaders.valid)


        notify(f"Validation Loss : {val_loss:.2f}  Validation Accuracy: {val_accuracy*100:.2f}%")

            
    model.save(epoch=epoch)
    notify("Model saved")

    notify("----------------------FINISHED TRAINING----------------------")


def generate_models(
    feature_extractions: dict[str, torch.nn.Module],
    classifier: torch.nn.Module,
    classifier_name: str
) -> SequentialSaveableModel:
    """Generates models for all given feature extractions given a classifier

    Parameters
    ----------
    feature_extractions: list[str, torch.nn.Module]
        a list of feature extractions with their names
    classifier: torch.nn.Module
        the classifier to use
    classifier_name: the name of the classifier

    Returns
    list[SaveableModel]
        a list of saveable models, one for each feature extraction
    """

    models = []

    for name, feature_extraction in feature_extractions.items():
        classifier_kwarg = {}
        classifier_kwarg["input_size"] = feature_extraction.output_shape
        model = SequentialSaveableModel(
            (feature_extraction(), name), (classifier(**classifier_kwarg), classifier_name)
        )

        model.to(device)
        models.append(model)

    return models
