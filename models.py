import logging
import torch
from torch.optim.lr_scheduler import OneCycleLR

from classifiers.nn_utils import SequentialSaveableModel, SaveableModel
from datasets import TrainValidTestDataLoader
from features import Spectrogram, MelSpectrogram, AstEncoder


FEATURES = {
    "spectrogram": Spectrogram,
    "mel_spectrogram": MelSpectrogram,
    "audio_spectrogram_transformers" : AstEncoder,
    # "cochleagram": Cochleagram
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    for epoch in range(epochs):
        notify(f"----------------------- EPOCH {epoch} -----------------------")

        running_loss = 0.0
        model.train(True)
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
            class_prediction = torch.argmax(torch.nn.Softmax()(predictions))
            train_accuracy += (labels == class_prediction).sum().item() / labels.size(0)

            running_loss += loss.item()

        train_accuracy = train_accuracy / len(loaders.train)

        notify(f"Batch {batch_num}, Train loss: {running_loss} Train accuracy :{train_accuracy*100:.2f}%")

        confusion = torch.zeros((50,50), dtype=torch.int16)

        with torch.no_grad():

            for batch_num, data in enumerate(loaders.valid):
                waveforms, labels = data
                waveforms, labels = waveforms.to(device), labels.to(device)
                
                outputs = model(waveforms).tolist()
                val_loss = loss_func(outputs, labels)
                labels = labels.tolist()

                for pred, truth in zip(outputs, labels):
                    confusion[pred, truth] += 1

        true_preds = torch.sum(torch.diagonal(confusion))

        accuracy = true_preds/torch.sum(confusion)

        notify(f"Validation Accuracy: {accuracy*100:.2f}%")
        torch.set_printoptions(profile="full")
        torch.set_printoptions(profile="default")
                



        model.save(epoch=epoch)

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
