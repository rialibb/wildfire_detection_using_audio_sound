import logging
import torch
from torch.optim.lr_scheduler import OneCycleLR
import datetime
from classifiers.nn_utils import SequentialSaveableModel, SaveableModel
from datasets import TrainValidTestDataLoader
from features import Spectrogram, MelSpectrogram, AstEncoder


# Feature to consider for crnn_zhang model
FEATURES = {
    "spectrogram": Spectrogram,
    "mel_spectrogram": MelSpectrogram,
}

# device where to run the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def notify(*msg: str):
    """ print logs in terminal

    Parameters
    ----------
    msg: str
        the message to print
    """
    message = " ".join(msg)
    print(message)

notify("Job started!")

def train(
    model: SaveableModel,
    loaders: TrainValidTestDataLoader,
    epochs: int = 100,
    learning_rate: float = 0.001,
    logging_file: str = "classifiers.log",
    eps_early_stopping: float =  1e-7,
):
    """Train a neural network on the ESC-50 dataset.

    Parameters
    ----------
    model: torch.nn.Module
        the model to train on the ESC-50 dataset
    epochs: int
        number of epchs for training
    learning_rate: float
        the learning rate to use for training
    logging_file: str
        path to the file to load logs 
    eps_early_stopping: float
        early stopping parameter
    """

    # retrieve current Date/time
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    notify(f"{date}")
    logging.basicConfig(filename=logging_file)

    # We use cross-entropy as it is well-known for performing well in classification problems
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = OneCycleLR(optimizer,max_lr=learning_rate, epochs = 100, steps_per_epoch= len(loaders.train), pct_start = 0.1)
    softmax   = torch.nn.Softmax()

    # initialize parameters
    previous_val_accuracy = 0.0
    count_occurence_no_change = 0
    notify(f"----------------------- Starting training -----------------------")
    for epoch in range(epochs):

        train_loss = 0.0
        model.train()
        train_accuracy = 0.0
        for batch_num, data in enumerate(loaders.train):
            # retrive data
            waveforms, labels = data
            # send data to device
            waveforms, labels = waveforms.to(device), labels.to(device)
            # set optimizer params to zero
            optimizer.zero_grad()
            # predict the output
            predictions         = model(waveforms)
            # calculate the loss
            loss                = loss_func(predictions, labels)
            # do backpropagation
            loss.backward()
            # update parameters
            optimizer.step()
            scheduler.step()
            class_prediction = softmax(predictions).argmax(dim=-1)
            train_accuracy += ((class_prediction == labels).sum())/class_prediction.shape[0]
            train_loss += loss.item()

        # get the mean values over all the batches
        train_accuracy  = train_accuracy / len(loaders.train)
        train_loss      = train_loss / len(loaders.train)

        # validation step
        model.eval()

        with torch.no_grad():
            # initialize parameters
            val_loss        = 0
            val_accuracy    = 0
            for batch_num, data in enumerate(loaders.valid):
                # retrieve data
                waveforms, labels   = data
                # send data to device
                waveforms, labels   = waveforms.to(device), labels.to(device)
                # predict output
                predictions         = model(waveforms)
                # calculate loss
                val_loss            += loss_func(predictions, labels)
                # find the classwith highest probability
                class_prediction    = softmax(predictions).argmax(dim=-1)
                # update accuracy
                val_accuracy        += ((class_prediction == labels).sum()) / class_prediction.shape[0]

            # get the mean values over the all the validation batches
            val_accuracy = val_accuracy /  len(loaders.valid)
            val_loss     = val_loss / len(loaders.valid)

        notify(f"Epoch {str(epoch).rjust(3)} Train loss: {train_loss:.6f} Train accuracy: {train_accuracy*100:.6f}% Validation Loss : {val_loss:.6f} Validation Accuracy: {val_accuracy*100:.6f}%  Last value of learning rate for this epoch: {scheduler.get_last_lr()}")
        # verify if the accuracy improvement is higher than early stopping parameter
        if abs(previous_val_accuracy - val_accuracy) < eps_early_stopping:
            count_occurence_no_change += 1
        else:
            count_occurence_no_change = 0
            previous_val_accuracy = val_accuracy
        if count_occurence_no_change >5 :
            notify("The model has not improved for 5 epochs in a row. Early stopping.")
            notify("Model saved")
            notify("----------------------FINISHED TRAINING----------------------")
            break
    # save the model
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
        # set the input size of the model
        classifier_kwarg["input_size"] = feature_extraction.output_shape
        # initialize the model
        model = SequentialSaveableModel(
            (feature_extraction(), name), (classifier(**classifier_kwarg), classifier_name)
        )
        # send model to device
        model.to(device)
        # append the modle in the list of models
        models.append(model)

    return models
