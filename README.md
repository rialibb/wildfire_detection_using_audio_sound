# Wildfire_detection
# Etude de cas - Feux de forêt

## Introduction

This is a case study of the 2024-2025 Filière Recherche third year course at CentraleSupélec.
This project's duration was two weeks.
The goal of this project is to detect wildfires via audio data.

## Installation

Make sure you have install the [virtualenv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) package. 

To setup the repository, first create a virtual environment by running `python3 -m venv .env`.

Then, activate it by running `source .env/bin/activate` on Unix/macOS, or `.\.env\Scripts\activate` on Windows.

Finally, install the dependecies using `pip install -r requirements.txt`

## Run a training

The code to start training a model is in the file `main.py` . You uncomment the line with the model you want to train and then you run the file.
However, we recommend running this file on the HubIA's DGX, on the partitions 40 or 80.
The DGX's configuration file is in `DGX_runs/run_model_training.batch` . 
Type on the terminal : 
```shell
sbatch DGX_runs/run_model_training.batch
```
You can see the status of your job by typing:

```shell
squeue
```

To cancel a job, type:

```shell
scancel <num_job>
```


## Our work

We reused the code from the 2023-2024 team who had already worked on this case study.
However, we made some significant changes to their code. 
Here is the list of what we did:

### Bibliography

At the beginning of the project, we searched and read papers related to our topic.
We found two novel approaches that weren't indicated by the previous teams who worked on the project:
- [The audio spectrogram transformer (AST)](https://arxiv.org/pdf/2104.01778), a model that uses the attention mechanism to perform speech classification on ESC-50 and AudioSet dataset
- [Whisper A-T](https://arxiv.org/pdf/2307.03183), a paper that uses the encoder from the Speech2Text model Whisper as a backbone model to perform audio classification

We decided to focus on the first approach.

###  Features

We created new features  that can be found in the `features` folder: `ast_encoder.py` and `ast_feature_zero.py`.
- `ast_encoder.py` is the feature using the pretrained model Audio Spectrogram Transformer, which weights are downloaded from Hugging FAce's hub. 
  - This feature is to be used with the `MLPPostAst` classifier.
  - This is a feature, therefore the AST model itself is not trained. Its weights are freezed.
- `ast_feature_zero.py` is a feature that only performs resampling of the audio and computes the log-mel spectrogram.
  - This feature is to be used with the `AstZero` classifier. 
  - It is the feature for the model trained from scratch (i.e. not pretrained)
- `crnn_pretrained` is the feature that uses a pretrained model that has the same architecture as Zhang's CRNN. 
  - This feature is to be used with the `MlpPostCrnnPetrained` classifier.
  - The weights of the pretrained model are frozen.

### Classifiers

We added an `output_shape`class attribute to every feature. This is the shape of the output feature (without the batch dimension).


We created new audio classification models that can be found in the `classifiers` folder
- `ast_zero.py` : This is the code of the AST model that we train from scratch. We reuse the ASTModel class from the `transformers`library and we add an MLP on top.
- `mlp_post_ast.py`: This is the Multi-Layer-Perceptron (MLP) model that is put on top of the pretrained AST model.
- `mlp_post_crnn_pretrained.py`: This is the MLP model that is put on top of the pretrained CRNN model.- 

We also modified the code so that the structure of the models can be changed dynamically when the approach (binary classification, classification on ESC-10 or classification on ESC-50) changes.
This parameter that we added, `approach`, changes the the number of neurons of the last layer. This parameter is selected in the `main.py`file.
We also modified the default folder where models where trained. The previous team called it `~`and we changed it to `models_saved`.

### DGX_runs

This is a folder we created. It contains the file `run_model_training.batch` that is to be run when training on the DGX.

### models.py

We made significant modifications to this file, since the train function from the previous team didn't work:  models were not trained and accuracy wasn't computed the right way.

- We removed the `Cochleagram` feature from the list `FEATURES` since there was an error in the package `chcochleagram`
- In the `train` function:
  - we added an `eps_early_stopping` parameter to perform early-stopping. Training stops when the val accuracy hasn't increased by more than `eps_early_stopping` for five epochs in a row.
  - We added a learning rate scheduler `OneCycleLR`
  - We added a `model.train()` at the beginning of each epoch.
  - We completely changed how the accuracy was computed. It used to be computed only on the validation data and not on the train, and no softmax + argmax existed after the call to the model to get the prediction. The previous team used a confusion matrix that was therefore wrongly computed. We added four variables, `val_accuracy`, `val_loss`, `train_accuracy`, `train_loss`that we update at each batch.
  - We changed how the logs were shown, so that all the values for one epoch are put on one line. Value for different epochs are aligned so that the logs.log file can be easily processed via Excel.
  - We removed the computation and the printing of the confusion matrix at each epoch.

We changed the `generate_models` function so that it uses the `output_shape`class attribute of the features when that information is needed in the classifier.

### datasets.py

In this file, we also made many modifications in order to add our binary classification approach. 

- class `SplitableDatasetBin` : This  new class is added in order to perform the dinary dataset split into training, validation and testing sets. The index of the training observations is fixed and received as input in order to prevent information overlapping between the training and the val/test steps.
- class `ESCDatasetBin` : This is a subclass of SplitableDatasetBin and DownloadableDataset that will define the different methods needed for binary classification. 

### data_augmentation.py

Since our initial ESC50 dataset contains 50 classes with 40 observations each, converting this data to a binary classification will cause an imbalance between the fire and no fire classes. To solve this issue, we implemented a data augmentation approach.

- `data_augmentation.py` : This file contains the needed steps to perform data augmentation. it enale to create the datasets for training and val/test steps.
  - function `delete_audio` : this funciton will delete existing audio file in `audio_data/audio` folder. This step in necessary in order to track the indexes of training and val/test dataset.
  - function `extract_classes` : will extract the binary classes (fire : 1, no fire : 0) from ESC50 dataset
  - funciton `add_audio` : enables to add the different audio files from ESC50 in the folder `audio_data/audio` while maintaining track of each class.
  - function `add_background` : the data augmentation step is implemented in this function. The basic principe of this step is to take randomly a fire audio file as a main audio, and a non fire audio file as a background. Then, we decrease the frequency of the background file to avoid sounf domination. Finally, we merge both files in order to obtain as a result a new augmented audio file with non-fire background. Note that we perform if necessary a normalization step in order to prevent clipping.
  - function `generate_data` : we collect different defined functions here in order to implement the data augmentation. We start by training data augmentation for index tracking reason. Then we perform augmentation on val/test data. The final output in a dataset with equal proportions of fire and no-fire data.

