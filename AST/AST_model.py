import os
import librosa
import numpy as np
import torch
from transformers import ASTFeatureExtractor, ASTForAudioClassification
from datasets import DatasetDict, Dataset
from sklearn.metrics import accuracy_score


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}



def load_audio_files(path):
    audio_files = []
    labels = []
    for label, class_name in enumerate(['non_fire_audio', 'augmented_fire_audio']):
        class_path = os.path.join(path, class_name)
        for file_name in os.listdir(class_path):
            if file_name.endswith('.wav'):
                file_path = os.path.join(class_path, file_name)
                audio_files.append(file_path)
                labels.append(label)
    return audio_files, labels

def load_data(audio_files, labels):
    audio_data = []
    for file in audio_files:
        audio, sr = librosa.load(file, sr=None)
        audio_data.append(audio)
    return {'audio': audio_data, 'label': labels}, sr

path = 'audio_data'  
audio_files, labels = load_audio_files(path)
data, sr = load_data(audio_files, labels)

dataset = Dataset.from_dict(data)
dataset = dataset.train_test_split(test_size=0.3)









from transformers import ASTFeatureExtractor

feature_extractor = ASTFeatureExtractor()

def preprocess_function(examples):
    audio = examples['audio']
    return feature_extractor(audio, sampling_rate=sr, return_tensors='pt')

encoded_dataset = dataset.map(preprocess_function, remove_columns=['audio'])












from transformers import ASTForAudioClassification, Trainer, TrainingArguments

model = ASTForAudioClassification.from_pretrained('facebook/ast-finetuned-audio-classification')

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['test'],
    compute_metrics=compute_metrics 
)

trainer.train()
