from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from shutil import unpack_archive
import os
import os.path


from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import urllib.request
import torch
import torchaudio


class ESC(Enum):
    TEN = 10
    FIFTY = 50
    TWO =2


@dataclass
class TrainValidTestDataLoader:
    train: DataLoader
    valid: DataLoader
    test: DataLoader


@dataclass
class TrainValidTestDataset:
    train: Dataset
    valid: Dataset
    test: Dataset

    def into_loaders(self, batch_size: int = 32) -> TrainValidTestDataLoader:
        """Turn the datasets into DataLoaders.

        Parameters
        ----------
        batch_size: int
            the size of the batches in the dataset
        """
        train_loader = DataLoader(self.train, batch_size, shuffle=True)
        valid_loader = DataLoader(self.valid, batch_size, shuffle=True)
        test_loader = DataLoader(self.test, batch_size, shuffle=True)

        return TrainValidTestDataLoader(
            train=train_loader, valid=valid_loader, test=test_loader
        )


class SplitableDataset(ABC, Dataset):
    def __init__(
        self, train_percentage: float = 0.7, test_percentage: float = 0.15
    ) -> None:
        super().__init__()
        self.train_percentage = train_percentage
        self.test_percentage = test_percentage

    def train_test_split(self) -> TrainValidTestDataset:
        """Split the dataset into train and test datasets

        Returns
        -------
        TrainValidTestDataset
            an object holding the train dataset and the test (validation) dataset
        """
        train_size = int(self.train_percentage * len(self))
        test_size = int(self.test_percentage * len(self))
        valid_size = len(self) - train_size - test_size

        train_dataset, valid_dataset, test_dataset = random_split(
            self, [train_size, valid_size, test_size]
        )

        return TrainValidTestDataset(
            train=train_dataset, valid=valid_dataset, test=test_dataset
        )






class SplitableDatasetBin(ABC, Dataset):
    def __init__(
        self, train_percentage: float = 0.7, test_percentage: float = 0.15, train_index : list=list(range(1960))
    ) -> None:
        super().__init__()
        self.train_percentage = train_percentage
        self.test_percentage = test_percentage
        self.train_index = train_index

    def train_test_split(self) -> TrainValidTestDataset:
        """Split the dataset into train and test datasets

        Returns
        -------
        TrainValidTestDataset
            an object holding the train dataset and the test (validation) dataset
        """
        train_size = int(self.train_percentage * len(self))
        test_size = int(self.test_percentage * len(self))
        valid_size = len(self) - train_size - test_size

        train_dataset,  = self[self.csv.index.isin(self.train_index)].sample(train_size)

        valid_dataset, test_dataset = random_split(
            self[~ self.csv.index.isin(self.train_index)], [valid_size, test_size]
        )


        return TrainValidTestDataset(
            train=train_dataset, valid=valid_dataset, test=test_dataset
        )




class DownloadableDataset(ABC, Dataset):
    def __init__(self, path: str, download: bool = False):
        self.path = path

        if download and not os.path.exists(path):
            self.download()

    @abstractmethod
    def download(self):
        raise NotImplementedError

    def _make_dirs(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)


class ESCDataset(DownloadableDataset, SplitableDataset):

    def __init__(
        self,
        path: str = "data/esc50",
        download: bool = False,
        categories: ESC = ESC.FIFTY,
        train_percentage: float = 0.7,
        test_percentage: float = 0.15,
        data_size:int=100
    ) -> None:
        
        """
        Args:
            path: the path to where the dataset is or should be stored
            download: whether to download the data
            categories: whether to use ESC-10 or ESC-50
        """
        DownloadableDataset.__init__(self=self, path=path, download=download)
        SplitableDataset.__init__(
            self=self,
            train_percentage=train_percentage,
            test_percentage=test_percentage,
        )

        self.csv = pd.read_csv(os.path.join(path, "meta/esc50.csv"))

        self.csv=self.csv.sample(frac=1)
        self.csv=self.csv.iloc[:data_size]
        self.categories = categories

    def download(self):
        """Automatically downloads and extracts the dataset in the desired data directory"""
        self._make_dirs()

        url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
        ZIP_FILE_NAME = "temp-esc50.zip"

        urllib.request.urlretrieve(url, ZIP_FILE_NAME)
        unpack_archive(ZIP_FILE_NAME, os.path.join(self.path, ".."))

        os.rename(os.path.join(self.path, "..", "ESC-50-master"), self.path)
        os.remove(ZIP_FILE_NAME)

    def __len__(self) -> int:
        """Computes the size of the dataset.

        Returns
        -------
        int
            the size of the dataset
        """
        if self.categories == ESC.TEN:
            return len(self.csv[self.csv.esc10 == True])
        else:
            return len(self.csv)

    def _get_wav_file_path(self, index: int) -> str:
        """Returns the path to the wav file corresponding to sample at given index in the csv.

        Parameters
        ----------
        index: int
            the index of the item in the csv annotations filemkdir

        Returns
        -------
        string
            the path to the wav file
        """
        return os.path.join(self.path, "audio", self.csv.iloc[index, 0])

    def _get_sample_label(self, index: int) -> str:
        return self.csv.iloc[index, 2]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Gets the dataset item at given index

        Parameters
        ----------
        index: int
            the index number where to look for the item

        Returns
        -------
        int
            a tuple that contains the waveform and the corrsponding label at given index
        """
        wav_path = self._get_wav_file_path(index)
        label = self._get_sample_label(index)
        sample, sample_rate = torchaudio.load(wav_path)
        assert sample_rate == 44100

        return sample, label


    def get_all_labels(self) -> list[torch.Tensor]:
        """Returns all possible labels in this dataset

        Returns
        -------
        list[torch.Tensor]
            a list of all possible labels
        """
        return [x for x in self.csv["target"].unique()]
    

class ESCDatasetBin(DownloadableDataset, SplitableDatasetBin):
    def __init__(
        self,
        path: str = "audio_data",
        download: bool = False,
        categories: ESC = ESC.TWO,
        train_percentage: float = 0.7,
        test_percentage: float = 0.15,
        train_index : list=list(range(1960)),
        data_size: int=100
    ) -> None:
        """
        Args:
            path: the path to where the dataset is or should be stored
            download: whether to download the data
            categories: whether to use ESC-10 or ESC-50 or ESC-2
        """
        DownloadableDataset.__init__(self=self, path=path, download=download)
        SplitableDatasetBin.__init__(
            self=self,
            train_percentage=train_percentage,
            test_percentage=test_percentage,
            train_index = train_index
        )

        self.csv = pd.read_csv("audio_data/meta/esc2.csv")
        self.categories = categories
        

    def __len__(self) -> int:
        """Computes the size of the dataset.

        Returns
        -------
        int
            the size of the dataset
        """
        
        return len(self.csv[self.csv.index.isin(self.train_index)])

    def _get_wav_file_path(self, index: int) -> str:
        """Returns the path to the wav file corresponding to sample at given index in the csv.

        Parameters
        ----------
        index: int
            the index of the item in the csv annotations filemkdir

        Returns
        -------
        string
            the path to the wav file
        """
        return os.path.join(self.path, "audio", self.csv.iloc[index, 0])

    def download(self):
        """Method needed to instantiate the Dataset without error
        """
        raise NotImplementedError

    def get_all_labels(self) -> list[torch.Tensor]:
        """Returns all possible labels in this dataset

        Returns
        -------
        list[torch.Tensor]
            a list of all possible labels
        """
        return [x for x in self.csv["target"].unique()]
    
    
    def _get_sample_label(self, index: int) -> str:
        return self.csv.iloc[index, 1]
    

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Gets the dataset item at given index

        Parameters
        ----------
        index: int
            the index number where to look for the item

        Returns
        -------
        int
            a tuple that contains the waveform and the corrsponding label at given index
        """
        wav_path = self._get_wav_file_path(index)
        label = self._get_sample_label(index)
        sample, sample_rate = torchaudio.load(wav_path)
        assert sample_rate == 44100

        return sample, label



class ESCDatasetBinNoOverlap(DownloadableDataset, SplitableDataset):
    def __init__(
        self,
        path: str = "data/esc50",
        download: bool = False,
        categories: ESC = ESC.TWO,
        train_percentage: float = 0.7,
        test_percentage: float = 0.15,
        data_size: int=100
    ) -> None:
        """
        Args:
            path: the path to where the dataset is or should be stored
            download: whether to download the data
            categories: whether to use ESC-10 or ESC-50 or ESC-2
        """
        DownloadableDataset.__init__(self=self, path=path, download=download)
        SplitableDataset.__init__(
            self=self,
            train_percentage=train_percentage,
            test_percentage=test_percentage,
        )

        self.csv = pd.read_csv(os.path.join(path, "meta/esc50.csv"))
        self.csv=self.csv.sample(frac=1)

        self.csv_fire=self.csv[self.csv["target"]==12]
        self.csv_no_fire=self.csv[self.csv["target"]!=12].iloc[:data_size]

        self.csv=pd.concat([self.csv_fire,self.csv_no_fire],axis=0)
        self.csv=self.csv.sample(frac=1)

        
        self.csv.loc[self.csv["target"]!=12,["target"]]=0
        self.csv.loc[self.csv["target"]==12,["target"]]=1

        self.categories = categories
        
        

    def __len__(self) -> int:
        """Computes the size of the dataset.

        Returns
        -------
        int
            the size of the dataset
        """
        
        return len(self.csv)

    def _get_wav_file_path(self, index: int) -> str:
        """Returns the path to the wav file corresponding to sample at given index in the csv.

        Parameters
        ----------
        index: int
            the index of the item in the csv annotations filemkdir

        Returns
        -------
        string
            the path to the wav file
        """
        return os.path.join(self.path, "audio", self.csv.iloc[index, 0])

    def download(self):
        """Method needed to instantiate the Dataset without error
        """
        raise NotImplementedError

    def get_all_labels(self) -> list[torch.Tensor]:
        """Returns all possible labels in this dataset

        Returns
        -------
        list[torch.Tensor]
            a list of all possible labels
        """
        return [x for x in self.csv["target"].unique()]
    
    
    def _get_sample_label(self, index: int) -> str:
        return self.csv.iloc[index, 2]
    

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Gets the dataset item at given index

        Parameters
        ----------
        index: int
            the index number where to look for the item

        Returns
        -------
        int
            a tuple that contains the waveform and the corrsponding label at given index
        """
        wav_path = self._get_wav_file_path(index)
        label = self._get_sample_label(index)
        sample, sample_rate = torchaudio.load(wav_path)
        assert sample_rate == 44100

        return sample, label
