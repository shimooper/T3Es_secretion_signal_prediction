import os

import numpy as np

import torch
from torch import nn
from sklearn import metrics


# A simple CNN model
class CNN(nn.Module):
    def __init__(self, number_of_classes, vocab_size, embedding_dim, input_len, device):
        super(CNN, self).__init__()

        self.device = device

        if number_of_classes == 2:
            self.is_multiclass = False
            number_of_output_neurons = 1
            loss = torch.nn.functional.binary_cross_entropy_with_logits
            output_activation = nn.Sigmoid()
        else:
            self.is_multiclass = True
            number_of_output_neurons = number_of_classes
            loss = torch.nn.CrossEntropyLoss()
            output_activation = lambda x: x

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.cnn_model = nn.Sequential(
            nn.Conv1d(in_channels=embedding_dim, out_channels=16, kernel_size=8, bias=True),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=8, bias=True),
            nn.BatchNorm1d(8),
            nn.MaxPool1d(2),

            nn.Conv1d(in_channels=8, out_channels=4, kernel_size=8, bias=True),
            nn.BatchNorm1d(4),
            nn.MaxPool1d(2),

            nn.Flatten()
        )
        self.dense_model = nn.Sequential(
            nn.Linear(self.count_flatten_size(input_len), 512),
            nn.Linear(512, number_of_output_neurons)
        )
        self.output_activation = output_activation
        self.loss = loss

    def count_flatten_size(self, input_len):
        zeros = torch.zeros([1, input_len], dtype=torch.long)
        x = self.embeddings(zeros)
        x = x.transpose(1, 2)
        x = self.cnn_model(x)
        return x.size()[1]

    def forward(self, x):
        x = self.embeddings(x)
        x = x.transpose(1, 2)
        x = self.cnn_model(x)
        x = self.dense_model(x)
        x = self.output_activation(x)
        return x
