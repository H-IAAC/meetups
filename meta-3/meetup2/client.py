import argparse
import os

from numpy import genfromtxt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import flwr as fl

# Torne os logs do TensorFlow menos detalhados
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class HeadModel:
    def __init__(self, features, ACT_LABELS) -> None:
        self.n_features = len(features)
        self.n_labels = len(ACT_LABELS)
        self.base_model = self.build_base_model()

    def get_base_model(self) -> Sequential:
        return self.base_model

    def build_base_model(self) -> Sequential:
        model = Sequential()
        model.add(Dense(units=64, kernel_initializer='uniform', activation='relu', input_dim=self.n_features))
        model.add(Dense(units=128, kernel_initializer='uniform', activation='relu'))
        model.add(Dense(units=64, kernel_initializer='uniform', activation='relu'))
        model.add(Dense(units=self.n_labels, kernel_initializer='uniform', activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model


# Define o cliente Flower
class ModelClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Treina parametros no conjunto de treinamento mantido localmente."""

        # Atualiza os parametros do modelo local
        self.model.set_weights(parameters)

        # Hiperparametros para o round corrente
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        early_stopping = EarlyStopping(min_delta=0.001, patience=10, restore_best_weights=True)

        # Treine o modelo usando hiperparametros da configuracao
        history = self.model.fit(
            x=self.x_train,
            y=self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1,
            callbacks=[early_stopping]
        )

        # Retornar parametros e resultados do modelo atualizados
        parameters_prime = self.model.get_weights()
        num_examples_train = int(0.9 * len(self.x_train))

        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }

        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)

        # Hiperpatrametros para o round corrente
        batch_size: int = config["batch_size"]
        steps: int = config["eval_steps"]

        loss, accuracy = self.model.evaluate(x=self.x_test, y=self.y_test, batch_size=batch_size, steps=steps)
        num_examples_test = len(self.x_test)

        return loss, num_examples_test, {"accuracy": accuracy}

def main() -> None:
    # Parsing do argumento `partition`, da linha de comando
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--partition", type=int, choices=range(0, 10), required=True)
    args = parser.parse_args()

    features = ['attitude.roll', 'attitude.pitch', 'attitude.yaw', 'userAcceleration.x', 'userAcceleration.y',
                'userAcceleration.z', 'gravity.x', 'gravity.y', 'gravity.z', 'rotationRate.x', 'rotationRate.y',
                'rotationRate.z']

    ACT_LABELS = ["dws", "ups", "wlk", "jog", "std", "sit"]

    model_obj = HeadModel(features, ACT_LABELS)
    model = model_obj.get_base_model()

    # Carrega a particao dos dados locais
    (x_train, y_train), (x_test, y_test) = load_partition(args.partition)

    # Inicia o cliente Flower
    client = ModelClient(model, x_train, y_train, x_test, y_test)
    fl.client.start_numpy_client("[10.16.5.5]:8080", client=client)


def load_partition(idx: int):
    x_train = genfromtxt('./data/MotionSense_x_train.csv', delimiter=',')
    # filtrando a primeira linha (labels das colunas)
    x_train = x_train[1:, :]

    x_test = genfromtxt('./data/MotionSense_x_test.csv', delimiter=',')
    # filtrando a primeira linha (labels das colunas)
    x_test = x_test[1:, :]

    y_train = genfromtxt('./data/MotionSense_y_train.csv', delimiter=',')
    # filtrando a primeira linha (labels das colunas)
    y_train = y_train[1:, :]

    y_test = genfromtxt('./data/MotionSense_y_test.csv', delimiter=',')
    # filtrando a primeira linha (labels das colunas)
    y_test = y_test[1:, :]

    # Carregue 1/100 dos dados de treinamento e teste para simular uma particao.
    assert idx in range(10)

    # Calculates number of samples per partition
    n_samples_train_partition = int(x_train.shape[0] / 100)
    n_samples_test_partition = int(x_test.shape[0] / 100)

    return (
        x_train[idx * n_samples_train_partition: (idx + 1) * n_samples_train_partition, :],
        y_train[idx * n_samples_train_partition: (idx + 1) * n_samples_train_partition, :],
    ), (
        x_test[idx * n_samples_test_partition: (idx + 1) * n_samples_test_partition, :],
        y_test[idx * n_samples_test_partition: (idx + 1) * n_samples_test_partition, :],
    )


if __name__ == "__main__":
    main()
