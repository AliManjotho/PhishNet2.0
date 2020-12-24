import tensorflow as tf
import pandas as pd
from model import PhishingModel


def main():
    dataset = pd.read_csv('dataset/dataset_B_05_2020.csv')
    dataset = dataset.drop(dataset.columns[list(range(81, 88))], axis=1)
    tf.keras.Sequential
    print(dataset.shape)

    #x = []
    #y = []
    #model = PhishingModel()

    #model.compile(loss='mse', optimizer='adam', metrics='accuracy')
    #model.fit(x, y, batch_size=32, epochs=10)
    #print(model.summary())


if __name__ == '__main__':
    main()
