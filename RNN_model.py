import random
import time
import numpy as np
import pandas as pd
from sklearn import preprocessing
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard


# Define main variables
SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3
EPOCHS = 20
BATCH_SIZE = 64
NAME = f"{SEQ_LEN}-SEQ-BTC-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"
TRAIN_START = "2017-08-17"
TRAIN_END = "2020-11-25"
VALIDATION_START = "2020-11-26"
VALIDATION_END = "2021-12-31"


def classify(current, future):
    """
    Simple classifier
    :param current: number
    :param future: number
    :return: 1 or 0
    """
    if float(future) > float(current):
        return 1
    else:
        return 0


def preprocess_df(df):
    """
    Normalizing and creating sequences.
    Balancing sequence data.
    :param df: dataframe
    :return: tuple of arrays
    """
    df = df.drop("Future", 1)

    for col in df.columns:
        if col != "Target":
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)

    df.dropna(inplace=True)

    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)

    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])

    random.shuffle(sequential_data)

    buys, sells = [], []
    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])

    random.shuffle(buys)
    random.shuffle(sells)

    lower = min(len(buys), len(sells))

    buys = buys[:lower]
    sells = sells[:lower]

    sequential_data = buys + sells
    random.shuffle(sequential_data)

    x, y = [], []

    for seq, target in sequential_data:
        x.append(seq)
        y.append(target)

    return np.array(x).astype("float32"), np.array(y)


def create_main_and_validation_df(filepath):
    """
    Reading csv file and creating train data and validation data
    :param filepath: string
    :return: tuple of dataframes
    """
    df = pd.read_csv(filepath, parse_dates=["Date"], index_col="Date")
    df = df[["Close", "Volume"]]
    df["Future"] = df["Close"].shift(-FUTURE_PERIOD_PREDICT)
    df["Target"] = list(map(classify, df["Close"], df["Future"]))

    main_df = df.loc[TRAIN_START:TRAIN_END].copy()
    validation_df = df.loc[VALIDATION_START:VALIDATION_END].copy()

    return main_df, validation_df


def create_model(train_x, train_y, validation_x, validation_y, filepath="Final_RNN_Model"):
    """
    Creating and saving model
    """
    model = Sequential()
    model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())

    model.add(LSTM(128, input_shape=(train_x.shape[1:])))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(2, activation="softmax"))

    opt = tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-6)

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy"]
    )

    tensorboard = TensorBoard(log_dir=f"logs/{NAME}")

    model.fit(
        train_x, train_y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(validation_x, validation_y),
        callbacks=[tensorboard]
    )

    model.save(filepath)


def main():
    main_df, validation_df = create_main_and_validation_df("bitcoin.csv")

    train_x, train_y = preprocess_df(main_df)
    validation_x, validation_y = preprocess_df(validation_df)

    create_model(train_x, train_y, validation_x, validation_y, filepath="models/RNN-Final-seq60")


if __name__ == '__main__':
    main()
