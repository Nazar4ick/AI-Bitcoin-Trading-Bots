import pandas as pd
import tensorflow as tf
import numpy as np
from RNN_model import classify


# Define main variables
FUTURE_PERIOD_PREDICT = 3
DF_START = "2020-09-25"
TEST_START = "2020-11-26"
TEST_END = "2021-12-31"


def read_df(filepath):
    """
    Creating dataframe from csv file
    :param filepath: string
    :return: dataframe
    """
    df = pd.read_csv(filepath, parse_dates=["Date"], index_col="Date")
    df = df[["Close", "Volume"]]
    df["Future"] = df["Close"].shift(-FUTURE_PERIOD_PREDICT)
    df["Target"] = list(map(classify, df["Close"], df["Future"]))

    test_df = df.loc[DF_START:TEST_END].copy()

    return test_df


def main():
    df = read_df("bitcoin.csv")
    model = tf.keras.models.load_model("models/RNN-Final-seq60")
    positions = [np.nan for _ in range(60)]

    x = []

    for i in range(len(df[["Close", "Volume"]])):
        if i > 59:
            x.append(df[["Close", "Volume"]][i-60:i].to_numpy())

    x = np.array(x)

    prediction = model.predict(x)

    for val in prediction:
        val = list(val)
        if val[0] > val[1]:
            positions.append(1)
        else:
            positions.append(-1)

    df["Position"] = positions
    df.dropna(inplace=True)

    df["returns"] = np.log(df.Close / df.Close.shift(1))
    df["strategy"] = df["Position"].shift(1) * df["returns"]
    df["trades"] = df.Position.diff().fillna(0).abs()
    df.strategy = df.strategy + df.trades * (-0.00075)

    multiple = np.exp(df["strategy"].sum())

    print(f"Multiple coefficient for RNN model: {multiple}")
    print(f"Buy&Hold multiple:                  {np.exp(df.returns.sum())}")


if __name__ == '__main__':
    main()
