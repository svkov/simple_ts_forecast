import pmdarima as pmd
import pandas as pd

from simple_ts_forecast.models.model import Model
from simple_ts_forecast.utils import transform_date_start

import warnings
warnings.filterwarnings("ignore")


class ARIMA(Model):

    def __init__(self, df, n=14, column_name='price', exog_names=None, verbose=False, **kwargs):
        super().__init__(df, n=n, verbose=verbose)
        self.exog_names = exog_names
        self.column_name = column_name
        self.model = fit_model(df.dropna(), self.n, self.column_name, self.exog_names)

    def predict(self, df):
        return predict_model(self.model, df, self.n, self.exog_names)

    def predict_for_report(self, df, date_start, date_end):
        return arima_predict_for_report(df.dropna(), date_start, date_end, self.n, self.column_name, self.exog_names)


def fit_model(pivoted_df, n, column_name, exog_names):
    train = pivoted_df.iloc[:-n]
    # train_ex = train.drop([column_name], axis=1)
    if exog_names:
        train_ex = train[exog_names]
        model = pmd.auto_arima(train[column_name], train_ex)
    else:
        model = pmd.auto_arima(train[column_name])
    return model


def predict_model(model, pivoted_df, n, exog_names):
    if exog_names:
        train = pivoted_df.iloc[:-n]
        # last_val = train.drop([column_name], axis=1).values[-1].reshape(1, -1).tolist()
        last_val = train[exog_names].values[-1].reshape(1, -1).tolist()
        last_ex = last_val * n
        return model.predict(n, last_ex)
    else:
        return model.predict(n)


def update_model(model, pivoted_df, n, column_name):
    train = pivoted_df.iloc[:-n]
    new_price = train[column_name].values[-1]
    new_val = train.drop([column_name], axis=1).values[-1].reshape(1, -1).tolist()
    model.update(new_price, new_val)
    return model


def arima_predict_for_report(df, start_date, end_date, n, column_name, exog_names):
    pivoted_df = df[:start_date]

    model = fit_model(pivoted_df, n, column_name, exog_names)
    predictions = []

    for pivot in pd.date_range(start_date, end_date):
        pivoted_df = df[:pivot]
        model = update_model(model, pivoted_df, n, column_name)
        prediction = predict_model(model, pivoted_df, n, exog_names)
        predictions.append(prediction)

    columns = [f'{column_name} n{i + 1}' for i in range(n)]

    date_start = transform_date_start(start_date, n)
    date_end = transform_date_start(end_date, n)
    index = pd.date_range(date_start, date_end)

    return pd.DataFrame(predictions, columns=columns, index=index)
