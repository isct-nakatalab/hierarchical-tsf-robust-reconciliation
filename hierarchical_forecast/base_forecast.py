import lightgbm as lgb
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb


def base_prophet(df, num_train):
    print("== START BASE_FORECAST ==")

    df_base = []
    for _, df_i in df.items():
        # prepare data
        df_i = df_i.reset_index()
        df_i.columns = ["ds", "y"]
        df_i_train = df_i.iloc[:num_train]

        # forecast
        model = Prophet()
        model.fit(df_i_train)
        df_i_base = model.predict(df_i)
        df_base += [df_i_base.yhat]

    # convert to DataFrame
    df_base = pd.concat(df_base, axis=1)
    df_base.index = df.index
    df_base.columns = df.columns

    print("== END BASE_FORECAST ==")

    return df_base


def base_lgb(df: pd.DataFrame, num_train: int):
    print("==START BASE_FORECAST ==")

    df = df.copy()
    df_base = []
    index = df.index
    target_columns = df.columns
    df.reset_index(names="ds", inplace=True)
    df["year"] = df["ds"].dt.year
    df["month"] = df["ds"].dt.month
    df["weekday"] = df["ds"].dt.weekday + 1
    for column in target_columns:
        # prepare data
        df_i = df[[column, "year", "month", "weekday"]]
        df_i["prev"] = df_i[column].shift(1)
        df_i_train = df_i.iloc[:num_train].copy()
        df_i_test = df_i.iloc[num_train:].copy()
        features = ["year", "month", "weekday", "prev"]
        X_train = df_i_train[features]
        y_train = df_i_train[column]
        train_dataset = lgb.Dataset(X_train, label=y_train)

        # forecast
        params = {
            "objective": "regression",
            "metric": "mae",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "verbose": -1,
        }
        model = lgb.train(params, train_dataset, num_boost_round=1000)
        df_i_train["yhat"] = model.predict(X_train)
        for i in range(len(df_i_test)):
            if i == 0:
                df_i_test.loc[df_i_test.index[i], "prev"] = df_i_train.iloc[-1][column]
            else:
                df_i_test.loc[df_i_test.index[i], "prev"] = df_i_test.loc[
                    df_i_test.index[i - 1], "yhat"
                ]
            X_test = (
                df_i_test.loc[df_i_test.index[i], features].to_numpy().reshape(1, -1)
            )
            df_i_test.loc[df_i_test.index[i], "yhat"] = model.predict(X_test)[0]
        pred_train = df_i_train["yhat"]
        pred_test = df_i_test["yhat"]
        df_i_base = pd.concat([pred_train, pred_test])
        df_base.append(df_i_base)

    # convert to DataFrame
    df_base = pd.concat(df_base, axis=1)
    df_base.index = index
    df_base.columns = target_columns

    print("== END BASE_FORECAST ==")

    return df_base


def base_xgb(df: pd.DataFrame, num_train: int):
    print("==START BASE_FORECAST ==")

    df = df.copy()
    df_base = []
    index = df.index
    target_columns = df.columns
    df.reset_index(names="ds", inplace=True)
    df["year"] = df["ds"].dt.year
    df["month"] = df["ds"].dt.month
    df["weekday"] = df["ds"].dt.weekday + 1
    for column in target_columns:
        # prepare data
        df_i = df[[column, "year", "month", "weekday"]]
        df_i["prev"] = df_i[column].shift(1)
        df_i_train = df_i.iloc[:num_train].copy()
        df_i_test = df_i.iloc[num_train:].copy()
        features = ["year", "month", "weekday", "prev"]
        X_train = df_i_train[features]
        y_train = df_i_train[column]
        train_dataset = xgb.DMatrix(X_train, label=y_train, feature_names=features)

        # forecast
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "learning_rate": 0.05,
            "max_depth": 6,
            "colsample_bytree": 0.8,
            "subsample": 0.8,
        }
        model = xgb.train(params, train_dataset, num_boost_round=1000)
        df_i_train["yhat"] = model.predict(xgb.DMatrix(X_train, feature_names=features))
        for i in range(len(df_i_test)):
            if i == 0:
                df_i_test.loc[df_i_test.index[i], "prev"] = df_i_train.iloc[-1][column]
            else:
                df_i_test.loc[df_i_test.index[i], "prev"] = df_i_test.loc[
                    df_i_test.index[i - 1], "yhat"
                ]
            X_test = xgb.DMatrix(
                df_i_test.loc[df_i_test.index[i], features].to_numpy().reshape(1, -1),
                feature_names=features,
            )
            df_i_test.loc[df_i_test.index[i], "yhat"] = model.predict(X_test)[0]
        pred_train = df_i_train["yhat"]
        pred_test = df_i_test["yhat"]
        df_i_base = pd.concat([pred_train, pred_test])
        df_base.append(df_i_base)

    # convert to DataFrame
    df_base = pd.concat(df_base, axis=1)
    df_base.index = index
    df_base.columns = target_columns

    print("== END BASE_FORECAST ==")

    return df_base


def base_arima(df: pd.DataFrame, num_train: int):
    print("== START BASE_FORECAST ==")

    df_base = []
    for _, df_i in df.items():
        # prepare data
        df_i_train = df_i.iloc[:num_train]
        df_i_test = df_i.iloc[num_train:]

        # forecast
        model = ARIMA(df_i_train, order=(1, 1, 1))
        model_fit = model.fit()
        pred_train = model_fit.predict(start=0, end=len(df_i_train) - 1)
        pred_test = model_fit.forecast(steps=len(df_i_test))
        df_i_base = pd.concat([pred_train, pred_test])
        df_base += [df_i_base]

    # convert to DataFrame
    df_base = pd.concat(df_base, axis=1)
    df_base.index = df.index
    df_base.columns = df.columns

    print("== END BASE_FORECAST ==")

    return df_base
