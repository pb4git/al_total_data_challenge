import pandas as pd
from al_total_data_challenge.featureengineering import make_non_cyclic_categorical
from sklearn.metrics import mean_absolute_error
import catboost
import lightgbm as lgb
from itertools import repeat


def ts_split_predict(data, split_start_dates, split_end_dates, features, model_param):

    model = model_param["model"]
    data = data.copy().dropna(subset=features)
    data["farm_number_copy"] = data["farm_number"]
    data["horizon_copy"] = data["horizon"]

    # Loop on data similarly to TimeSeriesSplit
    for split_start_date, split_end_date in zip(split_start_dates, split_end_dates):
        data = make_non_cyclic_categorical(data, split_start_date)
        data = data.set_index(
            [
                "forecast_for",
                "forecast_from",
                "is_test",
                "horizon_copy",
                "farm_number_copy",
                "hh4",
            ]
        ).sort_index()

        # Train a model per horizon, or a model for all horizons ?
        for horizon in model_param["horizons"]:
            print("horizon", horizon)
            # Train a model per farm, or a model for all farms ?
            for farm_number in model_param["farms"]:
                print("farm", farm_number)
                # Train a model per hh4ofday, or a model for all hh4ofday ?
                for hh4ofday in model_param["hh4ofdays"]:
                    print("hh4ofday", hh4ofday)

                    X_train = data.loc[
                        (
                            slice(split_start_date - pd.to_timedelta("1h")),
                            slice(None),
                            False,
                            horizon,
                            farm_number,
                            hh4ofday,
                        ),
                        features,
                    ].copy()

                    y_train = data.loc[
                        (
                            slice(split_start_date - pd.to_timedelta("1h")),
                            slice(None),
                            False,
                            horizon,
                            farm_number,
                            hh4ofday,
                        ),
                        "wp",
                    ].copy()

                    X_predict = data.loc[
                        (
                            slice(split_start_date, split_end_date),
                            slice(None),
                            slice(None),
                            horizon,
                            farm_number,
                            hh4ofday,
                        ),
                        features,
                    ]

                    # Main call to fit()
                    model.fit(X_train, y_train)

                    # Main call to predict()
                    data.loc[
                        (
                            slice(split_start_date, split_end_date),
                            slice(None),
                            slice(None),
                            horizon,
                            farm_number,
                            hh4ofday,
                        ),
                        model_param["name"],
                    ] = model.predict(X_predict)

        data = data.reset_index()

    # Print local cross_validation score for this model
    data = data.set_index(
        ["forecast_for", "farm_number", "horizon", "forecast_from", "is_test"]
    ).sort_index()
    mask = ~data[[model_param["name"], "wp"]].isna().any(axis=1)
    mask &= data.index.get_level_values(0).year >= 2011
    print(
        mean_absolute_error(
            data[mask]["wp"].sort_index().groupby(level=[0, 1]).first(),
            data[mask][model_param["name"]].sort_index().groupby(level=[0, 1]).first(),
        )
    )

    # Extract and return prediction
    pred = data[model_param["name"]].copy()
    data = data.reset_index().drop(model_param["name"], axis=1)
    return pred


def ts_split_predict_all_farms(
    data,
    split_start_dates,
    split_end_dates,
    features_per_farm,
    features_common,
    model_param,
):

    model = model_param["model"]
    data = data.copy().dropna(subset=features_per_farm + features_common)
    data["horizon_copy"] = data["horizon"]

    # Empty placehold where prediction will be stored
    predictions = pd.DataFrame(
        index=data.pivot(
            index=["forecast_for", "forecast_from", "is_test", "horizon_copy", "hh4"],
            columns="farm_number",
        ).index
    )

    for farm_number in range(1, 7):
        print("farm", farm_number)
        # Construct feature list for this farm
        features = [
            (feature, fn)
            for fn in model_param["farms_to_use_as_features"][farm_number]
            for feature in features_per_farm
        ]

        for split_start_date, split_end_date in zip(split_start_dates, split_end_dates):
            data = make_non_cyclic_categorical(data, split_start_date)

            data_pivoted_per_farm = data.pivot(
                index=[
                    "forecast_for",
                    "forecast_from",
                    "is_test",
                    "horizon_copy",
                    "hh4",
                ],
                columns="farm_number",
            )

            X = pd.concat(
                [
                    data_pivoted_per_farm[features],
                    data_pivoted_per_farm[
                        list(zip(features_common, repeat(1)))
                    ].droplevel(1, axis=1),
                ],
                axis=1,
            )
            y = data_pivoted_per_farm["wp"]

            for horizon in model_param["horizons"]:
                print("horizon", horizon)
                for hh4ofday in model_param["hh4ofdays"]:
                    print("hh4ofday", hh4ofday)

                    X_train = X.loc[
                        (
                            slice(split_start_date - pd.to_timedelta("1h")),
                            slice(None),
                            False,
                            horizon,
                            hh4ofday,
                        ),
                        :,
                    ].copy()

                    y_train = y.loc[
                        (
                            slice(split_start_date - pd.to_timedelta("1h")),
                            slice(None),
                            False,
                            horizon,
                            hh4ofday,
                        ),
                        farm_number,
                    ].copy()

                    X_predict = X.loc[
                        (
                            slice(split_start_date, split_end_date),
                            slice(None),
                            slice(None),
                            horizon,
                            hh4ofday,
                        ),
                        :,
                    ].copy()

                    model.fit(X_train, y_train)

                    predictions.loc[
                        (
                            slice(split_start_date, split_end_date),
                            slice(None),
                            slice(None),
                            horizon,
                            hh4ofday,
                        ),
                        "pred" + str(farm_number),
                    ] = model.predict(X_predict)

    predictions = predictions[["pred" + str(i) for i in range(1, 7)]]
    predictions.columns = pd.MultiIndex.from_product(
        [[model_param["name"]], list(range(1, 7))], names=(None, "farm_number")
    )
    predictions = predictions.stack().sort_index()

    data = data.set_index(
        [
            "forecast_for",
            "forecast_from",
            "is_test",
            "horizon_copy",
            "hh4",
            "farm_number",
        ]
    ).sort_index()
    data[model_param["name"]] = predictions

    # Print local cross_validation score for this model
    mask = ~data[[model_param["name"], "wp"]].isna().any(axis=1)
    mask &= data.index.get_level_values(0).year >= 2011
    print(
        mean_absolute_error(
            data[mask]["wp"].sort_index().groupby(level=[0, 1]).first(),
            data[mask][model_param["name"]].sort_index().groupby(level=[0, 1]).first(),
        )
    )

    pred = data[model_param["name"]].copy()
    data = data.reset_index().drop(model_param["name"], axis=1)

    return (
        pred.reset_index()
        .rename(columns={"horizon_copy": "horizon"})
        .set_index(
            ["forecast_for", "farm_number", "horizon", "forecast_from", "is_test"]
        )
        .sort_index()[model_param["name"]]
    )