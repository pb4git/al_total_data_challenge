import datetime
import numpy as np
import seaborn as sns
import pandas as pd
from itertools import repeat
from sklearn.linear_model import Ridge
import numpy as np
import catboost
import al_total_data_challenge.readrawdata
import al_total_data_challenge.split

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


_, test_dates = al_total_data_challenge.readrawdata.readrawdata(
    r"C:\Users\paul.berhaut\projects\al_total_data_challenge\data\challenge_19_data\\"
)

data = pd.read_parquet(
    "..\data.1.1.0.0.parquet.gzip", engine="fastparquet"
)  # fastparquet required to preserve categorical data types


(
    split_start_dates,
    split_end_dates,
) = al_total_data_challenge.split.get_timeseries_split_boundaries(data, 52)

split_start_dates.insert(0, pd.to_datetime("2010-03-01 01:00:00"))
split_end_dates.insert(0, pd.to_datetime("2011-01-01 00:00:00"))

data["is_test"] = data["wp"].isna()
data = data.set_index(["forecast_for", "forecast_from", "is_test"]).sort_index()
data = data.reset_index()

##################################################################################################

data = data.set_index(["forecast_for", "farm_number", "horizon", "forecast_from"])

model_names = [
    "a",
    # 'b',
    # 'c',
    # 'd',
    # 'a.all',
    "c.all",
    # 'a.some.one',
    "a.some.three",
    # 'a.quant',
    # 'b.quant',
    # 'c.quant',
    # 'd.quant',
    # 'a.power',
    # 'b.power',
    # 'c.power',
    # 'd.power',
    # 'a.otherfeatures',
    # 'c.otherfeatures',
    # 'd.otherfeatures',
    "a.all.hordelta",
    # 'a.lgbm',
    # 'c.lgbm',
    # 'd.lgbm',
    # 'a.all.lgbm',
    "c.all.lgbm",
]
model_names_ae = [model_name + "_ae" for model_name in model_names]

for model_name in model_names:
    print("read", model_name)
    data[model_name] = pd.read_parquet(
        r"..\model_predictions\\" + model_name + ".parquet.gzip", engine="fastparquet"
    )
    data[model_name + "_ae"] = (data[model_name] - data["wp"]).abs()

data = data.reset_index()

##################################################################################################
# For each forecast_for, there can be 1, 2, 3 or 4 different forecasts (short, mid or long term)
# Instead of using the latest forecast, can we do better and use all forecasts available ?


class OneRidge(Ridge):
    def fit(self, X, y):
        super().fit(X, y)
        self.coef_ /= np.sum(self.coef_)
        return self


for base_model_name in model_names:
    print("Horiz", base_model_name)

    data = data.set_index(["forecast_for", "farm_number", "horizon"]).sort_index()
    predictions_by_horizon = data[[base_model_name, "wp"]].unstack(level=2)
    predictions_placeholder = pd.DataFrame(index=predictions_by_horizon.index)

    for n_predictions in range(4, 0, -1):

        predictions_available = list(range(4 - n_predictions, 4))

        mask_predictions_exist = ~(
            predictions_by_horizon[zip(repeat(base_model_name), predictions_available)]
            .isna()
            .any(axis=1)
        )
        mask_wp_exists = ~(predictions_by_horizon["wp"].isna().any(axis=1))

        mask_predictions_and_wp_both_exist = mask_predictions_exist & mask_wp_exists
        mask_predictions_exist_but_not_wp = mask_predictions_exist & ~mask_wp_exists
        mask_date_no_leak = (
            predictions_by_horizon.index.get_level_values(0).year <= 2010
        )

        lr = Ridge(fit_intercept=False)
        lr.fit(
            predictions_by_horizon[
                mask_predictions_and_wp_both_exist & mask_date_no_leak
            ][zip(repeat(base_model_name), predictions_available)],
            predictions_by_horizon[
                mask_predictions_and_wp_both_exist & mask_date_no_leak
            ][("wp", 3)],
        )

        predictions_placeholder.loc[
            mask_predictions_exist, 4 - n_predictions
        ] = lr.predict(
            predictions_by_horizon[mask_predictions_exist][
                zip(repeat(base_model_name), predictions_available)
            ]
        )

    predictions_placeholder.columns = pd.MultiIndex.from_product(
        [["lrhoriz_" + base_model_name], list(range(0, 4))], names=(None, "horizon")
    )
    predictions_placeholder["lrhoriz_" + base_model_name] = predictions_placeholder[
        "lrhoriz_" + base_model_name
    ].fillna(value=predictions_by_horizon[base_model_name])
    data["lrhoriz_" + base_model_name] = predictions_placeholder.stack()

    data = data.reset_index()

##################################################################################################
# Maybe a certain model is better for farm 0, and another model is better for farm 1 ?
# Maybe a certain model is better at short horizon, and another is better at long horizon ?
# Train a linear model with such categorical features and combine all models
# Note : in the latest code, all this is disabled : the feature "category_to_split_by" contains only zeroes as it worked better in practice....

data["category_to_split_by"] = pd.Categorical(
    data["farm_number"].astype(int) * 0 + data["horizon"].astype(int) * 0
)

predictions_cross_category = pd.concat(
    [
        pd.get_dummies(data["category_to_split_by"]).mul(data[model_name], axis=0)
        for model_name in model_names
    ],
    axis=1,
)
predictions_cross_category = pd.concat([predictions_cross_category, data["wp"]], axis=1)
predictions_cross_category.columns = list(
    range(len(predictions_cross_category.columns) - 1)
) + ["wp"]

maskfit = (~predictions_cross_category.isna().any(axis=1)) & (
    data["forecast_for"].dt.year <= 2010
)
maskuse = ~(predictions_cross_category.drop("wp", axis=1).isna().any(axis=1))

cb = catboost.CatBoostRegressor(loss_function="MAE")

cb.fit(
    predictions_cross_category[maskfit].drop("wp", axis=1),
    predictions_cross_category[maskfit]["wp"],
)
data.loc[maskuse, "catboost"] = cb.predict(
    predictions_cross_category[maskuse].drop("wp", axis=1)
)

##################################################################################################
# Same as above, but for the lrhoriz base predictions

data["category_to_split_by"] = pd.Categorical(
    data["farm_number"].astype(int) * 0 + data["horizon"].astype(int) * 0
)

predictions_cross_category = pd.concat(
    [
        pd.get_dummies(data["category_to_split_by"]).mul(
            data["lrhoriz_" + model_name], axis=0
        )
        for model_name in model_names
    ],
    axis=1,
)
predictions_cross_category = pd.concat([predictions_cross_category, data["wp"]], axis=1)
predictions_cross_category.columns = list(
    range(len(predictions_cross_category.columns) - 1)
) + ["wp"]

maskfit = (~predictions_cross_category.isna().any(axis=1)) & (
    data["forecast_for"].dt.year <= 2010
)
maskuse = ~(predictions_cross_category.drop("wp", axis=1).isna().any(axis=1))

cb = catboost.CatBoostRegressor(loss_function="MAE")

cb.fit(
    predictions_cross_category[maskfit].drop("wp", axis=1),
    predictions_cross_category[maskfit]["wp"],
)
data.loc[maskuse, "lrhoriz_catboost"] = cb.predict(
    predictions_cross_category[maskuse].drop("wp", axis=1)
)


##################################################################################################
# Add all the intermediate models calculated above to the list of models we have

to_add_to_model_names_ae = []
to_add_to_model_names = []

data["mean"] = data[model_names].mean(axis=1)
data["mean_ae"] = (data["mean"] - data["wp"]).abs()

to_add_to_model_names.append("mean")
to_add_to_model_names_ae.append("mean_ae")

for base_model_name in model_names:
    data["lrhoriz_" + base_model_name + "_ae"] = (
        data["lrhoriz_" + base_model_name] - data["wp"]
    ).abs()
    to_add_to_model_names_ae.append("lrhoriz_" + base_model_name + "_ae")
    to_add_to_model_names.append("lrhoriz_" + base_model_name)

data["catboost_ae"] = (data["catboost"] - data["wp"]).abs()
to_add_to_model_names.append("catboost")
to_add_to_model_names_ae.append("catboost_ae")

data["lrhoriz_catboost_ae"] = (data["lrhoriz_catboost"] - data["wp"]).abs()
to_add_to_model_names.append("lrhoriz_catboost")
to_add_to_model_names_ae.append("lrhoriz_catboost_ae")


model_names += to_add_to_model_names
model_names_ae += to_add_to_model_names_ae

##################################################################################################

#%%
data["horizon_copy"] = data["horizon"]


all_predictions = (
    data[
        ["forecast_for", "farm_number", "horizon", "horizon_copy", "forecast_from"]
        + model_names
        + model_names_ae
    ]
    .set_index(["forecast_for", "farm_number", "horizon_copy"])
    .sort_index()
    .groupby(level=[0, 1])
    .first()
    .loc[
        (
            slice(
                pd.to_datetime("2011-01-01 00:00:00"),
                pd.to_datetime("2015-01-01 00:00:00"),
            ),
            slice(None),
        ),
        :,
    ]
    .copy()
)
all_predictions = all_predictions.set_index(["horizon", "forecast_from"], append=True)
print(all_predictions[model_names_ae].groupby(level=1).mean())
print(all_predictions[model_names_ae].groupby(level=1).mean().mean())
sns.heatmap(all_predictions[model_names_ae].groupby(level=1).mean())

#################################################################################
# Write the submit file for the final model
#

final_model_chosen_for_submit = "catboost"

predictions = test_dates.copy()
predictions.index = predictions.index.rename("forecast_for")
predictions = predictions.join(
    other=all_predictions[final_model_chosen_for_submit]
    .reset_index()
    .set_index(["forecast_for", "farm_number"])[final_model_chosen_for_submit]
    .unstack()
)
predictions.index = predictions.index.rename("date")
predictions.index = predictions.index.strftime(date_format="%Y%m%d%H")
predictions = (
    predictions.stack()
    .reset_index(level=1)
    .rename(columns={"level_1": "farm_number", 0: "wppred"})
    .set_index("farm_number", append=True)
)
predictions = predictions.unstack()
now = datetime.datetime.now()
tag = now.strftime("%m-%d_%H:%M:%S")
predictions.columns = ["wp" + str(i) for i in range(1, 7)]
predictions.to_csv(path_or_buf="submit.csv", sep=";")
