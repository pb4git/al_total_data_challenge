# al_total_data_challenge

Run scripts in the following order :

* 1.py to create the pre-processed *data.1.1.0.0.parquet.gzip* file
* 2_single_farm.py and 2_all_farms.py to create many single model predictions in the model_predictions folder.
* 3.py to combine single model predictions into one (better) final prediction