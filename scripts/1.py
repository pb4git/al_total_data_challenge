import numpy as np
import pandas as pd

import al_total_data_challenge.readrawdata
data, test_dates = al_total_data_challenge.readrawdata.readrawdata(r"..\data\challenge_19_data\\")

import al_total_data_challenge.featureengineering
data = al_total_data_challenge.featureengineering.featureengineering(data) 

data.to_parquet('data.1.1.0.0.parquet.gzip',
                engine = 'fastparquet',
                compression='gzip')
