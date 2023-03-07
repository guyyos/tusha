import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
from pymc import HalfCauchy, Model, Normal, sample

data_file = {DATA_FILE}
target = {TARGET}

df = pd.read_csv(data_file)

features = list(set(df.columns).difference(set([target])))
print(f' target = {target}, features = {features}')

display(df.head())
display(df.info())
display(df.describe())
display(df.corr())