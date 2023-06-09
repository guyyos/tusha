from scipy.stats import entropy
from pandas import Series
import numpy as np
import pandas as pd
from enum import Enum


def is_counts(vals):
    return min(vals) >= 0 and vals.apply(lambda x: x.is_integer()).all()

def safe_log(vals):
    if min(vals)<0:
        raise ValueError
    if min(vals)==0:
        vals = vals+0.1
    
    return np.log(vals)

def vals_entropy(vals):
    num_bins = min(20,len(vals.value_counts()))
    print(vals)
    print(num_bins)
    return entropy(pd.cut(vals, num_bins).value_counts().values)


def is_none_val(v):

    try:
        return v is None or str(v).strip() == '' or np.isnan(v)
    except:
        return False


def possible_none(v):
    if is_none_val(v):
        return True
    try:
        fv = float(v)
        if fv == 0:
            return True
        return False
    except:
        return False


def possible_float(v):
    if v is None or type(v) == bool:
        return False
    try:
        fv = float(v)
        if np.isnan(fv):
            return False
        return True
    except:
        return False


def definitly_float(v):
    return type(v) == float


def definitly_int(v):
    return type(v) == int


def definitly_bool(v):
    return type(v) == bool


def possible_int(v):
    if v is None or type(v) == bool:
        return False
    try:
        fv = float(v)
        if np.isnan(fv) or fv != int(fv):
            return False
        return True
    except:
        return False


def convert_to_int(v):
    try:
        return int(v)
    except:
        return v


def convert_series_to_int(s):
    try:
        s = s.astype('int')
        return s
    except:
        return s.apply(convert_to_int)


def extract_int_vals(s):
    try:
        s = s.astype('int').dropna()
        return s
    except:
        int_s = s.apply(lambda v: convert_to_int(
            v) if possible_int(v) else np.nan).dropna()
        return int_s


def possible_bool(v):
    if type(v) == bool:
        return True

    sv = str(v).lower()
    if sv == 'true' or sv == 'false':
        return True

    try:
        fv = float(v)
        if fv == 0 or fv == 1:
            return True

        return False
    except:
        return False


class FeatureType(Enum):
    BOOL = 1
    CATEGORICAL = 2
    NUMERICAL = 3
    ORDINAL = 4
    COUNT = 5


def identify_series(vals):
    # series can be :
    #  boolean (True/False or 0/1 except Nones),
    #  numerical,
    #  count (integers 0-inf) if vals_entropy(log(vals))>vals_entropy(vals)
    #  ordinal (integers between 0-10)
    #  categorical
    # null values: None, 'None', '  ', 0 not in categorical  => transofrm to None
    # if one value only (categorical or numerical) and only None's , then its a boolean with and None is False
    df_vals = vals.value_counts(dropna=False).rename_axis(
        'unique_values').reset_index(name='counts')

    num_vals = len(vals)
    num_unique_vals = len(df_vals)

    df_vals['possible_int'] = df_vals.unique_values.apply(possible_int)
    df_vals['possible_float'] = df_vals.unique_values.apply(possible_float)
    df_vals['definitly_float'] = df_vals.unique_values.apply(definitly_float)
    df_vals['definitly_bool'] = df_vals.unique_values.apply(definitly_bool)

    num_bool_vals = df_vals[df_vals['definitly_bool']]["counts"].sum()

    if num_unique_vals <= 2:
        return FeatureType.BOOL

    if num_bool_vals >= 0.5*len(vals):
        return FeatureType.BOOL

    num_unique_int_vals = len(df_vals[df_vals['possible_int']])
    num_int_vals = df_vals[df_vals['possible_int']]["counts"].sum()

    num_def_float = len(df_vals[df_vals['definitly_float']])

    if num_def_float <= 0 and num_int_vals > 0.5*num_vals:

        min_int_val = df_vals[df_vals.possible_int]['unique_values'].min()

        val_counts = df_vals[df_vals.possible_int][['counts']]

        if min_int_val >= 0:
            ivals = extract_int_vals(vals)
            print(f'ivals = {ivals}')
            if vals_entropy(safe_log(ivals)) > vals_entropy(ivals):
                return FeatureType.COUNT
        return FeatureType.ORDINAL

    num_possible_float = len(df_vals[df_vals['possible_float']])

    if num_possible_float > 0.5*num_vals:
        return FeatureType.NUMERICAL

    return FeatureType.CATEGORICAL
