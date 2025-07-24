"""
Make user and item indexes.
"""

import pandas as pd


def encode(data, col_name, shift):
    """Encode items/users to consecutive ids.

    :param col_name: column to do label encoding, e.g. 'item_id'
    :param shift: shift encoded values to start from shift
    """
    data[col_name + '_old'] = data[col_name]
    data[col_name] = data[col_name].astype("category").cat.codes + shift
    return data


def encode_with_desc(interactions, desc, col_name, shift):
    """Encode items/users to consecutive ids.

    :param interactions: interactions dataframe
    :param desc: description dataframe
    :param col_name: column to do label encoding, e.g. 'item_id'
    :param shift: shift encoded values to start from shift

    :return: encoded interactions and desc, num warm items
    """
    codes = interactions[col_name].drop_duplicates().astype("category").cat
    encoder = dict(zip(codes.categories.values, codes.codes.values + shift))
    unk_id = codes.categories.shape[0] + shift
    interactions[col_name] = interactions[col_name].map(encoder, na_action='ignore').fillna(unk_id).astype(int)
    desc[col_name] = desc[col_name].map(encoder, na_action='ignore').fillna(unk_id).astype(int)
    return interactions, desc, unk_id