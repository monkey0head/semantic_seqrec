import warnings

def last_item_split(df, user_col='user_id', timestamp_col='timestamp'):
    """Split user sequences to input data and ground truth with one last item."""

    if df[user_col].value_counts().min() == 1:
         warnings.warn('Each user must have at least two interactions.')

    df = df.sort_values([user_col, timestamp_col], kind='stable')
    df['time_idx_reversed'] = df.groupby(user_col).cumcount(ascending=False)

    inputs = df[df['time_idx_reversed'] >= 1]
    last_item = df[df['time_idx_reversed'] == 0]

    inputs = inputs.drop(columns=['time_idx_reversed']).reset_index(drop=True)
    last_item = last_item.drop(columns=['time_idx_reversed']).reset_index(drop=True)

    return inputs, last_item
