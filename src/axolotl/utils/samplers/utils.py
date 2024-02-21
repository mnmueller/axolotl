"""
helper util to calculate dataset lengths
"""
import numpy as np
from pyarrow import ChunkedArray

def get_dataset_lengths(dataset):
    if "length" in dataset.data.column_names:
        lengths = np.array(dataset.data.column("length"))
    elif "position_ids" in dataset.data.column_names:
        position_ids = dataset.data.column("position_ids")
        if isinstance(position_ids,ChunkedArray):
            lengths = np.array([x[-1].as_py() + 1 for x in position_ids])
        else:
            lengths = np.array([int(x[-1]) + 1 for x in position_ids])
    else:
        input_ids = dataset.data.column("input_ids")
        lengths = np.vectorize(len)(np.array(input_ids, dtype=object))
        return lengths
    return lengths
