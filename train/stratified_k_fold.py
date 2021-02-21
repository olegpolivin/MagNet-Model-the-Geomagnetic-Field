import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def get_train_val_split(data, n_splits):


    train_a = data[data['period'] =='train_a'].index
    train_b = data[data['period'] =='train_b'].index
    train_c = data[data['period'] =='train_c'].index

    # it is not a bug that it is train_a below.
    num_elem_a = len(train_a)//n_splits
    num_elem_b = len(train_a)//n_splits
    num_elem_c = len(train_a)//n_splits

    
    for i in range(n_splits):
        train_index, test_index = [], []
        train_index.extend(np.roll(train_a, shift = -i*num_elem_a)[:-num_elem_a])
        test_index.extend(np.roll(train_a, shift = -i*num_elem_a)[-num_elem_a:])

        train_index.extend(np.roll(train_b, shift = -i*num_elem_b)[:-num_elem_b])
        test_index.extend(np.roll(train_b, shift = -i*num_elem_b)[-num_elem_b:])

        train_index.extend(np.roll(train_c, shift = -i*num_elem_c)[:-num_elem_c])
        test_index.extend(np.roll(train_c, shift = -i*num_elem_c)[-num_elem_c:])

        yield train_index, test_index
