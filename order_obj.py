import pandas as pd
import collections


def order_obj(obj):
    new_obj = {}
    for k, v in obj.items():
        new_k = int(k)
        new_obj[new_k] = v
    new_obj = collections.OrderedDict(sorted(new_obj.items()))
    return new_obj


if __name__ == '__main__':
    file_path = r"E:\rafi\got_your_back\data\toyota\1\T1_res_pkl.pkl"
    obj = pd.read_pickle(file_path)
    order_obj(obj)