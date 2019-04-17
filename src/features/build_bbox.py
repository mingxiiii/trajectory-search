import io
import numpy as np
from sklearn import preprocessing
import pickle


def load_trajectory(trajectory_path, n=None):
    trajectory = {}
    with io.open(trajectory_path, buffering=io.DEFAULT_BUFFER_SIZE) as f:
        content = f.readlines()
        count = 0
        for line in content:
            count += 1
            pair = line.split()
            order_id = pair[0]
            values = pair[1]
            values = values.split(",")[:-1]
            values = [x.split(":")[1:] for x in values]
            values = [(float(x), float(y)) for (x,y) in values]
            #d ictionary -  key: order_id, value: array of (x,y) coordinates for all timestamps, timestamp info not saved
            if len(values) > 50 and len(values) < 400:
                trajectory[order_id] = values
                if n is not None:
                    if count%n == 0:
                        break
    return trajectory


def get_bbox(qmean_list):
    x_min = y_min = 10 ** 9
    x_max = y_max = -1
    for (x, y) in qmean_list:
        if x < x_min:
            x_min = x
        if x > x_max:
            x_max = x
        if y < y_min:
            y_min = y
        if y > y_max:
            y_max = y
    return x_min, y_min, x_max, y_max


def build_qgram(data, k=20):
    order_id_list = []
    qgram = {}
    for order_id, values in data.items():
        order_id_list.append(order_id)
        grams = [values[idx:idx + k] for idx in range(len(values))]  # build q-grams
        grams_mean = [tuple(map(np.mean, zip(*x))) for x in grams]  # find q-gram means
        qgram[order_id] = [(np.around(x[0], decimals=5), np.around(x[1], decimals=5)) for x in grams_mean]  # find q-gram means
    return qgram, order_id_list


def build_id_dict(id_list):
    order_dict = {}
    le_user = preprocessing.LabelEncoder()
    le_user.fit(id_list)
    for id, num in zip(id_list, le_user.transform(id_list)):
        order_dict[id] = num
    return order_dict


def read_pickle(path):
    print(path)
    with open(path, 'rb') as openfile:
        while True:
            try:
                objects = pickle.load(openfile)
            except EOFError:
                break
    openfile.close()
    return objects


def save_pickle(obj, path):
    with (open(path, "wb")) as openfile:
        pickle.dump(obj, openfile)
    openfile.close()
    return True

