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
    bbox = {}  # dictionary - key:order_id, value: (x_min,y_min,x_max,y_max)
    order_id_list = []
    qgram = {}
    for order_id, values in data.items():
        order_id_list.append(order_id)
        grams = [values[idx:idx + k] for idx in range(len(values))]  # build q-grams
        qgram[order_id] = [tuple(map(np.mean, zip(*x))) for x in grams]  # find q-gram means
        bbox[order_id] = get_bbox(qgram[order_id])  # calculate bounding box range
    return qgram, bbox, order_id_list


def build_id_dict(id_list):
    order_dict = {}
    le_user = preprocessing.LabelEncoder()
    le_user.fit(id_list)
    for id, num in zip(id_list, le_user.transform(id_list)):
        order_dict[id] = num
    return order_dict


def read_id_dict(path):
    objects = []
    with (open(path, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    return objects


def save_id_dict(obj_list, path):
    with (open(path, "wb")) as openfile:
        for obj in obj_list:
            pickle.dump(obj, openfile)
        openfile.close()
    return True

