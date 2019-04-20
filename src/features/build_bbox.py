import io
import numpy as np
from sklearn import preprocessing
import pickle


def load_trajectory(trajectory_path, n=None, pruning=False):
    trajectory = {}
    with io.open(trajectory_path, buffering=io.DEFAULT_BUFFER_SIZE) as f:
        content = f.readlines()
        count = 0
        for line in content:

            pair = line.split()
            order_id = pair[0]
            values = pair[1]
            values = values.split(",")[:-1]
            values = [x.split(":")[1:] for x in values]
            values = [(float(x), float(y)) for (x,y) in values]

            #dictionary -  key: order_id, value: array of (x,y) coordinates for all timestamps, timestamp info not saved
            if pruning:
                if len(values) >= 50 and len(values) <= 400:  # remove too short/ too long trajectories
                    count += 1
                    trajectory[order_id] = values
            else:
                count += 1
                trajectory[order_id] = values

            if n is not None:
                if count%n == 0:
                    break
    return trajectory


def build_qgram(data, k=10):
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

    #key: trajectory id in string, value: encoded key
    return order_dict


def build_order_dict(id_list):
    order_dict = {}
    le_user = preprocessing.LabelEncoder()
    le_user.fit(id_list)
    for id, num in zip(id_list, le_user.transform(id_list)):
        order_dict[num] = id

    #key: key, value: trajectory id in string
    return order_dict


def read_pickle(path):
    # print(path)
    try:
        with open(path, 'rb') as openfile:
            objects = pickle.load(openfile)
    except UnicodeDecodeError:
        with open(path, 'rb') as openfile:
            objects = pickle.load(openfile, encoding="latin1")
    except Exception as e:
        print(e)
        raise
    openfile.close()
    return objects


def save_pickle(obj, path):
    with (open(path, "wb")) as openfile:
        pickle.dump(obj, openfile, protocol=2)
    openfile.close()
    return True


def build_coordinate(traj_id, traj_data):
    try:
        data = traj_data[traj_id]  # list of tuples
    except KeyError:
        return None
    x = []
    y = []
    for t in data:
        x.append(t[1])
        y.append(t[0])
    return x, y


def load_top_k(file_path, k):
    f = open(file=file_path)
    id = next(f)
    top_k = []
    for i in range(k):
        row = next(f)
        row = row.strip()
        row = row.split(' ')
        top_k.append(int(row[0]))
    return top_k
