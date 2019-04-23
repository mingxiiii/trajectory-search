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


def build_qgram(data, q):
    order_id_list = []
    qgram = {}
    for order_id, values in data.items():
        order_id_list.append(order_id)
        grams = [values[idx:idx + q] for idx in range(len(values))]  # build q-grams
        grams_mean = [tuple(map(np.mean, zip(*x))) for x in grams]  # find q-gram means
        qgram[order_id] = [(np.around(x[0], decimals=5), np.around(x[1], decimals=5)) for x in grams_mean]  # find q-gram means
    return qgram, order_id_list


def build_id_to_key(id_list):
    order_dict = {}
    le_user = preprocessing.LabelEncoder()
    le_user.fit(id_list)
    for id, num in zip(id_list, le_user.transform(id_list)):
        order_dict[id] = num  # key: trajectory id in string, value: encoded key
    return order_dict


def swap_k_v(dictionary):
    return {v: k for k, v in dictionary.items()}


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
    next(f)
    top_k = []
    top_k_dist = {}
    for i in range(k):
        row = next(f)
        row = row.strip()
        row = row.split(' ')
        top_k.append(int(row[0]))
        top_k_dist[int(row[0])] = float(row[1])
    return top_k, top_k_dist


def match(coor1, coor2, threshold=0.05):
    return abs(coor1[0]-coor2[0]) <= threshold and abs(coor1[1]-coor2[1]) <= threshold


def calculateEdr(trajectory1, trajectory2):
    if len(trajectory1) == 0:
        return len(trajectory2)
    elif len(trajectory2) == 0:
        return len(trajectory1)
    else:
        return min(calculateEdr(trajectory1[1:], trajectory2[1:]) + subcost(trajectory1[0], trajectory2[0]),
                   calculateEdr(trajectory1[1:], trajectory2)+1,
                   calculateEdr(trajectory1, trajectory2[1:])+1)


def subcost(t1, t2):
    if match(t1, t2):
        return 1
    else:
        return 0
