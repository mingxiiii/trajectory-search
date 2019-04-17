from src.features.build_bbox import *
import sys
from src.models.predict_model import calculateEdr


def main(query_path, train_path, query_id_path, train_id_path):
    train_data = load_trajectory(train_path)
    query_data = load_trajectory(query_path)
    query_id_dict = read_pickle(query_id_path)
    train_id_dict = read_pickle(train_id_path)

    result = []
    for query_id, query_trajectory in query_data.items():
        query_key = query_id_dict[query_id]
        distance_list = []
        train_key_list = []
        for train_id, train_trajectory in train_data.items():
            train_key = train_id_dict[train_id]
            distance = calculateEdr(train_trajectory, query_trajectory)
            distance_list.append(distance)
            train_key_list.append(train_key)

        ix = sorted(range(len(distance_list)), key=lambda k: distance_list[k])
        # distance_list_sorted = [distance_list[i] for i in ix]
        train_key_sorted = [train_key_list[i] for i in ix]
        result.append([query_key, train_key_sorted])


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: query trajectory <file>, train trajectory <file>, query id dictionary <file>, train id dictionary <file>", file=sys.stderr)
        sys.exit(-1)
    query = sys.argv[1]
    train = sys.argv[2]
    query_id = sys.argv[3]
    train_id = sys.argv[4]
    main(query_path=query, train_path=train, query_id_path=query_id, train_id_path=train_id)
