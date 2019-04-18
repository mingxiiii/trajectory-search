from src.features.build_bbox import save_pickle, read_pickle


# rtree_id_dict = read_pickle('./data/processed/rtree_id_dict.txt')
# save_pickle(rtree_id_dict, './data/processed/rtree_id_dict2.txt')

query_id_dict = read_pickle('./data/processed/query_id_dict.txt')
query_id_dict_reverse = {v: k for k, v in query_id_dict.items()}
print(query_id_dict_reverse[5])