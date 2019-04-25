# -*- coding: utf-8 -*-
import logging
import io
import re
import sys


def main(raw_data, processed_data):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger('make_trajectory')
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler('./log/%s' % raw_data)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    input_filepath = './data/raw/%s.txt' % raw_data
    output_filepath = './data/processed/%s.txt' % processed_data

    logger.info('---------------------------- Making trajectory sequence from raw data ----------------------------')
    logger.info('Load raw data from: %s' % input_filepath)
    f = io.open(input_filepath, 'r', encoding="utf8", buffering=io.DEFAULT_BUFFER_SIZE)
    data = {}
    counter = 0
    for row in f:
        counter += 1
        row = row.strip()
        row_token = re.split(r',', row)
        orderId = row_token[1]
        time = row_token[2]
        x = row_token[3]
        y = row_token[4]
        val = [time, x, y]
        try:
            data[orderId].append(val)
        except KeyError:
            data[orderId] = []
            data[orderId].append(val)
        # if counter % 5000 == 0:
        #     break
    logger.info('Write trajectory sequence to: %s' % output_filepath)
    f = io.open(output_filepath, 'w', encoding="utf8", buffering=io.DEFAULT_BUFFER_SIZE)
    for key, val in data.items():
        if len(val) >= 50 and len(val) <= 400:  # excluding too long and too short trajectories
            f.write(key + '\t')
            time_stamp = [int(e[0]) for e in val]
            time_stamp_len = len(time_stamp)
            sorted_ix = sorted(range(time_stamp_len), key=lambda k: time_stamp[k])
            for ix in sorted_ix:
                item = val[ix]
                f.write(':'.join(item) + ',')
            f.write('\r\n')
    f.close()
    logger.info('Finished')


if __name__ == '__main__':
    d_in = sys.argv[1]
    d_out = sys.argv[2]
    main(d_in, d_out)


