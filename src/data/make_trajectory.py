# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import io
import re
import sys


def main(input_filepath, output_filepath, n=50):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

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

    f = io.open(output_filepath, 'w', encoding="utf8", buffering=io.DEFAULT_BUFFER_SIZE)
    for key, val in data.items():
        if len(val) >= 50 and len(val) <= 400:
            f.write(key + '\t')
            time_stamp = [int(e[0]) for e in val]
            time_stamp_len = len(time_stamp)
            sorted_ix = sorted(range(time_stamp_len), key=lambda k: time_stamp[k])
            for ix in sorted_ix:
                item = val[ix]
                f.write(':'.join(item) + ',')
            f.write('\r\n')
    f.close()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    f_in = sys.argv[1]
    f_out = sys.argv[2]

    main(f_in, f_out)


