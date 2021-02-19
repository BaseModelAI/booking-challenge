import argparse
import logging
import os
import random
import pandas as pd
import numpy as np
from collections import Counter


log = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True, help='Filename of training dataset provided by challenge organizers')
    parser.add_argument("--validation-size", type=int, default=70662, help='Number of trips in validation dataset')
    parser.add_argument("--train-output-filename", type=str, default='train.csv', help='Filename of output training dataset')
    parser.add_argument("--valid-output-filename", type=str, default='valid.csv', help='Filename of output validation dataset')
    parser.add_argument("--ground-truth-filename", type=str, default='ground_truth.csv', help='Filename of ground truth for validation dataset')
    parser.add_argument("--working-dir", type=str, default = 'data', help='Directory where files will be saved')
    return parser


def preprocess_data(data, utrips_counter):
    """
    Released test set contains additional columns: `row_num`, `total_rows`.
    Those columns are added here.
    """
    data['total_rows'] = data.apply(lambda row: utrips_counter[row['utrip_id']], axis = 1)
    row_num = []
    counter = 1
    for row in data.itertuples():
        row_num.append(counter)
        counter += 1
        if counter > row.total_rows:
            counter = 1
    data['row_num'] = row_num


def get_validation_utrips(data, utrips_less_than_4, validation_size):
    val_utrips = list()
    train_cities_so_far = set()
    utrip_cities = []

    for row in data.itertuples():
        utrip_id = row.utrip_id
        if utrip_id in utrips_less_than_4:
            # test set has at least 4 cities in a trip
            continue

        utrip_cities.append(row.city_id)
        if row.total_rows == row.row_num:
            if all(elem in train_cities_so_far for elem in utrip_cities) and random.random() < 0.5:
                val_utrips.append(row.utrip_id)
            else:
                train_cities_so_far.update(set(utrip_cities))
            utrip_cities = []

        if len(val_utrips) == validation_size:
            break

    log.info(f"Number of validation trips: {len(val_utrips)}")
    return val_utrips


def get_ground_truth(test):
    ground_truth = []
    for i, row in test.iterrows():
        if row['row_num'] == row['total_rows']:
            # this city should be predicted
            ground_truth.append({'utrip_id': row['utrip_id'],
                                'city_id': row['city_id'],
                                'hotel_country': row['hotel_country']})
            test.at[i, 'city_id'] = np.int64(0)
            test.at[i, 'hotel_country'] = ''
    return ground_truth


def main(params: dict):
    os.makedirs(params.working_dir, exist_ok=True)
    data = pd.read_csv(params.train, parse_dates=['checkin', 'checkout'])
    utrips_counter = Counter(data['utrip_id'])
    utrips_single = []
    utrips_less_than_4 = []
    for k, v in utrips_counter.items():
        if v == 1:
            utrips_single.append(k)
        if v < 4:
            utrips_less_than_4.append(k)

    log.info(f"Remove {len(utrips_single)} trips with single row")
    data = data.loc[~data['utrip_id'].isin(utrips_single)]
    data = data.sort_values(['utrip_id', 'checkin'])
    data.reset_index(inplace=True, drop=True)
    preprocess_data(data, utrips_counter)
    val_utrips = get_validation_utrips(data, utrips_less_than_4, params.validation_size)

    train = data.loc[~data['utrip_id'].isin(val_utrips)]
    train.reset_index(inplace=True, drop=True)

    test = data.loc[data['utrip_id'].isin(val_utrips)]
    test.reset_index(inplace=True, drop=True)
    log.info(f"Length of train set: {len(train)}")
    log.info(f"Length of test set: {len(test)}")

    ground_truth = get_ground_truth(test)
    pd.DataFrame(ground_truth).to_csv(os.path.join(params.working_dir, params.ground_truth_filename), index=False, sep='\t')
    train.to_csv(os.path.join(params.working_dir, params.train_output_filename), index=False, sep='\t')
    test.to_csv(os.path.join(params.working_dir, params.valid_output_filename), index=False, sep='\t')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = get_parser()
    params = parser.parse_args()
    main(params)