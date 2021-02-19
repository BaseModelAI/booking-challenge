import argparse
import logging
import os
import pickle
from tqdm import tqdm
import pandas as pd
from typing import List


log = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True, help='Filename of training dataset')
    parser.add_argument("--test", type=str, required=True, help='Filename of validation dataset')
    parser.add_argument("--ground-truth", type=str, required=True, help='Filename of ground truth predictions for validation dataset')
    parser.add_argument("--working-dir", type=str, default = 'data', help='Directory where files will be saved')
    parser.add_argument("--test-datapoints", type=str, default = 'testDatapoints', help='Validation datapoints')
    parser.add_argument("--train-datapoints", type=str, default = 'trainDatapoints', help='Training datapoints')
    return parser


def parse_date(date, min_year):
    """
    Parse date and convert it indicies starting from 0
    """
    return date.day-1, date.month-1, date.year - min_year


def update_mapping(mapping, value):
    if not value in mapping:
        mapping[value] = len(mapping)


def preprocess_dataframe(data: pd.DataFrame, ground_truth: dict, utrip2lastcheckin: dict, utrip2firstcheckin:dict):
    booker_country2id = {}
    affiliate_id2id = {}
    device_class2id = {}
    hotel_country2id = {}

    min_year = data['checkin'].loc[data['total_rows'] == data['row_num']].min().year
    training_datapoints = []
    validation_datapoints = []
    utrip_cities = []

    log.info("Start preprocessing dataframe")
    for i, row in tqdm(data.iloc[:-1].iterrows(), total=data.shape[0]):
        if row['row_num'] == row['total_rows']:
            continue
        target_row = data.iloc[i+1]
        utrip_id = row['utrip_id']
        utrip_cities.append(row['city_id'])
        assert target_row['utrip_id'] == row['utrip_id']

        day_in, month_in, year_in = parse_date(target_row['checkin'], min_year)
        day_out, _, _ = parse_date(target_row['checkout'], min_year)

        update_mapping(device_class2id, target_row['device_class'])
        update_mapping(affiliate_id2id, target_row['affiliate_id'])
        update_mapping(booker_country2id, target_row['booker_country'])
        update_mapping(hotel_country2id, row['hotel_country'])

        datapoint = {
            'checkout_last': row['checkout'],
            'checkin_last': row['checkin'],
            'day_in': day_in,
            'day_out': day_out,
            'month_in': month_in,
            'year_in': year_in,
            'checkout': target_row['checkout'],
            'checkin': target_row['checkin'],
            'cities': utrip_cities.copy(),
            'utrip_id': row['utrip_id'],
            'device_class': device_class2id[target_row['device_class']],
            'affiliate_id': affiliate_id2id[target_row['affiliate_id']],
            'booker_country': booker_country2id[target_row['booker_country']],
            'num_cities_in_trip': row['total_rows'],
            'hotel_country': hotel_country2id[row['hotel_country']],
            'first_checkin': utrip2firstcheckin[utrip_id],
            'last_checkin': utrip2lastcheckin[utrip_id]
        }

        if target_row['row_num'] == target_row['total_rows']:
            datapoint['is_target_last_city'] = True
            if target_row['is_train']:
                datapoint['target'] = target_row['city_id']
                training_datapoints.append(datapoint.copy())
            else:
                datapoint['target'] = ground_truth[row['utrip_id']]
                validation_datapoints.append(datapoint.copy())
            utrip_cities = []
        else:
            datapoint['is_target_last_city'] = False
            datapoint['target'] = target_row['city_id']
            training_datapoints.append(datapoint.copy())
    log.info("Finished preprocessing dataframe")
    return training_datapoints, validation_datapoints


def main(params: dict):
    os.makedirs(params.working_dir, exist_ok=True)
    train = pd.read_csv(params.train, sep='\t',  parse_dates=['checkin', 'checkout'])
    train = train.sort_values(['utrip_id', 'row_num'])

    test = pd.read_csv(params.test, sep='\t', parse_dates=['checkin', 'checkout'])
    test = test.sort_values(['utrip_id', 'row_num'])

    ground_truth_df = pd.read_csv(params.ground_truth, sep='\t')
    ground_truth = dict(zip(ground_truth_df['utrip_id'], ground_truth_df['city_id']))
    train['is_train'] = True
    test['is_train'] = False

    data = pd.concat([train, test])
    data['is_train'] = data['is_train'].astype('bool')
    data.reset_index(inplace=True, drop=True)

    utrip2lastcheckin = {}
    utrip2firstcheckin = {}

    for row in data.itertuples():
        if row.row_num == 1:
            utrip2firstcheckin[row.utrip_id] = row.checkin
        elif row.row_num == row.total_rows:
            utrip2lastcheckin[row.utrip_id] = row.checkin

    training_datapoints, validation_datapoints = preprocess_dataframe(data, ground_truth, utrip2lastcheckin, utrip2firstcheckin)

    with open(os.path.join(params.working_dir, params.train_datapoints), 'wb') as f:
        pickle.dump(training_datapoints, f)

    with open(os.path.join(params.working_dir, params.test_datapoints), 'wb') as f:
        pickle.dump(validation_datapoints, f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = get_parser()
    params = parser.parse_args()
    main(params)