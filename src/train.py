import pickle
import argparse
import logging
import os
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from dataset import BookingsDataset
from trainer import BookingTrainer
from model import Model
from torch.utils.data import DataLoader
import pytorch_lightning as pl


log = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--working-dir", type=str, default = 'data', help='Directory where files will be saved')
    parser.add_argument("--test-datapoints", type=str, default = 'testDatapoints', help='Validation datapoints')
    parser.add_argument("--train-datapoints", type=str, default = 'trainDatapoints', help='Training datapoints')
    parser.add_argument("--codes-filename", type=str, default='codes', help='Filename of final sketches for each city')
    parser.add_argument("--sketch-dim", type=int, default = 128, help='Sketch width')
    parser.add_argument("--batch-size", type=int, default = 128, help='Batch size')
    parser.add_argument("--lr", type=float, default = 5e-4, help='Learning rate')
    parser.add_argument("--hidden-size", type=int, default = 3000, help='Hidden size')
    return parser


def load_data(params: dict):
    log.info("Loading data")
    with open(os.path.join(params.working_dir, params.train_datapoints), 'rb') as f:
        train_datapoints = pickle.load(f)

    with open(os.path.join(params.working_dir, params.test_datapoints), 'rb') as f:
        validation_datapoints = pickle.load(f)

    with open(os.path.join(params.working_dir, params.codes_filename), 'rb') as f:
        city2codes = pickle.load(f)

    return train_datapoints, validation_datapoints, city2codes


def sparse2dense(codes, input_dim):
    res = np.zeros(input_dim, dtype=np.float32)
    res[codes] += 1
    return res


def get_nunique_values(value, train, valid):
    return len(set([i[value] for i in train] + [i[value] for i in valid]))


N_SKETCHES = 3  # 3 sketches: first city, previous city, all other cities
N_FEATURES = 16*8 # 8 continuous features - each represents as 16 scales of sin and cos. Check `dataset.encode_scalar_column` function


def main(params: dict):
    train_datapoints, validation_datapoints, city2codes = load_data(params)
    n_sketches = len(city2codes[list(city2codes.keys())[0]])
    log.info(f"Sketch depth: {n_sketches}")

    train_non_final_destinations = [i for i in train_datapoints if not i['is_target_last_city']]
    train_final_destinations = [i for i in train_datapoints if i['is_target_last_city']]

    input_dim = params.sketch_dim * n_sketches

    train_final_dataset = BookingsDataset(train_final_destinations, city2codes, input_dim, params.sketch_dim)
    train_final_loader = DataLoader(train_final_dataset, batch_size=params.batch_size, num_workers=10, shuffle=True, drop_last=False)

    train_non_final_dataset = BookingsDataset(train_non_final_destinations, city2codes, input_dim, params.sketch_dim)
    train_non_final_loader = DataLoader(train_non_final_dataset, batch_size=params.batch_size, num_workers=10, shuffle=True, drop_last=False)

    valid_dataset = BookingsDataset(validation_datapoints, city2codes, input_dim, params.sketch_dim)
    valid_loader = DataLoader(valid_dataset, batch_size=params.batch_size, num_workers=10, shuffle=False, drop_last=False)

    all_cities = list(city2codes.keys())
    all_codes = [city2codes[city] for city in all_cities]
    absolute_codes = np.array(all_codes)
    product_decoder_codes = [sparse2dense(absolute_codes[k], input_dim) for k in range(absolute_codes.shape[0])]
    product_decoder_codes = np.vstack(product_decoder_codes)
    product_decoder_codes /= product_decoder_codes.sum(-1, keepdims=True)
    product_decoder_codes = product_decoder_codes.T
    product_decoder_codes = torch.from_numpy(product_decoder_codes)

    n_years = get_nunique_values('year_in', train_datapoints, validation_datapoints)
    n_device_class = get_nunique_values('device_class', train_datapoints, validation_datapoints)
    n_booker_country = get_nunique_values('booker_country', train_datapoints, validation_datapoints)
    n_affiliate_id = get_nunique_values('affiliate_id', train_datapoints, validation_datapoints)
    n_countries = get_nunique_values('hotel_country', train_datapoints, validation_datapoints)

    net = Model(n_sketches, params.sketch_dim, N_SKETCHES, params.hidden_size, n_years, n_device_class, n_booker_country,
                n_affiliate_id, n_countries, features_size=N_FEATURES)
    model = BookingTrainer(net, params.lr, params.sketch_dim, product_decoder_codes, all_cities)
    trainer = pl.Trainer(gpus=1,  max_epochs=1, logger=False, checkpoint_callback=False)

    # 1st epoch
    trainer.fit(model, train_non_final_loader)
    trainer.fit(model, train_final_loader)

    # 2nd epoch
    trainer.fit(model, train_non_final_loader)
    trainer.fit(model, train_final_loader)

    # 3rd epoch only on final destinations
    model.learning_rate = params.lr / 10
    trainer.fit(model, train_final_loader, valid_loader)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = get_parser()
    params = parser.parse_args()
    main(params)