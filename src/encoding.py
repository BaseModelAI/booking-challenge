import argparse
import logging
import os
import pickle
import numpy as np
import pandas as pd
from typing import List
from coders import DLSH
from cleora import prepare_cleora_directed_input, run_cleora_directed


log = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True, help='Filename of training dataset')
    parser.add_argument("--test", type=str, required=True, help='Filename of validation dataset')
    parser.add_argument("--working-dir", type=str, default = 'data', help='Directory where files will be saved')
    parser.add_argument("--cleora-input-filename", type=str, default = 'cleoraInput', help='Filename of the input to the Cleora')
    parser.add_argument("--sketch-dim", type=int, default = 128, help='Sketch width')
    parser.add_argument("--n-sketches", type=int, default = 40, help='Cleora sketch depth')
    parser.add_argument("--n-sketches-random", type=int, default = 80, help='Random sketch depth')
    parser.add_argument("--cleora-iterations", nargs="+", default=[1,3], help='Iteration numbers of cleora')
    parser.add_argument("--cleora-dim", type=int, default = 1024, help='Emedding length of Cleora embedding')
    parser.add_argument("--codes-filename", type=str, default='codes', help='Filename of the final file with sketches for each city')
    return parser


def compute_codes(embeddings: np.ndarray, n_sketches: int, sketch_dim: int):
    """
    Compute LSH codes
    """
    emb_dim = embeddings.shape[1]
    vcoder = DLSH(n_sketches, sketch_dim)
    vcoder.fit(embeddings)
    codes = vcoder.transform_to_absolute_codes(embeddings)
    return codes


def merge_modalities(all_cities: List[str], modalities: List[dict], offsets: List[int]):
    city2codes = {}
    for city in all_cities:
        codes = []
        for i, modality in enumerate(modalities):
            codes.append(modality[city] + offsets[i])
        city2codes[str(city)] = list(np.concatenate(codes))
    return city2codes


def main(params: dict):
    os.makedirs(params.working_dir, exist_ok=True)
    train = pd.read_csv(params.train, sep='\t')
    train = train.sort_values(['utrip_id', 'row_num'])

    test = pd.read_csv(params.test, sep='\t')
    test = train.sort_values(['utrip_id', 'row_num'])

    # remove rows from test set that will be predicted
    test = test.loc[test['row_num']!=test['total_rows']]

    data = pd.concat([train, test])
    all_cities = data['city_id'].unique()

    cleora_input_filename = os.path.join(params.working_dir, params.cleora_input_filename)

    prepare_cleora_directed_input(cleora_input_filename, data)
    modalities = []
    for iter_ in params.cleora_iterations:
        log.info(f"Run Cleora with iteration {iter_}")
        ids, embeddings = run_cleora_directed(params.working_dir, cleora_input_filename, params.cleora_dim, iter_, all_cities)
        log.info("Computing LSH codes")
        codes = compute_codes(embeddings, params.n_sketches, params.sketch_dim)
        modalities.append(dict(zip(ids, codes)))

    log.info("Generate random sketch codes")
    random_embeddings = np.random.normal(0, 0.1, size=[len(all_cities), 2048])
    random_codes = compute_codes(random_embeddings, n_sketches=params.n_sketches_random, sketch_dim=params.sketch_dim)
    modalities.append(dict(zip(all_cities, random_codes)))

    city2codes = merge_modalities(all_cities, modalities, offsets = [i * params.n_sketches * params.sketch_dim for i in range(len(modalities))])

    log.info(f"Saving LSH codes into {params.codes_filename}")
    with open(os.path.join(params.working_dir, params.codes_filename), 'wb') as handle:
        pickle.dump(city2codes, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = get_parser()
    params = parser.parse_args()
    main(params)