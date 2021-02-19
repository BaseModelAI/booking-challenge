import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import normalize


def normalize_sketch(sketch, sketch_dim, input_dim, norm='l2'):
    """
    Normalize sketch of all products or create zeros if doesn't have sketch
    """
    if sketch.shape[0] > 0:
        sketch = normalize(np.sum(sketch, axis=0).reshape(-1, sketch_dim), norm).reshape((input_dim,))
    else:
        sketch = np.zeros((input_dim,))
    return sketch


def multiscale(x, scales):
    return np.hstack([x.reshape(-1,1)/pow(2., i) for i in scales])


def encode_scalar_column(x, scales=[-1, 0, 1, 2, 3, 4, 5, 6]):
    return np.hstack([np.sin(multiscale(x, scales)), np.cos(multiscale(x, scales))])


class BookingsDataset(Dataset):
    def __init__(self, data, city2codes, input_dim, sketch_dim, decay_value = 0.95):
        self.data = data
        self.city2codes = city2codes
        self.decay_value = decay_value
        self.input_dim = input_dim
        self.sketch_dim = sketch_dim
    
    def __len__(self):
        return len(self.data)
    
    def codes_to_sketch(self, codes):
        """
        Convert abosulte codes into sketch sparse vector
        """
        x = np.zeros(self.input_dim)
        for ind in codes:
            x[ind] += 1
        return x

    def __getitem__(self, idx):
        example = self.data[idx]
        cities = example['cities']

        # create three sketches: 
        # `first_city_sketch` representing the first city in a trip
        # `prev_city_sketch` representing the previous city in a trip,
        # `all_cities_sketch` representing all other cities
        first_city_sketch = self.codes_to_sketch(self.city2codes[str(cities[0])])
        prev_city_sketch = self.codes_to_sketch(self.city2codes[str(cities[-1])])

        all_cities_sketch = np.zeros(self.input_dim)
        
        for city in cities[1:-1]:
            all_cities_sketch *= self.decay_value
            all_cities_sketch += self.codes_to_sketch(self.city2codes[str(city)])
        all_cities_sketch = normalize(all_cities_sketch.reshape(-1, self.sketch_dim), 'l2').reshape((self.input_dim,))

        # create features
        num_days = (example['checkout'] - example['checkin']).days
        num_cities_so_far = len(cities)
        days_since_last_booking = (example['checkin']-example['checkout_last']).days
        previous_num_days = (example['checkout_last'] - example['checkin_last']).days

        since_first_checkin = (example['checkin'] - example['first_checkin']).days
        till_last_checkin = (example['last_checkin'] - example['checkin']).days

        features = encode_scalar_column(np.array([num_days, num_cities_so_far, days_since_last_booking, 
                    example['num_cities_in_trip'], previous_num_days, since_first_checkin, till_last_checkin,
                    len(set(cities))])).flatten()

        result = {'input_sketch': np.concatenate([first_city_sketch, all_cities_sketch, prev_city_sketch]),
                  'device_class': example['device_class'],
                  'booker_country': example['booker_country'],
                  'affiliate_id': example['affiliate_id'],
                  'is_target_last_city': example['is_target_last_city'],
                  'hotel_country': example['hotel_country'],
                  'features': features,
                  'day_in': example['day_in'],
                  'day_out': example['day_out'],
                  'month_in': example['month_in'],
                  'year_in': example['year_in'],
                  'weekday_in': example['checkin'].weekday(),
                  'weekday_out': example['checkout'].weekday(),
                  'target_city': example['target'],
                  'target': self.codes_to_sketch(self.city2codes[str(example['target'])])
                  }

        return result