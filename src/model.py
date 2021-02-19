import torch
import torch.nn as nn
import torch.nn.functional as F


def trunc_normal_(x, mean=0., std=1.):
    "Truncated normal initialization (approximation)"
    # From https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
    return x.normal_().fmod_(2).mul_(std).add_(mean)


class Embedding(nn.Embedding):
    "Embedding layer with truncated normal initialization"
    def __init__(self, ni, nf, std=0.01):
        super().__init__(ni, nf)
        trunc_normal_(self.weight.data, std=std)


SMALL_EMBEDDING_SIZE = 20
BIG_EMBEDDING_SIZE = 120


class Model(nn.Module):
    def __init__(self, n_sketches_all, sketch_dim, num_count_sketches_input, hidden_size, n_years, n_device_class,
                    n_booker_country, n_affiliate_id, n_countries, features_size):
        super().__init__()
        input_dim = n_sketches_all * sketch_dim * num_count_sketches_input  + SMALL_EMBEDDING_SIZE*8 \
                            + BIG_EMBEDDING_SIZE*2 + features_size + 1
        self.year = Embedding(n_years, SMALL_EMBEDDING_SIZE)
        self.month = Embedding(12, SMALL_EMBEDDING_SIZE)
        self.day_checkin = Embedding(31, SMALL_EMBEDDING_SIZE)
        self.day_checkout = Embedding(31, SMALL_EMBEDDING_SIZE)
        self.device_class = Embedding(n_device_class, SMALL_EMBEDDING_SIZE)
        self.booker_country = Embedding(n_booker_country, SMALL_EMBEDDING_SIZE)
        self.affiliate_id = Embedding(n_affiliate_id, BIG_EMBEDDING_SIZE)
        self.last_hotel_country = Embedding(n_countries, BIG_EMBEDDING_SIZE)
        self.wd_in = Embedding(7, SMALL_EMBEDDING_SIZE)
        self.wd_out = Embedding(7, SMALL_EMBEDDING_SIZE)

        self.n_sketches_all = n_sketches_all
        self.output_dim = n_sketches_all * sketch_dim
        self.sketch_dim = sketch_dim
        self.l1 = nn.Linear(input_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, hidden_size)
        self.l_output = nn.Linear(hidden_size, self.output_dim)
        self.projection = nn.Linear(input_dim, hidden_size)

        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.bn4 = nn.BatchNorm1d(hidden_size)

    def forward(self, sketches, device, booker_country, affiliate_id, is_target_last_city, hotel_country, features,
                    day_in, day_out, month_in, year_in, weekday_in, weekday_out):
        """
        Feed forward network with residual connections.
        """
        x_input = torch.cat((sketches, self.device_class(device), self.booker_country(booker_country), self.affiliate_id(affiliate_id),
                        is_target_last_city[:,None], self.last_hotel_country(hotel_country), features, self.day_checkin(day_in),
                        self.day_checkout(day_out), self.month(month_in), self.year(year_in), self.wd_in(weekday_in),
                        self.wd_out(weekday_out)), axis=-1)
        x_proj = self.projection(x_input)
        x_ = self.bn1(F.leaky_relu(self.l1(x_input)))
        x = self.bn2(F.leaky_relu(self.l2(x_) + x_proj))
        x = self.bn3(F.leaky_relu(self.l3(x) + x_proj))
        x = self.l_output(self.bn4(F.leaky_relu(self.l4(x) + x_)))
        x = F.softmax(x.view(-1, self.n_sketches_all, self.sketch_dim), dim=2).view(-1, self.output_dim)
        return x