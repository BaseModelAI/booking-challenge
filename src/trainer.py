import numpy as np
import pytorch_lightning as pl
import torch
import logging


log = logging.getLogger(__name__)


def flatten(t):
    return [item for sublist in t for item in sublist]


def decode(sketch, product_decoder_codes):
    return torch.exp(torch.mm(torch.log(1e-9+sketch), product_decoder_codes))


def categorical_cross_entropy(y_pred, y_true):
    y_pred = torch.clamp(y_pred, 1e-9, 1 - 1e-9)
    return -(y_true * torch.log(y_pred)).sum(dim=1).mean()


class BookingTrainer(pl.LightningModule):
    def __init__(self, model, learning_rate, sketch_dim, product_decoder_codes, all_cities):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.sketch_dim = sketch_dim
        self.product_decoder_codes = product_decoder_codes
        self.all_cities = all_cities

    def forward(self, batch):
        return self.model(batch['input_sketch'].float(), batch['device_class'], batch['booker_country'], batch['affiliate_id'],
                    batch['is_target_last_city'], batch['hotel_country'], batch['features'].float(), batch['day_in'], batch['day_out'],
                    batch['month_in'], batch['year_in'], batch['weekday_in'], batch['weekday_out'])

    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss = categorical_cross_entropy(output.view(-1, self.sketch_dim).float(), batch['target'].float().view(-1,self.sketch_dim).float())
        return loss

    def training_epoch_end(self, losses):
        log.info(f"Training loss: {np.mean([i['loss'].item() for i in losses])}")

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        loss = categorical_cross_entropy(output.view(-1, self.sketch_dim).float(), batch['target'].float().view(-1,self.sketch_dim).float()).detach().item()

        op_geom = decode(output.detach().cpu(), self.product_decoder_codes)
        op_geom = op_geom.cpu().numpy()
        ranking = np.argsort(op_geom)[:, ::-1]
        top4 = ranking[:,:4]
        targets = batch['target_city'].cpu().numpy()

        acc = []
        for example_id in range(targets.shape[0]):
            if str(targets[example_id]) in [self.all_cities[c] for c in top4[example_id]]:
                acc.append(1)
            else:
                acc.append(0)
        return {'acc': acc, 'loss': loss}

    def validation_epoch_end(self, out):
        loss = [i['loss'] for i in out]
        acc = [i['acc'] for i in out]
        acc  = np.array(flatten(acc)).mean()
        loss  = np.array(loss).mean()
        log.info(f"Precission@4: {acc}")
        log.info(f"loss: {loss}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        return optimizer
