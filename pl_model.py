import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from models.keypointdetr import KeypointDETR
from data.st_data import KPS_Geodesic_Dataset
from utils.loss import Criterion


class LitModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.save_hyperparameters()
        self.net = KeypointDETR(args)
        self.criterion = Criterion(args)

    def forward(self, x):
        return self.net.forward(x)

    def infer(self, pc, g_heat):
        pts, gts  = self.net.inference(pc, g_heat)
        return pts, gts

    def training_step(self, batch, _):
        pc, heat, _ = batch

        probs, pred_heat = self(pc)

        loss = self.criterion(probs, pred_heat, heat)
        self.log('loss', loss, batch_size=pc.size(0))
        self.log('lr', self.optimizers().param_groups[0]['lr'])

        return loss

    def validation_step(self, batch, _):
        pc, heat, _ = batch

        probs, pred_heat = self(pc)

        loss = self.criterion(probs, pred_heat, heat)
        self.log('val_loss', loss, True, batch_size=pc.size(0))

    def test_step(self, batch, _):
        pc, heat, _ = batch

        probs, pred_heat = self(pc)

        loss = self.criterion(probs, pred_heat, heat)
        self.log('test_loss', loss, True, batch_size=pc.size(0))

    def configure_optimizers(self):
        args = self.hparams.args
        optimizer = torch.optim.Adam(self.net.parameters(), args.lr_max, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, float(args.lr_max),
                                                        pct_start=args.pct_start, div_factor=float(args.div_factor),
                                                        final_div_factor=float(args.final_div_factor),
                                                        epochs=args.max_epochs,
                                                        steps_per_epoch=len(self.train_dataloader()))
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def train_dataloader(self):
        args = self.hparams.args
        return DataLoader(KPS_Geodesic_Dataset(args, args.train_file, True),
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=args.train_workers,
                          pin_memory=True)

    def val_dataloader(self):
        args = self.hparams.args
        return DataLoader(KPS_Geodesic_Dataset(args, args.val_file, False),
                          batch_size=args.batch_size,
                          shuffle=False,
                          num_workers=args.val_workers,
                          pin_memory=True)

    def test_dataloader(self):
        args = self.hparams.args
        return DataLoader(KPS_Geodesic_Dataset(args, args.test_file, False),
                          batch_size=args.batch_size,
                          shuffle=False,
                          num_workers=args.test_workers,
                          pin_memory=True)


class LitModelInference(LitModel):
    def forward(self, x):
        return torch.argmax(self.net(x), dim=2)
