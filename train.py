import yaml
import argparse
import pytorch_lightning as pl

from pl_model import LitModel
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


def get_parser():
    parser = argparse.ArgumentParser(description='Tooth Landmark Detection')
    parser.add_argument("--config", type=str, default="config/keypoint_saliency.yaml", help="path to config file")
    parser.add_argument("--gpus", type=int, default=1)

    args_cfg = parser.parse_args()
    with open(args_cfg.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args_cfg, k, v)

    return args_cfg


if __name__ == "__main__":
    args = get_parser()
    pl.seed_everything(args.seed)

    model = LitModel(args)
    if args.load_from_checkpoint:
        model = LitModel.load_from_checkpoint(args.load_from_checkpoint)

    logger = TensorBoardLogger("runs", args.experiment)
    callback = ModelCheckpoint(monitor='val_loss', save_top_k=5, save_last=True, mode='min')

    debug = False
    debug_args = {'limit_train_batches': 10} if debug else {}
    trainer = pl.Trainer(logger, accelerator='gpu', devices=1, max_epochs=args.max_epochs, callbacks=[callback],
                         **debug_args)
    trainer.fit(model)

    results = trainer.test(ckpt_path='best')
    print(results)
