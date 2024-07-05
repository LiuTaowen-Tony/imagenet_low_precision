import argparse

import torch.utils
import torch.utils.data
import pytorch_lightning as pl
import torch
import qtorch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from timm import create_model
from timm.data import Mixup
from timm.data.transforms_factory import create_transform
import timm.scheduler
import wandb
from pytorch_lightning.loggers import WandbLogger
from low_precision_utils import metrics as lp_metrics
from low_precision_utils import utils as lp_utils

torch.set_float32_matmul_precision("medium")



# Argument parser for command-line options
def get_args():
    parser = argparse.ArgumentParser(description="Train a model on ImageNet")
    parser.add_argument('--train_res', type=int, default=160, help='Resolution for training images')
    parser.add_argument('--test_res', type=int, default=224, help='Resolution for test images')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training and validation')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.05, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warm-up epochs')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing')
    parser.add_argument('--mixup_alpha', type=float, default=0.1, help='Mixup alpha value')
    parser.add_argument('--cutmix_alpha', type=float, default=1.0, help='Cutmix alpha value')
    parser.add_argument('--test_crop_ratio', type=float, default=0.95, help='Crop ratio for test images')
    parser.add_argument("--act_man_width", type=int, default=1, help="Mantissa width for activation quantization")
    parser.add_argument("--weight_man_width", type=int, default=1, help="Mantissa width for activation quantization")
    parser.add_argument("--back_man_width", type=int, default=1, help="Mantissa width for activation quantization")
    parser.add_argument("--act_rounding", type=str, default="stochastic", help="Rounding mode for activation quantization")
    parser.add_argument("--weight_rounding", type=str, default="stochastic", help="Rounding mode for activation quantization")
    parser.add_argument("--back_rounding", type=str, default="stochastic", help="Rounding mode for activation quantization")
    parser.add_argument("--same_input", type=lambda x: (str(x).lower() == 'true'), default=True, help="Use the same input for all quantized layers")
    parser.add_argument("--same_weight", type=lambda x: (str(x).lower() == 'true'), default=True, help="Use the same output for all quantized layers")
    parser.add_argument("--back_input_man_width", type=int,default=1)
    parser.add_argument("--back_weight_man_width", type=int,default=1)
    parser.add_argument("--back_input_rounding", type=str,default="stochastic")
    parser.add_argument("--back_weight_rounding", type=str,default="stochastic")
    args =  parser.parse_args()
    args.learning_rate *= torch.cuda.device_count()
    return args
    # return parser.parse_args()


class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def setup(self, stage=None):
        train_transform = create_transform(
            input_size=self.args.train_res,
            is_training=True,
            auto_augment='rand-m6-n3-mstd0.5',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        test_transform = create_transform(
            input_size=self.args.test_res,
            crop_pct=self.args.test_crop_ratio,
            interpolation='bicubic',
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        self.train_dataset = ImageNet(root='./imagenet_data', split='train', transform=train_transform)
        # self.train_dataset = torch.utils.data.Subset(self.train_dataset, range(0, 2048))
        self.val_dataset = ImageNet(root='./imagenet_data', split='val', transform=test_transform)
        # self.val_dataset = torch.utils.data.Subset(self.val_dataset, range(0, 256))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

class LitModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters()

        model = create_model("resnet50", pretrained=False, num_classes=1000)
        build_number = lambda x: qtorch.FloatingPoint(8, x)
        quant_scheme = lp_utils.QuantScheme(
            fnumber=build_number(args.act_man_width),
            bnumber=build_number(args.back_man_width),
            wnumber=build_number(args.weight_man_width),
            fround_mode=args.act_rounding,
            bround_mode=args.back_rounding,
            wround_mode=args.weight_rounding,
            same_input=args.same_input,
            same_weight=args.same_weight,
            bfnumber=build_number(args.back_input_man_width),
            bwnumber=build_number(args.back_weight_man_width),
            bfround_mode=args.back_input_rounding,
            bwround_mode=args.back_weight_rounding
        )
        self.model = lp_utils.QuantWrapper(model, quant_scheme)
        print(self.model.module, flush=True)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.mixup_fn = Mixup(
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            label_smoothing=args.label_smoothing,
            num_classes=1000
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = self.mixup_fn(x, y)
        logits = self(x)
        loss = self.criterion(logits, y)
        # y is one hot
        acc = (torch.argmax(logits, dim=1) == torch.argmax(y, dim=1)).float().mean()
        lr = self.optimizers().param_groups[0]['lr']
        self.log("lr", lr, on_step=True, prog_bar=True)
        self.log("train_loss", loss, on_step=True)
        self.log("train_acc", acc, on_step=True)
        return loss

    def on_train_epoch_end(self, unused=None):
        grad = lp_metrics.grad_on_trainset(
            self.model,
            self.trainer.datamodule.train_dataset,
            self.args.batch_size,
            self.criterion
        )
        self.log_dict(grad)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.args.learning_rate, weight_decay=self.hparams.args.weight_decay, momentum=0.9, nesterov=True)
        scheduler = timm.scheduler.CosineLRScheduler(
            optimizer,
            t_initial=self.args.num_epochs, 
            lr_min=1e-5,
            warmup_lr_init=1e-3, 
            warmup_t=5,
            cycle_limit=1,
        )
        return [optimizer], [scheduler]

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.current_epoch) 

if __name__ == '__main__':
    args = get_args()
    data_module = ImageNetDataModule(args)
    model = LitModel(args)
    wandb_logger = WandbLogger(log_model=True, project="imagenet-training")
    wandb_logger.log_hyperparams(args)
    wandb_logger.watch(model)
    trainer = pl.Trainer(max_epochs=args.num_epochs, accelerator="cuda", logger=wandb_logger)
    trainer.fit(model, datamodule=data_module)
    wandb.finish()
