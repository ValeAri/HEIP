import argparse
import pytorch_lightning as pl
from typing import Dict, Any

from cellseg_models_pytorch.datasets import SegmentationHDF5Dataset
from cellseg_models_pytorch.datamodules.custom_datamodule import CustomDataModule
from cellseg_models_pytorch.training.lit import SegmentationExperiment

from cellseg_models_pytorch.training.callbacks.wandb_callbacks import (
    WandbGetExamplesCallback,
    WandbClassLineCallback,
)

from src.unet import get_seg_model


def train(args: Dict[str, Any]) -> None:
    """Train a model with the in-built tools."""
    pl.seed_everything(args.seed)

    # Set the datasets
    train_ds = SegmentationHDF5Dataset(
        path=args.train_ds,
        img_transforms=args.img_tr.split(","),
        inst_transforms=args.inst_tr.split(","),
        normalization=args.norm,
        retrun_binary=bool(args.ret_binary),
        return_inst=False,
        return_type=bool(args.ret_type),
        return_sem=bool(args.ret_sem),
    )
    valid_ds = SegmentationHDF5Dataset(
        path=args.valid_ds,
        img_transforms=[],
        inst_transforms=args.inst_tr.split(","),
        normalization=args.norm,
        retrun_binary=bool(args.ret_binary),
        return_inst=False,
        return_type=bool(args.ret_type),
        return_sem=bool(args.ret_sem),
    )
    test_ds = SegmentationHDF5Dataset(
        path=args.test_ds,
        img_transforms=[],
        inst_transforms=args.inst_tr.split(","),
        normalization=args.norm,
        retrun_binary=bool(args.ret_binary),
        return_inst=bool(args.ret_inst) or bool(args.run_test),
        return_type=bool(args.ret_type),
        return_sem=bool(args.ret_sem),
    )

    # Lightning datamodule
    datamodule = CustomDataModule(
        [train_ds, valid_ds, test_ds],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Set up segmentation experiment
    model = get_seg_model(args.depth, args.encoder)
    experiment = SegmentationExperiment.from_yaml(model, args.yaml_path)

    # Set up loggers
    exp_dir = args.exp_dir
    experiment_name = args.exp_name
    experiment_version = args.exp_version

    loggers = []
    if bool(args.use_wandb):
        loggers.append(
            pl.loggers.WandbLogger(
                save_dir=exp_dir,
                project=experiment_name,
                name=experiment_version,
                version=experiment_version,
                log_model=False,
            )
        )
    else:
        loggers.append(pl.loggers.TensorBoardLogger(save_dir=exp_dir))

    # Set up model callbacks
    callbacks = []
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        dirpath=exp_dir,
        save_top_k=1,
        save_last=True,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )
    ckpt_callback.CHECKPOINT_NAME_LAST = f"{experiment_version}_last"
    callbacks.append(ckpt_callback)

    if bool(args.use_wandb):
        cell_types = None
        if args.classes_type is not None:
            cell_types = {i: c for i, c in enumerate(args.classes_type.split(","))}

        tissue_types = None
        if args.classes_tissue is not None:
            tissue_types = {i: c for i, c in enumerate(args.classes_tissue.split(","))}

        # callbacks.append(WandbImageCallback(cell_types, tissue_types))
        if bool(args.class_metrics):
            callbacks.append(WandbClassLineCallback(cell_types, tissue_types))

        if bool(args.run_test):
            callbacks.append(
                WandbGetExamplesCallback(
                    cell_types,
                    tissue_types,
                    instance_postproc=model.aux_key,
                    inst_key=model.inst_key,
                    aux_key=model.aux_key,
                    inst_act="softmax",
                    aux_act=None,
                )
            )

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.n_devices,
        strategy=args.strategy,
        max_epochs=args.n_epochs,
        logger=loggers,
        callbacks=callbacks,
        profiler="simple",
        move_metrics_to_cpu=bool(args.move_metrics_to_cpu),
        log_every_n_steps=100,
    )

    trainer.fit(model=experiment, ckpt_path=args.ckpt_path, datamodule=datamodule)

    if bool(args.run_test):
        trainer.test(model=experiment, ckpt_path=args.ckpt_path, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--yaml_path",
        type=str,
        required=True,
        metavar="YAML",
        help="The path to the segmentation experiment yaml config file.",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        required=True,
        metavar="ENCODER",
        help="The name of the timm encoder. See timm docs for more info.",
    )
    parser.add_argument(
        "--depth",
        type=str,
        required=True,
        metavar="DEPTH",
        help="The depth of the encoder.",
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        required=True,
        metavar="EXPDIR",
        help="The path to the experiment dir.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        metavar="EXPNAME",
        help="The experiment name.",
    )
    parser.add_argument(
        "--exp_version",
        type=str,
        default=None,
        metavar="EXPVER",
        help="The experiment version.",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        metavar="ACC",
        help="The lit accelerator arg.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        metavar="STRAT",
        help="The lit strategy arg.",
    )
    parser.add_argument(
        "--n_devices",
        type=int,
        default=1,
        metavar="ND",
        help="Number of cpus/gpus to use for training the model.",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        required=True,
        metavar="NE",
        help="Number of epochs to train.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        required=True,
        metavar="NW",
        help="Number of workers for dataloaders.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        metavar="BS",
        help="The batch size.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        metavar="CKPT",
        help="Path to the checkpoint file. Optional",
    )
    parser.add_argument(
        "--train_ds",
        type=str,
        required=True,
        metavar="TRAINDS",
        help="The path to the training hdf5 dataset.",
    )
    parser.add_argument(
        "--valid_ds",
        type=str,
        required=True,
        metavar="VALIDDS",
        help="The path to the validation hdf5 dataset.",
    )
    parser.add_argument(
        "--test_ds",
        type=str,
        required=True,
        metavar="TESTDS",
        help="The path to the testing hdf5 dataset.",
    )
    parser.add_argument(
        "--inst_tr",
        type=str,
        required=True,
        metavar="INSTTR",
        help="The inst map tranformations. Comma separated list, NO SPACES!",
    )
    parser.add_argument(
        "--img_tr",
        type=str,
        required=True,
        metavar="IMTR",
        help="The img tranformations. Comma separated list, NO SPACES!",
    )
    parser.add_argument(
        "--norm",
        type=str,
        required=True,
        metavar="NORM",
        help="The normalization method.",
    )
    parser.add_argument(
        "--ret_inst",
        type=int,
        default=0,
        metavar="RETINST",
        help="Return the inst map during dataloading.",
    )
    parser.add_argument(
        "--ret_binary",
        type=int,
        default=1,
        metavar="RETBIN",
        help="Return the Binary inst map during dataloading.",
    )
    parser.add_argument(
        "--ret_type",
        type=int,
        default=1,
        metavar="RETTYPE",
        help="Return the type map during dataloading.",
    )
    parser.add_argument(
        "--ret_sem",
        type=int,
        default=1,
        metavar="RETSEM",
        help="Return the sem map during dataloading.",
    )
    parser.add_argument(
        "--use_wandb",
        type=int,
        default=1,
        metavar="WANDB",
        help="Use a wandb logger.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="SEED",
        help="random seed (default: 42)",
    )
    parser.add_argument(
        "--classes_type",
        type=str,
        default=None,
        metavar="TYPECLS",
        help=(
            "comma separated list of the cell type classes. ",
            "Has to be in order and include bg class",
        ),
    )
    parser.add_argument(
        "--classes_tissue",
        type=str,
        default=None,
        metavar="TISSCLS",
        help=(
            "comma separated list of the tissue type classes. ",
            "Has to be in order and include bg class",
        ),
    )
    parser.add_argument(
        "--class_metrics",
        type=int,
        default=1,
        metavar="CLSMTR",
        help="Log per class metrics to wandb. Works with 1 gpu only.",
    )
    parser.add_argument(
        "--move_metrics_to_cpu",
        type=int,
        default=1,
        metavar="METRCPU",
        help="Move metrics to cpu.",
    )
    parser.add_argument(
        "--run_test",
        type=int,
        default=0,
        metavar="TESTRUN",
        help="Run test metrics on test dataset.",
    )

    args = parser.parse_args()
    train(args)
