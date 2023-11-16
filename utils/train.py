import os
from os.path import join as pathjoin

import numpy as np
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

import libs.foxutils.utils.core_utils as core_utils
from libs.anomalib.config import get_configurable_parameters
from libs.anomalib.data import get_datamodule
from libs.anomalib.models import get_model
from libs.anomalib.utils.callbacks import get_callbacks, LoadModelCallback
from utils import anomaly_detection as ad

MODELS_DIR = core_utils.settings['DIRECTORY']['models_dir']
DEFAULT_ANOMALIB_MODELS_DIR = pathjoin('libs', 'anomalib', 'models')
ANOMALY_DETECTION_CHECKPOINT_FILE = pathjoin(MODELS_DIR,
                                             core_utils.settings['MODELS']['anomaly_detection_checkpoint_file'])
ANOMALY_CONFIG_PATH = pathjoin(MODELS_DIR, core_utils.settings['MODELS']['anomaly_detection_config_file'])
ANOMALY_DETECTION_MODEL = core_utils.settings['MODELS']['anomaly_detection_model']

DATASET_NAME = core_utils.settings['DIRECTORY']['dataset_name']
EXP_NAME = core_utils.settings['TRAINING']['exp_name']
VERSION = int(core_utils.settings['TRAINING']['version'])
MAX_EPOCHS = int(core_utils.settings['TRAINING']['epochs'])
WEIGHTS_DIR = core_utils.settings['TRAINING']['weights_dir']
BATCH_SIZE = int(core_utils.settings['TRAINING']['batch_size'])
IM_HEIGHT = int(core_utils.settings['TRAINING']['im_height'])
IM_WIDTH = int(core_utils.settings['TRAINING']['im_width'])
PRETRAINED = bool(eval(core_utils.settings['TRAINING']['pretrained']))
INIT_WEIGHTS = core_utils.settings['TRAINING'][
    'init_weights']  # imagenet_V1: wide_resnet50_2-95faca4d.pth, imagenetv2: wide_resnet50_2-9ba9bcbe.pth
SEED = int(core_utils.settings['TRAINING']['seed'])


def get_lightning_log_dir(model_name):
    lightning_log_dir = pathjoin(core_utils.models_dir, model_name)
    return lightning_log_dir


def train_model(task, dataset_dir, exp_name=EXP_NAME):
    print(f'Training {ANOMALY_DETECTION_MODEL} for task {task} during experiment {exp_name}, '
          f'using directory {dataset_dir}.\n')
    model_fullname = pathjoin(ANOMALY_DETECTION_MODEL, exp_name)
    project_path = pathjoin(MODELS_DIR, task)
    core_utils.mkdir_if_not_exist(project_path)

    model_config_path = pathjoin(project_path, model_fullname, ANOMALY_DETECTION_MODEL + '.yaml')
    core_utils.mkdir_if_not_exist(model_config_path)

    num_files = len([x for x in os.listdir(dataset_dir) if '.jpg' in x])
    log_steps = max(int(np.ceil(num_files / BATCH_SIZE)), 1)

    new_update = {
        'dataset': {'path': dataset_dir, 'root': dataset_dir, 'name': exp_name,
                    'format': 'folder', 'category': DATASET_NAME,
                    'task': task, 'image_size': IM_HEIGHT, 'train_batch_size': BATCH_SIZE,
                    'test_batch_size': BATCH_SIZE, 'num_workers': 0,
                    'normal_dir': 'normal', 'abnormal_dir': 'anomaly',
                    'normal_test_dir': None,
                    'test_split_mode': 'from_dir', 'test_split_ratio': 0.2,
                    'val_split_mode': 'same_as_test',
                    'mask_dir': None, 'extensions': None},
        # 'init_weights': pathjoin(weights_dir, init_weights),
        'metrics': {'image': ['F1Score', 'AUROC'],
                    'pixel': None},
        'project': {'path': project_path, 'seed': SEED, 'unique_dir': True},
        'model': {'early_stopping': {'metric': 'train_loss',
                                     # `train_loss`, `train_loss_step`, `image_F1Score`,
                                     # `image_AUROC`, `train_loss_epoch`
                                     'mode': 'min',
                                     'patience': 3},
                  'pre_trained': PRETRAINED},
        'trainer': {'accelerator': 'cpu', 'devices': 1, 'max_epochs': MAX_EPOCHS, 'enable_model_summary': True,
                    'num_sanity_val_steps': 2, 'gradient_clip_val': 0.1, 'val_check_interval': 1.0,
                    'check_val_every_n_epoch': 1, 'log_every_n_steps': log_steps},
        'optimization': {'export_mode': 'torch'},
        'visualization': {'log_images': True, 'mode': 'full', 'save_images': True, 'show_images': False},
    }

    ad.update_yaml(ad.MODEL_CONFIG_PAIRS[ANOMALY_DETECTION_MODEL], model_config_path, new_update)

    with open(model_config_path) as f:
        updated_config = yaml.safe_load(f)
    print(f'New config path saved in {model_config_path}')

    if updated_config['project']['seed'] != 0:
        seed_everything(updated_config['project']['seed'])

    config = get_configurable_parameters(
        model_name=updated_config['model']['name'],
        config_path=model_config_path
    )

    # pass the config file to model, logger, callbacks and datamodule
    model = get_model(config).to(core_utils.device)

    lightning_log_dir = get_lightning_log_dir( pathjoin(task, model_fullname))
    print(f'Lightning logs at {lightning_log_dir}')
    experiment_logger = TensorBoardLogger(lightning_log_dir)

    # experiment_logger = get_experiment_logger(config)
    callbacks = get_callbacks(config)
    datamodule = get_datamodule(config)

    datamodule.setup()

    print(f'Train data: {len(datamodule.train_data)}')
    print(f'Val data: {len(datamodule.val_data)}')
    print(f'Test data: {len(datamodule.test_data)}')

    # start training
    trainer = Trainer(**config.trainer,
                      logger=experiment_logger,
                      callbacks=callbacks)
    trainer.fit(model=model,
                datamodule=datamodule,
                )

    print(f'Training callback metrics: {trainer.callback_metrics}')

    # load best model from checkpoint before evaluating
    load_model_callback = LoadModelCallback(
        weights_path=trainer.checkpoint_callback.best_model_path
    )
    print(f'Best model path {trainer.checkpoint_callback.best_model_path}')

    trainer.callbacks.insert(0, load_model_callback)
    trainer.test(model=model, datamodule=datamodule)

    run_dir = [x for x in os.listdir(pathjoin(project_path, model_fullname)) if 'run.' in x][-1]
    image_dir = pathjoin(project_path, model_fullname, run_dir, 'images')
    print(f'Reading files after training from {image_dir}.')
    img_list_normal, img_list_anomaly = ad.show_validation_results(image_dir, n_images=5)
    print('Finished training and testing.\n')

    return img_list_normal, img_list_anomaly
