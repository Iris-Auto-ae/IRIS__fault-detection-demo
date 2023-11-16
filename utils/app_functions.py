import os
from os.path import join as pathjoin

import gradio as gr
from PIL import Image
import utils.anomaly_detection as ad
from os.path import sep

from utils.train import train_model, EXP_NAME, ANOMALY_DETECTION_MODEL, MODELS_DIR
from libs.foxutils.utils import core_utils

results_dict = {}


def customize_model(task, dataset_dir):
    task = task.lower()
    print(f'Running for {task} task.')
    if task == 'classification':
        img_list_normal, img_list_anomaly = train_model(task, dataset_dir, EXP_NAME)
        return img_list_normal, img_list_anomaly
    else:
        gr.Error(f'Segmentation task not implemented yet.')
        return None, None


def split_filename_folder(filename):
    image_file = filename.split(sep)[-1]
    folder = filename.split(sep)[-2]
    filepath = sep.join(filename.split(sep)[:-1])
    return image_file, folder, filepath


def prepare_inference_trainer(task, anomaly_detection_model, exp_name, device='gpu'):
    model_fullname = pathjoin(task, anomaly_detection_model, exp_name)
    project_path = pathjoin(MODELS_DIR, model_fullname)
    run_dir = [x for x in os.listdir(project_path) if 'run.' in x][-1]
    config_path = pathjoin(project_path, run_dir, 'config_original.yaml')
    config_path_infer = pathjoin(project_path, run_dir, 'config_infer.yaml')
    checkpoint_file = pathjoin(project_path, run_dir, 'weights', 'lightning', 'model.ckpt')

    print(f'Loading trained model from {checkpoint_file}.')

    new_update = {
        'dataset': {'name': exp_name + '-test'},
    }
    ad.update_yaml(config_path, config_path_infer, new_update)

    model, trainer, config = ad.load_anomaly_detection_model(model_path=checkpoint_file, config_path=config_path_infer,
                                                             device=device)
    return model, trainer, config


def evaluate_model(task, filename):
    task = task.lower()
    filename = filename.name
    image_file, folder, filepath = split_filename_folder(filename)
    print(f'Reading image from {filename}.')
    img = Image.open(filename)
    orig_dim = img.size

    print('\n\n------------------Run anomaly detection------------------')
    model, trainer, config = prepare_inference_trainer(task, ANOMALY_DETECTION_MODEL, EXP_NAME, device=core_utils.device)
    results = ad.infer(model, trainer, config.dataset.image_size, filepath)
    result_image_path = pathjoin(config.project.path, ad.INFER_FOLDER_NAME, folder, image_file)
    anomaly_img = Image.open(result_image_path)
    print(f'Inferred image saved at {result_image_path}.')
    results_dict['anomaly_detection'] = ad.set_results_anomaly_detection(filename, anomaly_img, results[0],
                                                                         pathjoin(config.project.path, folder),
                                                                         orig_dim=orig_dim)

    return results_dict['anomaly_detection']['image']
