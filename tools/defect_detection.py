import time
from os import listdir
import configuration
from libs.foxutils.utils import core_utils
import streamlit as st
from os.path import join as pathjoin
from PIL import Image

import logging
logger = logging.getLogger("defect_detection")

IMAGE_VIEW_SIZE = (200, 200)


def show_example_images(placeholder):
    choice = st.session_state["dataset"]
    target_dirs = listdir(pathjoin(core_utils.datasets_dir, choice, "test"))
    target_dir = pathjoin(pathjoin(core_utils.datasets_dir, choice, "test", target_dirs[0]))
    target_file = pathjoin(target_dir, listdir(target_dir)[0])
    target_img = Image.open(target_file).resize(size=IMAGE_VIEW_SIZE)
    placeholder.image(target_img, width=200)


def update_show_example_images(placeholder):
    if st.session_state["changed_dataset"]:
        st.session_state["dataset"] = st.session_state.radio_dataset
        show_example_images(placeholder)
        st.session_state["changed_dataset"] = False
    else:
        show_example_images(placeholder)


def get_example_images(choice):
    target_dirs = listdir(pathjoin(core_utils.datasets_dir, choice, "test"))
    img_list = []
    for x in target_dirs[0:2]:
        target_dir = pathjoin(pathjoin(core_utils.datasets_dir, choice, "test", x))
        for y in listdir(target_dir):
            target_file = pathjoin(target_dir, y)
            target_img = Image.open(target_file).resize(size=IMAGE_VIEW_SIZE)
            img_list.append(target_img)

            target_file = pathjoin(target_dir.replace("test", "ground_truth"), y)
            target_img = Image.open(target_file.replace(".png", "_mask.png")).resize(size=IMAGE_VIEW_SIZE)
            img_list.append(target_img)
            if len(img_list) == 6:
                break
    return img_list


def get_result_images(choice):
    target_dirs = listdir(pathjoin(core_utils.datasets_dir, choice, "test"))
    img_list = []
    for x in target_dirs[-2:]:
        target_dir = pathjoin(pathjoin(core_utils.datasets_dir, choice, "test", x))
        for y in listdir(target_dir):
            target_file = pathjoin(target_dir, y)
            target_img = Image.open(target_file).resize(size=IMAGE_VIEW_SIZE)

            target_file = pathjoin(target_dir.replace("test", "ground_truth"), y)
            overlay_img = Image.open(target_file.replace(".png", "_mask.png")).resize(size=IMAGE_VIEW_SIZE)
            overlay_img.putalpha(128)
            target_img.paste(overlay_img, mask=overlay_img)

            img_list.append(target_img)
            if len(img_list) == 3:
                break

    return img_list


def run_defect_detection_demo(task_, placeholder):
    choice = st.session_state["dataset"]

    with placeholder:
        st.markdown(configuration.instructions[task_]["train"])

        input_gallery_imgs = get_example_images(choice)
        st.markdown("##### :orange[Training Images]")
        st.image(input_gallery_imgs)

        st.markdown(configuration.instructions[task_]["evaluate"])

        progress_bar = st.progress(value=0, text="Starting...")
        time.sleep(0.5)
        for percent_complete in range(1, 100):
            time.sleep(0.05)
            if (percent_complete % 10) == 0:
                progress_bar.progress(percent_complete + 1, text=f"Currently at {percent_complete}%")
        progress_bar.progress(100, text=f"Finished")

        result_gallery_imgs = get_result_images(choice)
        st.markdown("##### :orange[Result Images]")
        st.image(result_gallery_imgs)

        st.markdown(configuration.find_out_more)


def clicked_changed_dataset():
    st.session_state["changed_dataset"] = True


def setup_defect_detection_demo(placeholder):
    task_ = st.session_state["task"]

    st.session_state["dataset"] = configuration.instructions[task_]["datasets"][0]
    st.session_state["changed_dataset"] = True

    with placeholder.container():
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(":orange[Select a target dataset]")
            available_datasets = configuration.instructions[task_]["datasets"]
            st.radio(options=available_datasets, label="Dataset", index=0, horizontal=True,
                     on_change=clicked_changed_dataset, key="radio_dataset")
            run_btn = st.button(label="See how it works!", key="btn_run")

        with col2:
            st.markdown(":orange[Example image for the selected dataset]")
            dataset_img_placeholder = st.empty()
            st.markdown(configuration.data_attribution["mvtec"])
            update_show_example_images(dataset_img_placeholder)

        results_placeholder = st.container()
        if run_btn:
            run_defect_detection_demo(task_, results_placeholder)