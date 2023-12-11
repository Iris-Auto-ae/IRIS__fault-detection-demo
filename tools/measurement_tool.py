import time
from os import listdir
import configuration
from libs.foxutils.utils import core_utils
import streamlit as st
from os.path import join as pathjoin
from PIL import Image

import logging
logger = logging.getLogger("measurement_tool")



def get_result_images(choice):
    img_list = []
    target_dir = pathjoin(pathjoin(core_utils.datasets_dir, choice))
    for y in listdir(pathjoin(target_dir, "test")):
        target_file = pathjoin(target_dir, "test", y)
        overlay_img = Image.open(target_file)
        img_list.append(overlay_img)
        if len(img_list) == 6:
            break

    return img_list


def run_measurement_tool_demo(task_, placeholder):

    with placeholder:
        st.markdown(configuration.instructions[task_]["evaluate"])

        progress_bar = st.progress(value=0, text="Starting...")
        time.sleep(0.2)
        for percent_complete in range(1, 100):
            time.sleep(0.01)
            if (percent_complete % 10) == 0:
                progress_bar.progress(percent_complete + 1, text=f"Currently at {percent_complete}%")
        progress_bar.progress(100, text=f"Finished")

        choice = configuration.instructions[task_]["datasets"][0]
        result_gallery_imgs = get_result_images(choice)
        st.markdown(f"##### :{configuration.primary_color}[Result Images]")
        st.markdown(configuration.instructions[task_]["result_explanation"])
        st.image(result_gallery_imgs)

        st.markdown(configuration.find_out_more)


def setup_measurement_tool_demo(placeholder):
    task_ = st.session_state["task"]

    with placeholder.container():
        st.title(configuration.modules[task_])
        st.markdown(configuration.instructions[task_]["explanation"])
        run_btn = st.button(label="See how it works!", key="btn_run")

        results_placeholder = st.container()
        if run_btn:
            run_measurement_tool_demo(task_, results_placeholder)
