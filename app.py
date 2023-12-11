import configuration
from libs.foxutils.utils import core_utils
import streamlit as st
from os.path import join as pathjoin
from tools import defect_detection as def_det
from tools import object_detection as obj_det
from tools import measurement_tool as mes_tool
from tools import ocr_tool as ocr_tool
import logging

logger = logging.getLogger("app")

device = core_utils.device

settings = core_utils.settings
DATASET_NAME = settings['DIRECTORY']['dataset_name']
NORMAL_CLASS_DIR = settings['DIRECTORY']['normal_class_dir']
ABNORMAL_CLASS_DIR = settings['DIRECTORY']['abnormal_class_dir']
TEST_CLASS_DIR = settings['DIRECTORY']['test_class_dir']
DATASETS_DIR = pathjoin(settings['DIRECTORY']['datasets_dir'], DATASET_NAME)


def run_module():
    if st.session_state.task == "defect_detection":
        def_det.setup_defect_detection_demo(run_model_placeholder)
    elif st.session_state.task == "object_detection":
        obj_det.setup_object_detection_demo(run_model_placeholder)
    elif st.session_state.task == "ocr_tool":
        ocr_tool.setup_ocr_tool_demo(run_model_placeholder)
    elif st.session_state.task == "measurement_tool":
        mes_tool.setup_measurement_tool_demo(run_model_placeholder)
    else:
        pass


if "task" not in st.session_state:
    st.session_state["task"] = None

if "changed_dataset" not in st.session_state:
    st.session_state["changed_dataset"] = True

if "dataset" not in st.session_state:
    st.session_state["dataset"] = None

st.set_page_config(
    page_title=configuration.demo_title, page_icon=pathjoin('assets', 'favicon.ico'),
    layout="centered", initial_sidebar_state="expanded"
)

run_model_placeholder = st.empty()

with st.sidebar:
    left_co, cent_co, last_co = st.columns(3)
    with cent_co:
        st.image(configuration.demo_logo, width=100)

    st.markdown(configuration.demo_explanation)

    if st.button(configuration.modules["defect_detection"], key="btn_defect_detection", type="primary",
                 use_container_width=True):
        st.session_state["task"] = "defect_detection"

    if st.button(configuration.modules["object_detection"], key="btn_object_detection", type="primary",
                 use_container_width=True):
        st.session_state["task"] = "object_detection"
        pass

    if st.button(configuration.modules["ocr_tool"], key="btn_ocr", type="primary", use_container_width=True):
        st.session_state["task"] = "ocr_tool"
        pass

    if st.button(configuration.modules["measurement_tool"], key="btn_measurement_tool", type="primary",
                 use_container_width=True):
        st.session_state["task"] = "measurement_tool"
        pass

    run_module()
