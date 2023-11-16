import time
from os import listdir

from libs.foxutils.utils import core_utils
import streamlit as st
from os.path import join as pathjoin
from PIL import Image

device = core_utils.device

settings = core_utils.settings
DATASET_NAME = settings['DIRECTORY']['dataset_name']
NORMAL_CLASS_DIR = settings['DIRECTORY']['normal_class_dir']
ABNORMAL_CLASS_DIR = settings['DIRECTORY']['abnormal_class_dir']
TEST_CLASS_DIR = settings['DIRECTORY']['test_class_dir']
DATASETS_DIR = pathjoin(settings['DIRECTORY']['datasets_dir'], DATASET_NAME)

IMAGE_VIEW_SIZE = (200, 200)

def show_example_images(placeholder):
    choice = st.session_state["dataset"]
    if st.session_state["changed_dataset"]:
        target_dirs = listdir(pathjoin(core_utils.datasets_dir, choice, "test"))
        target_dir = pathjoin(pathjoin(core_utils.datasets_dir, choice, "test", target_dirs[0]))
        target_file = pathjoin(target_dir, listdir(target_dir)[0])
        target_img = Image.open(target_file).resize(size=IMAGE_VIEW_SIZE)
        placeholder.image(target_img, width=200)
        st.session_state["changed_dataset"] = False


def get_example_images(choice):
    target_dirs = listdir(pathjoin(core_utils.datasets_dir, choice, "test"))
    img_list = []
    for x in target_dirs[0:2]:
        target_dir = pathjoin(pathjoin(core_utils.datasets_dir, choice, "test", x))
        for y in  listdir(target_dir):
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


def run_demo(placeholder):
    choice = st.session_state["dataset"]

    with placeholder.container():
        st.markdown("""
                    ### Train the model
                    Give some example images so that the model will know what kind of artifacts it should look for.
                    
                    Some example images will appear here.
                    """)

        input_gallery_imgs = get_example_images(choice)
        st.markdown("##### :orange[Training Images]")
        st.image(input_gallery_imgs)

        st.markdown("""
            ### Evaluate the model
            After the model is trained, we can pass new test images through the model.
            
            The results will appear here.
            """)

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

        st.markdown("""
                    Find out more in [iris-auto.ase](https://iris-auto.ae/)
                    """)


st.set_page_config(
    page_title="Iris Auto Demo", page_icon=pathjoin('assets', 'favicon.ico'),
    layout="centered", initial_sidebar_state="collapsed"
)
st.markdown("""
            ## Iris Auto AI Technology
            #### Customize an AI assistant model
            Give some example annotated images to the model and *train* it to run for your specific problem.
            
            Find out more in [iris-auto.ase](https://iris-auto.ae/)
            """)
available_datasets = ["nuts", "screw"]

if "dataset" not in st.session_state:
    st.session_state["dataset"] = available_datasets[0]
    st.session_state["changed_dataset"] = False

col1, col2 = st.columns(2)
with col1:
    st.markdown(":orange[Select a target dataset]")
    dataset_btn = st.radio(options=available_datasets, label='', index=0, horizontal=True)
    run_btn = st.button(label="See how it works!", key="run")

    for val_d in available_datasets:
        if dataset_btn == val_d:
            st.session_state["dataset"] = val_d
            st.session_state["changed_dataset"] = True

with col2:
    st.markdown(":orange[Example image for the selected dataset]")
    dataset_img_placeholder = st.empty()
    st.markdown("Example datasets provided by [MVTec](https://www.mvtec.com/company/research/datasets/mvtec-ad).")

    show_example_images(dataset_img_placeholder)


run_model_placeholder = st.empty()
if run_btn:
    run_demo(run_model_placeholder)





