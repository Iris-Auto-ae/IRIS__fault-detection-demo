from libs.foxutils.utils import core_utils
from libs.foxutils.gradio import gradio_utils
from utils.app_functions import customize_model, evaluate_model
import gradio as gr
from os.path import join as pathjoin

device = core_utils.device

settings = core_utils.settings
DATASET_NAME = settings['DIRECTORY']['dataset_name']
NORMAL_CLASS_DIR = settings['DIRECTORY']['normal_class_dir']
ABNORMAL_CLASS_DIR = settings['DIRECTORY']['abnormal_class_dir']
TEST_CLASS_DIR = settings['DIRECTORY']['test_class_dir']
DATASETS_DIR = pathjoin(settings['DIRECTORY']['datasets_dir'], DATASET_NAME)
IS_TKINTER_DISABLED = bool(eval(settings['TKINTER']['disabled']))

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Customize a fault detection model
        Add a few images of your own and see how the model performs
        
                        
        **Note:** This demo runs on CPU with num_workers=0, therefore is slow. 
        For demonstration purposes, the model is trained only for 1 epoch.
        If you want to change run configuration, update the config.ini file accordingly.
        The model is not properly pre-trained, it is just built with pretrained weights. 
        Segmentation is not yet implemented. 
        """)
    with gr.Row():
        with gr.Column(scale=1, min_width=600):
            gr.Markdown(
                f"""
                ## Give some normal images. 
                Prepare a directory with two folders: 
                one folder 'normal' with normal images and one folder 'anomaly' with faulty images.
                (e.g.{DATASETS_DIR})
                
                **Note:** For demo purposes this folder is not editable. If you want to use different input data, 
                then add files in the 'data/' folder and edit the config.ini file accordingly.
                
                **Note:** If the screen changes size and you can't go up, resize the page using Ctrl + '+' or with the
                 magnifying lens. 

                """)

            if IS_TKINTER_DISABLED:
                fixed_filedir = pathjoin(DATASETS_DIR)
                file_dir = gr.Textbox(label="Target Directory", value=fixed_filedir,
                                      info="Select directory from where examples will be fetched.",
                                      interactive=False,
                                      )
            else:
                file_dir, directory_button = gradio_utils.get_directory_textbox(label="Target Directory",
                                                                                info="Select directory from where examples will be fetched.",
                                                                                button_text="Select directory",
                                                                                default_dir=DATASETS_DIR)
            task_selection_button = gr.Radio(choices=["Classification", "Segmentation"], value="Classification",
                                             label="Target Task", info="Select the task you want to perform.")
            with gr.Row():
                train_button = gr.Button("Train", size="sm", variant="primary")

            gr.Markdown(
                """
                ##  Wait until the model is trained to see its test results.
                """)
            # Output components
            gr.Markdown("### Actual Normal Samples")
            glr1 = gr.Gallery(label="Actual Normal Samples", preview=True, scale=0)
            gr.Markdown("### Actual Faulty Samples")
            glr2 = gr.Gallery(label="Actual Faulty Samples", preview=True, scale=0)

        with gr.Column(scale=1, min_width=600):
            gr.Markdown(
                """
                ## Add a test image to see how the model performs. 
                
                Download a random image from 
                [https://huggingface.co/spaces/Interactive-Coventry/IRIS__fault-detection-demo/tree/main/data/nuts/test](https://huggingface.co/spaces/Interactive-Coventry/IRIS__fault-detection-demo/tree/main/data/nuts/test)
                and upload it to the 'Test Input Image' box. 
                """)

            test_input_file = gr.File(label="Test input image", info="Give an image for testing", scale=0)
            with gr.Row():
                clear_button = gr.ClearButton(size="sm")
                submit_button = gr.Button("Submit", size="sm", variant="primary")

            gr.Markdown("### Input image")
            test_input_img = gr.Image(label="Test input image", scale=0)
            gr.Markdown("### Model Result")
            test_result_img = gr.Image(label="Test result image", info="Here's the result of the model", scale=0)

    train_button.click(fn=customize_model, inputs=[task_selection_button, file_dir], outputs=[glr1, glr2])
    submit_button.click(fn=gradio_utils.view_uploaded_image, inputs=test_input_file, outputs=test_input_img)
    submit_button.click(fn=evaluate_model, inputs=[task_selection_button, test_input_file],
                        outputs=[test_result_img])

if __name__ == "__main__":
    # demo.launch(show_error=True, debug=True, server_port=7861, favicon_path=pathjoin('assets', 'favicon.ico'))
    demo.launch(show_error=True, favicon_path=pathjoin('assets', 'favicon.ico'))
