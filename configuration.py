demo_title = "Iris Auto AI Vision Demo"

demo_explanation = """
            ## Iris Auto AI Vision Technology
            #### Customize an AI assistant model
            Give some example annotated images to the model and *train* it to run for your specific problem.
            
            Find out more in [iris-auto.com](https://iris-auto.com/)
            """

modules = {"defect_detection": "Defect Detection",
           "object_detection": "Object Detection",
           "ocr": "Optical Character Recognition",
           "measurement_tool": "Measurement Tool"
           }

find_out_more = "Find out more in [iris-auto.com](https://iris-auto.com/)"

data_attribution = {
    "mvtec": "Example datasets provided by [MVTec](https://www.mvtec.com/company/research/datasets/mvtec-ad).",
}


def_det_instructions = {
    "train": """
            ### Train the model
            Give some example images so that the model will know what kind of artifacts it should look for.
            
            Some example images will appear here.
            """,
    "evaluate": """
                ### Evaluate the model
                After the model is trained, we can pass new test images through the model.
                
                The results will appear here.
                """,
    "datasets": ["nuts", "screw"],
}

obj_det_instructions = {
    "train": """
            ### Train the model
            Give some example images with bounding boxes so that the model will know what to look for.

            Some example images will appear here.
            """,
    "evaluate": """
                ### Evaluate the model
                After the model is trained, we can pass new test images through the model.

                The results will appear here.
                """,
    "datasets": ["cars", "pills", "suitcase", "toothbrush"],
}

instructions = {
    "defect_detection": def_det_instructions,
    "object_detection": obj_det_instructions,
    "ocr": {},
    "measurement_tool": {},
}