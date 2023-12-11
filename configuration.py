from os.path import join as pathjoin

demo_title = "Iris Auto AI Vision Demo"

demo_logo = pathjoin("assets", "logo.png")

demo_explanation = """
            ## Iris Auto AI Vision Technology
            #### Customize an AI assistant model
            Give some example annotated images to the model and train it to run for your specific problem.
            
            Find out more in [iris-auto.com](https://iris-auto.com/)
            """

modules = {"defect_detection": "Defect Detection",
           "object_detection": "Object Detection",
           "ocr_tool": "Optical Character Recognition",
           "measurement_tool": "Measurement Tool"
           }

find_out_more = "Find out more in [iris-auto.com](https://iris-auto.com/)"

data_attribution = {
    "mvtec": "Example datasets provided by [MVTec](https://www.mvtec.com/company/research/datasets/mvtec-ad).",
}


def_det_instructions = {
    "explanation": "Identifying image imperfections, ensuring products meet the highest quality standards. With "
                   "sophisticated image analysis, even the most minute discrepancies are detected, reducing waste "
                   "and enhancing product integrity.",
    "result_explanation": "The software is able to detect scratches, broken parts and color variations on a fast "
                          "moving assembly line",
    "train": """
            ### Train the model
            Give some example images so that the model will know what kind of artifacts it should look for.
            
            Some example images will appear here.
            """,
    "evaluate": """
                ### Evaluate the model
                After the model is trained, we can pass new test images through the model.
                """,
    "datasets": ["nuts", "screw"],
}

obj_det_instructions = {
    "explanation": "Recognizing and tracking various items in real-time, regardless of the complexity of the "
                   "production line. This function ensures accurate sorting, counting and control in high-paced "
                   "manufacturing environments.",
    "result_explanation": "The software is able to detect, localise and classify a large variety of objects of "
                          "different scale at once.",
    "train": """
            ### Train the model
            Give some example images with bounding boxes so that the model will know what to look for.

            Some example images will appear here.
            """,
    "evaluate": """
                ### Evaluate the model
                After the model is trained, we can pass new test images through the model.
                """,
    "datasets": ["cars", "pills", "suitcase", "toothbrush"],
}

mes_instructions = {
    "explanation": "Quantifying dimensions to the finest degree. This feature guarantees that all parts and products "
                   "fit perfectly to precise design specifications.",
    "result_explanation": "The software is able to calculate dimensions such as height, width and radius with up to "
                          "98% accuracy.",
    "train": """
            ### Train the model
            """,
    "evaluate": """
                ### Evaluate the model
                The results will appear here.
                """,
    "datasets": ["measurement"],
}

ocr_instructions = {
    "explanation": "Transforming visual data into actionable text, enabling seamless data capture, traceability and "
                   "streamlined inventory management. An essential tool for modern industries requiring precise label "
                   "and character verification.",
    "result_explanation": "The object is able to extract text from images and visual data.",
    "train": """
            ### Train the model
            """,
    "evaluate": """
                ### Evaluate the model
                The results will appear here.
                """,
    "datasets": ["ocr"],
}

instructions = {
    "defect_detection": def_det_instructions,
    "object_detection": obj_det_instructions,
    "ocr_tool": ocr_instructions,
    "measurement_tool": mes_instructions,
}

primary_color = "blue"
