---
title: IRIS  Fault-detection-demo
emoji: 👁
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 3.46.0
app_file: app.py
pinned: false
license: mit
---

# IRIS fault detection Demo 

- The model is running *LIVE* at [Huggingface](https://huggingface.co/spaces/Interactive-Coventry/IRIS__fault-detection-demo).

- Base repo: [Github | Interactive-Coventry/IRIS__fault-detection-demo](https://github.com/Interactive-Coventry/IRIS__fault-detection-demo/)

  - Run in a terminal with `python app.py` and open `http://localhost:7861` in your browser.
  
  - Alternatively, run with `gradio app.py` to start a dev server with hot reloading enabled.

![Demo1](/assets/demo_1.jpg)

![Demo2](/assets/demo_2.jpg)


## Customize a fault detection model
- Add a few images of your own and see how the model performs

                
**Note:** This demo runs on CPU with num_workers=0, therefore is slow. 
For demonstration purposes, the model is trained only for 1 epoch.
If you want to change run configuration, update the config.ini file accordingly.
The model is not properly pre-trained, it is just built with pretrained weights. 
Segmentation is not yet implemented. 

- Give some normal images. 
Prepare a directory with two folders: 
one folder 'normal' with normal images and one folder 'anomaly' with faulty images.
(e.g.{DATASETS_DIR})

**Note:** For demo purposes this folder is not editable. If you want to use different input data, 
then add files in the 'data/' folder and edit the config.ini file accordingly.

**Note:** If the screen changes size and you can't go up, resize the page using Ctrl + '+' or with the magnifying lens. 

- Give a test image. 
Download a random image from 
[https://huggingface.co/spaces/Interactive-Coventry/IRIS__fault-detection-demo/tree/main/data/nuts/test](https://huggingface.co/spaces/Interactive-Coventry/IRIS__fault-detection-demo/tree/main/data/nuts/test)
and upload it to the 'Test Input Image' box. 
