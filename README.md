# Predicting-dataset-size
Predicting dataset size for neural network fine-tuning with a given quality in object detection task. <br><br>
In this work, we propose the approach to make it possible to predict the size of the training set for detection network fine-tuning with the desired quality. The key feature of this approach is the ability to predict the scaling behavior of a given neural network on a certain data set using a small (no more than 6) number of images. <br><br>
We compared our model with NSL (Neural Scaling Laws). Here you can see the results on 4 images for detection of skin imperfections, surgical masks, and metal surface defects. <br><br>
![results](./images/results.png)
<br><br>

## Requirements
python >= 3.10 <br>
Run manually <code>pip install -r requirements.txt</code> if you have problems with automatically venv settings.

## Usage
Run GUI version with <code>python main.py</code> command. <br>
Add your directories with images and labels and enter the target mAP value. Press <code>Calculate result</code> button for getting the answer. <br>
Please upload labels in [Ultralytics YOLO](https://docs.ultralytics.com/datasets/detect/) standard for single class. Each lines in labels should be in format <code>0 x_center y_center w h</code>. <br><br>
![example](./images/usage_example.png)
<br><br>
Terms of use can be found in the section <code>File -> Info</code> in the upper left corner of the application window. <br>
