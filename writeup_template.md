# Vehicle Detection

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

### Histogram of Oriented Gradients (HOG)

#### 1. how I extracted HOG features from the training images.

The code for this step is contained in `1_exploration.ipynb` .  

1. Data Exploration

I started by reading in all the `vehicle` and `non-vehicle` images.  

dataset statistics:

```
- n_car 8792
- n_notcars 8968
- image_shape (64, 64, 3)
- data_type float32
```

example image:

<img src="./for_writeup/example-images.png" width="600">

2. Data Exploration

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `HSV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

<img src="./for_writeup/hog-features.png" width="1000">

#### 2. how I settled on my final choice of HOG parameters.

the process is the following ..

1. choose some combinations of parameters
2. extract HOG features
3. train classifier using this features
4. checking prediction accuracy of test-dataset

I have choosed the following parameters which produce the best accuracy.

```
colorspace = 'HSV'
channels = 3 (ALL of HSV)
orient = 9
pix_per_cell = 8
cell_per_block = 2
```

#### 3. how I trained a classifier.

I trained a linear SVM(C=1.0).

I used both of the HOG features & color features.

- HOG features (I have described)
- Color bin features (cspace='RGB', resize to (32,32,3))
- Color histogram (cspace='RGB', spatial_size=(32,32), hist_bins=32)

the train-dataset shape is `(17760, 8460)`

I got the clasiffier which predict test-dataset with accuracy ~ 0.98-0.99

### Sliding Window Search

#### 1. how I implemented a sliding window search. (scales to search, how much to overlap windows)

the process is the following ..

1. sliding window with multiple scape and overlap
2. apply classifier to each extracted image
3. save window position if `prediction == "Car"`
4. create heat map of window
5. drop window if `the value of each position of the heat map` is lower than `threshold`
6. combine window position

![alt text][image3]

#### 2. examples of test images to demonstrate how my pipeline is working.

I dropped window if `the value of each position of the heat map` is lower than `threshold`  to improve prediction performance

![alt text][image4]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues in my implementation, where my pipeline likely fail, what I can do to make it more robust.

* issue: pipeline is slow

* difficult case: if it's night, my pipeline may fail to detect vehicles

* other technique to make it more robust: use Convolutional Neural Network
for Object Detection (Faster RCNN, SSD, YOLO, ..)
