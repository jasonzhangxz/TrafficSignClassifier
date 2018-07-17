# **Traffic Sign Recognition** 

## Introduction

### This project implements the LeNet-5 CNN network to classify the [German Traffic Signs Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/barVisualization.png "barVisualization"
[image3]: ./examples/1.jpg "new image 1"
[image4]: ./examples/2.jpg "new image 2"
[image5]: ./examples/3.jpg "new image 3"
[image6]: ./examples/4.jpg "new image 4"
[image7]: ./examples/5.jpg "new image 5"
[image8]: ./examples/6.jpg "new image 6"
[image9]: ./examples/7.jpg "new image 7"
[image10]: ./examples/8.jpg "new image 8"
[image11]: ./examples/9.jpg "new image 9"
[image12]: ./examples/10.jpg "new image 10"
[image13]: ./examples/classified.png "classified"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ? Answer: 34799
* The size of the validation set is ? Answer: 4410
* The size of test set is ? Answer:12630
* The shape of a traffic sign image is ? Answer: (32,32,3)
* The number of unique classes/labels in the data set is ? Answer: 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. Also a bar chart showing how the data distributed, i.e. the counts for each different traffic signs.

![alt text][image1]
![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. 

As a first step, I decided to convert the images to grayscale because using grayscale will make it easier for the CNN to learn, still with high classification accuracy. Then I normalized the image data from [0,255] to [-1,1].

Then implemented the LeNet-5 from the CNN class with the following updates:
a.Changed the EPOCHS to 20
b.Changed the one_hot code to classify 43 types of traffic signs
c.Added dropout to fully connected layer to decrease the dependancy on the training data
e.Changed the activation method from relu to elu
There is no need to change the input layer, it still stays at (32,32,1) as I have already converted the image from rgb to grayscale.



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   							| 
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 28x28x6 	|
| ELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 2x2 stride,valid padding, outputs 10x10x16      									|
| ELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten	    |  outputs 400     									|
| Fully connected		| inputs 400, output 120        									|
| ELU					|												|
| Dropout					|	50% keep											|
| Fully connected		| inputs 120, output 84        									|
| ELU					|												|
| Dropout					|	50% keep											|
| Fully connected		| inputs 84, output 43        									|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used 
batch size: 128
Epoch: 20
Learning rate: 0.001
drop out: 50%

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.996
* validation set accuracy of 0.962
* test set accuracy of 0.938

In this project I chose to use the LeNet-5 we learned in the CNN class. I first simply implemented that one and found that it has the accuracy only up to 89%. Then I start to do some preprocess on the image data, like converting them to grayscale and normalize them. Updated the inputs and outputs of the LeNet-5 layers, found that the accuracy went up a little bit, still not good enough. Then I started to try different activation function and implementing the dropout on the fully connected layers. Also increased the EPOCHS to 20. Then the classifying accuracy goes up to around 96%. If I have more time, I would like also try to apply regulization on the input weights of different layers to see whether it can improve more. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 10 German traffic signs that I found on the web:

![alt text][image3] ![alt text][image4] ![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9] ![alt text][image10] ![alt text][image11] ![alt text][image12] 

I preprocess all these images, converting to grayscale, normalizing them to make them the same size to feed into the LeNet-5 network.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set 

Here are the results of the prediction:
[Yield               ] is predicted as [Yield]
[Priority road       ] is predicted as [Priority road]
[Children crossing   ] is predicted as [Children crossing]
[Road work           ] is predicted as [Road work]
[Speed limit (20km/h)] is predicted as [Speed limit (20km/h)]
[Slippery road       ] is predicted as [Slippery road]
[No entry            ] is predicted as [No entry]
[Keep right          ] is predicted as [Keep right]
[Speed limit (30km/h)] is predicted as [Speed limit (30km/h)]
[Yield               ] is predicted as [Yield]



The model was able to correctly predict all of the 10 new traffic signs. This network model still looks promising though this is a small test data set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

The code for making predictions on my final model is located in the 20th cell of the Ipython notebook.

![alt text][image13]


