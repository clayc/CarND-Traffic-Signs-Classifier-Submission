# **Traffic Sign Recognition** 

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

Link to project repository is here: [project code](https://github.com/clayc/CarND-Traffic-Signs-Classifier)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Using numpy to examine the shape of the training, validation, and testing data arrays, I was able to create a summary of the dataset:
    
    * Number of training examples = 34799
    * Number of validation examples = 4410
    * Number of testing examples = 12630
    * Image data shape = (32, 32, 3)
    * Number of classes = 43
    
The above data shows that the validation set is 12.7% the size of the training set, and the testing set is 36.3% the size of the training set. 

#### 2. Include an exploratory visualization of the dataset.

In order to gain a better understanding of the dataset, two visualizations were created. The first is a set of histograms showing the count of each sign type in each of the datasets. This visualization helps give an understanding of how many of each sign type the model is exposed to, and also allows for quick visualization differences between the training, validation, and testing datasets. In the case of the German Traffic Sign data, each sign accounted for a similar porportion of the total in each of the three datasets.

![Dataset Histograms](dataset-histograms.png)

In addition to the histograms, a visualization was also created that shows a random image of each sign type from each of the three datasets. This visualization is useful for understanding the range of image quality, and also serves as a useful reference for the different sign types. Because the igns are chosen randomly, it can be useful to refresh the output to see a range of signs.

![Sign Reference Images](signs.png)


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

For this project, images were preprocessed in three steps. The overall goal of preprocessing was to emphasize the geometry of the shapes on the signs. The first step in preprocessing is to convert the image to greyscale. The decision to convert to greyscale was made because signs all use similar colors, therefore the shapes on the signs are more important than the colors. Additionally, there is a possibility that the perceived color of a sign could change depending on the color of the light it is reflecting. One advantageous side effect of converting to greyscale is that the images decrease from three channels to one, simplifying the input data.

The second step in preprocessing was to increase the contrast by adaptive histogram equalization. Specifically, the CLAHE function in OpenCV was applied to images. This function takes a block of the image and adjusts the values of each pixel proportiontely so that the full range of possible pixel values are used within that block. The process is repeated across the entire image. This process insures that even areas with very low initial contrast can be adjusted to higher contrast.

The final step in the preprocessing process is to normalize the images. The goal of normalization is to adjust pixel values from the input range (0-256), to a range between -1 and 1. The following formula was applied to each pixel to normalize the images:

$$Normalized = \frac{Input - 128}{128}$$

An example of an image shown during each step of preprocessing is shown below. Note that the images below do not include the normalization step because that step only scales and shifts the data, but does not change the image.

![Preprocessing Steps](preprocessing.png)

After the data was preprocessed using the above steps, the training dataset was augmented in order to provide the model with additional data and improve accuracy. Augmentation was performed by applying an Affine Transform to the existing training images. This transform distorts the image such that all parallel lines remain parallel. Effectively, it makes the image appear as if it were taken of the same sign, but from a different angle. The figure below shows an image before and after the affine transform.

![Affine Transform](Augmented.png)



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:


| Layer         		|     Description	        					             | 
|:----------------------|:-----------------------------------------------------------| 
| Input         		| 32x32x1 Preprocessed Greyscale Image   					 |
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x64                 |
| RELU					|												             |
| Max pooling	      	| 2x2 stride,  outputs 14x14x64 				             |
| Convolution 5x5     	| 1x1 stride, same padding, outputs 10x10x256                |
| RELU					|												             |
| Max pooling	      	| 2x2 stride,  outputs 5x5x64 				                 |
| Flatten       	    | outputs 6400    									         |        			
| Dropout               | 15% Dropout Probability                                    |
| Fully Connected		| Outputs 1200      									     |
| RELU					|												             |
| Fully Connected		| Outputs 600     									         |
| RELU					|												             |		
| Fully Connected		| Outputs 43     									         |
| RELU					|												             |
| Softmax               | 				                                             |	


The model I used was based on the LeNet-5 solution for the MNIST dataset, but with a couple of changes. In order to avoid overtraining the model, a dropout layer was addedbefore the first fully connected layer. Because of the addition of droupout, I found that the network needed significantly more filters in the convoultional layers. Before increasing the number of filters, droupout significantly decreased the acuracy of predictions. The decrease was relatively small after the filter count increase. I ended up outputting 64 filters in the first convolutional layer, and 256 in the second.


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

In order to train the model a number of parameters were adjusted. First, the learning rate was adjusted to maximize accuracy. This was done through trial-and-error. I started with a learning rate of 0.001, then tried an order of magnitude larger and smaller. This process was repeated until a relatively optimized learning rate was found.

With the learning rate picked, I then adjusted the number of epochs to avoid overtraining the model. The goal in picking the number of epochs was to make sure that the model was constantly making significant ijprovements in accuracy with each epoch. I initially ran the model for 10 epochs, but found that after 5, the accuracy began to stagnate, which is how I ended up with my final choice of 5 epochs.

Batch size was 128, which I did not change from the LeNet solution. That batch size seemed to run well, so I did not spend any time optimizing it.

I ended up using the Adam optimizer, which seemed like a reasonable choice and generated good results.

Finally, there were two other hyperparameters related to data augmentation that I ended up adjusting as well. The first was the extent to which images in the augmented set were transformed. This was achieved by defining how far points could move during the affine transform described above. The second parameter was the number of transformed images to add to the training dataset during augmentation. I ended up allowing points to move up to 4 pixels from their initial location> I added 10 sets of transformed images, meaning that the augmented training set was 11x the size of the original set. 


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy: 96.2%
* validation set accuracy: ? 
* test set accuracy: ?
* Internet images accuracy: 100%

In order to develop my architecture, I ended up using an iterave approach and testing many possible options. I started with the LeNet architecture from the exercise on the MNIST dataset, which performed moderately well (around 92% accuracy), but there was still slearly room for improvement. 

From that initial architecture, I tried adding additional convolutional layers in order to make sure that the network was able to handle the complexity of the sign images. I found that additional layers did not actually increase the accuracy of the model, so I then went back to an architecture similar to LeNet-5. 

With the architecture of the network set, I tested the performance. Whilet the model was able to achieve a relatively high accuracy on the validation data (94%), performance on the sign images found from the internet was low (60%). 

I attempted to improve the low accuracy on the internet images by adding dropout to the model in order to prevent overtraining. Unfortunately, dropout significantly decreased the accuracy of the model on the validation images. In order to bring the accuracy back into an acceptable range, I added significantly more filters to the convolutional layers of the model. This change significantly improved the performance of the model with dropout. 

For each step in the process above, I iteratively optimized the learning rate by testing at 0.01, 0.001, and 0.0001. After those three tests were complete, I tried training the at learning rate values close to the best one.

Batch size was not changed during the development of the model.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

The five greman traffic signs that I found on the web are shown below.

![Sign 1](1.jpg) ![Sign 2](2.png) ![Sign 3](3.png) 
![Sign 4](4.png) ![Sign 5](5.png)

These sign images were all screenshots taken from Google Street View, primarily in the Berlin area. I chose this approach in order to find actual street sign images as they appear in reality. In general, the signs images are very clear and should be relatively simple to read. The only exception is the fourth sign (Children Crossing). This sign seemed likely to pose a challenge to the classifier since part of it is obscured by a tree branch.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Roundabout Mandatory  | Roundabout Mandatory   						| 
| General Caution     	| General Caution  								|
| Speed limit (60 km/h)	| Speed limit (60 km/h)							|
| Children crossing	    | Slippery road					 				|
| Go straight or right	| Go straight or right      					|


The accuracy of the model was 80% (4 of 5 correctly predicted), which is roughly what would be expected given the accuracy of the model during validation and testing. As noted above, the Children Crossing sign posed the most difficulty for the classifier and was the only one that was not corectly classified.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The figure below shows the top five predictions for each of the input signs, along with the softmax probabilities that the model returned for each. Note that the probabilities are shown in the figure over the corresponding bar.

![Predictions](probabilities.png)

Below is a discussion of the prediction results for each of the signs:

* Sign 1, Roundabout Mandatory: This sign was one of the easiest for the model to predict. The model predicted the correct sign with 99.992% accuracy.
* Sign 2, General Caution: While the model predicted this sign correctly, it was the most unsure about this prediction. The correct prediction was made with 34.88% probability, and the probability of the second place prediction was 25.38%. The most likely cause for the difficulty in classifying this sign is that the image is taken from a fairly steep angle, and not straight-on to the sign.
* Sign 3, Speed Limit (60 km/h): This was another case of a very high probability prediction that was correct (88.27%). Interestingly, while the lower ranked predictions had very low probabilities (7% or less), they were all also speed limit signs. That may be because the border and second number are the sam on all of those signs.
* Sign 4, Children Crossing: This was the only sign that the model failed to predict correctly. It was also fairly confident in its incorrect prediction (91%). The incorect prediction was for a "Slippery road" sign. Both signs share the same shape and border, and have relatively complex symbols inside the border. As mentioned previously, the partial obscuring of the edge of the sign in this image may have presented additional difficulties.
* Sign 5, Go straight or right: Similar to the first sign, this one was predicted correctly, and with very high probability (99.998%). The symbols in both this sign and the first sign are very distinct, which may have contributed to the very high prediction probability.


