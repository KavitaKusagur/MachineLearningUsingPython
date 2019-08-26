Human-Activity-Recognition Using Smartphones Data Set
===========================


---------------------------------------------------------------------------

Repository Overview:
--------------------
This project aims to build a model that predicts the human activities such as Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing and Laying from the Sensor data of smart phones. 

All the code is written in python 3 

**DEPENDENCIES**

* numpy
* pandas
* matplotlib
* seaborn
* sklearn
* itertools
* datetime

Introduction:
--------------
Every modern Smart Phone has a number of sensors. we are interested in two of the sensors Accelerometer and Gyroscope. 
The data is recorded with the help of sensors 
This is a 6 class classification problem as we have 6 activities to detect.

This project tunes and compares Logistic Regression, Linear support vector classifier, RBF(Radial Basis Function) SVM classifier, Decision Tree, Random Forest, LGBM Classifier and uses the data featured by domain expert.

-------------------------------------------

Dataset:
----------
The dataset can be downloaded from https://www.kaggle.com/uciml/human-activity-recognition-with-smartphones# 

Human Activity Recognition database is built from the recordings of 30 persons performing activities of daily living (ADL) while carrying a waist-mounted smartphone with embedded inertial sensors(accelerometer and Gyroscope). 

**Activities**

* Walking
* Walking Upstairs
* Walking Downstairs
* Sitting
* Standing
* Laying

[**Accelerometers**](https://en.wikipedia.org/wiki/Accelerometer) detect magnitude and direction of the proper acceleration, as a vector quantity, and can be used to sense orientation (because direction of weight changes) 
<br><br>
[**GyroScope**] maintains orientation along a axis so that the orientation is unaffected by tilting or rotation of the mounting, according to the conservation of angular momentum. 
<br><br>
Accelerometer measures the directional movement of a device but will not be able to resolve its lateral orientation or tilt during that movement accurately unless a gyro is there to fill in that info. 
<br>
With an accelerometer you can either get a really "noisy" info output that is responsive, or you can get a "clean" output that's sluggish. But when you combine the 3-axis accelerometer with a 3-axis gyro, you get an output that is both clean and responsive in the same time. 
<br><br>

#### Understanding the dataset
Both sensors generate data in 3 Dimensional space over time. Hence the data captured are '3-axial linear acceleration'(tAcc-XYZ) from accelerometer and '3-axial angular velocity' (tGyro-XYZ) from Gyroscope with several variations.
prefix 't' in those metrics denotes time.
suffix 'XYZ' represents 3-axial signals in X , Y, and Z directions.
The available data is pre-processed by applying noise filters and then sampled in fixed-width windows(sliding windows) of 2.56 seconds each with 50% overlap. ie., each window has 128 readings.
#### Featurization
For each window a feature vector was obtained by calculating variables from the time and frequency domain. each datapoint represents a window with different readings.
Readings are divided into a window of 2.56 seconds with 50% overlapping.

* Accelerometer readings are divided into gravity acceleration and body acceleration readings, which has x,y and z components each.

* Gyroscope readings are the measure of angular velocities which has x,y and z components.

* Jerk signals are calculated for BodyAcceleration readings.

* Fourier Transforms are made on the above time readings to obtain frequency readings.

* Now, on all the base signal readings., mean, max, mad, sma, arcoefficient, engerybands,entropy etc., are calculated for each window.

* We get a feature vector of 561 features and these features are given in the dataset.

* Each window of readings is a datapoint of 561 features,and we have 10299 readings.

* These are the signals that we got so far.(prefix t means time domain data, prefix f means frequency domain data)

#### Train and test data were saperated
- The readings from 70% of the volunteers(21 people) were taken as trianing data and remaining 30% volunteers recordings(9 people) were taken for test data
* All the data is present in 'UCIML_HAR_dataset/' folder in present working directory.
  - Feature names are present in 'UCIML_HAR_dataset/features.txt'
  - __Train Data__ (7352 readings)
  - __Test Data__ (2947 readinds)
    
------------------------------------------

Analysis
-----------

For detailed code of this section you can always check the [HumanActivityRecognitionPrediction Notebook](https://github.com/KavitaPK/MachineLearningUsingPython/blob/master/Human%20Activity%20Recognition/HumanActivityRecognitionPrediction.ipynb)
<br>
##### Check for Imbalanced class
if some class have too little or too large numbers of values compared to rest of the classes than the dataset is imbalanced.
**Plot-1**
<br>
<img src="https://github.com/KavitaPK/MachineLearningUsingPython/blob/master/Human%20Activity%20Recognition/images/plot1.png" height=500 width=700>
<br><br>
Our data is well balanced (almost) & will prevent bias.
<br>
**Plot-2**
<br>
<img src="https://github.com/KavitaPK/MachineLearningUsingPython/blob/master/Human%20Activity%20Recognition/images/plot1_1.png">
<br><br>
In this plot on the X-axis we have subjects(volunteers) 1 to 30. Each color represents an activity
On the y-axis we have amount of data for each activity by provided by each subject.
<br>
From Plot-1 & Plot-2, we can see, our data is equally distributed (almost well Balanced). 
<br><br>
#### Variable analysis
**Plot-3**
<br>
<img src="https://github.com/KavitaPK/MachineLearningUsingPython/blob/master/Human%20Activity%20Recognition/images/plot2_2.png">
The above plot is of tBodyAccMagmean which is mean values of magnitude of acceleration in time space. 
<br><br>
**Plot-4**
<br>
Box plot, mean of magnitude of an acceleration 
<br>
<img src="https://github.com/KavitaPK/MachineLearningUsingPython/blob/master/Human%20Activity%20Recognition/images/plot3_3.png">
<br><br>
From plot-3 and plot-4 we can see that stationary activities can be linearly separated from activities with motion. 
<br><br>
**Plot-5**
<br>

Dimensionality reduction using T-distributed Stochastic Neighbor Embedding (t-SNE) to visualize 561 dimension dataset. 
<br>
<img src="https://github.com/KavitaPK/MachineLearningUsingPython/blob/master/Human%20Activity%20Recognition/images/plot3.png">
<br>
Sitting and standing are overlapped while other 4 classes can be separated well.

<br><br>

------------------------------------------------------------------------------------

Models
------

#### Machine Learning Algorithms

scikit-learn is used for all the 6 alogorithms listed below.<br>
Hyperparameters of all models are tuned by grid search CV<br>
Models fitted:<br>
- Logistic Regression
- Linear Support Vector Classifier(SVC)
- Radial Basis Function (RBF) kernel SVM classifier 
- Decision Tree 
- Random Forest 
- LGBM Classifier

#### Models Comparisions
|  model  | Accuracy |  Error|
|---|---|---|
| Logistic Regression |  96.27% | 3.733% |
| Linear SVC | 96.61% |  3.393% |
|rbf SVM classifier  | 96.27%    |  3.733% |
|Decision Tree  |       86.43%   |   13.57% |
|Random Forest |      91.31%    |  8.687% |
|LGBM Classifier | 95.5%    |   |


> **Observing the Top 2 Models**

**Logistic Regression**

**Plot-6**

Normalized confusion matrix for Linear Regression Model

<img src="https://github.com/KavitaPK/MachineLearningUsingPython/blob/master/Human%20Activity%20Recognition/images/plot6_6.png">

Diagonal Value of 1 means 100% accuracy for that class, and 0 means 0% accuracy.<br>
considering the diagonal elements we have value 1 for rows corresponding to 'Laying' and 'Walking'.<br>
while 'sitting' has value of only 0.87. In the row 2nd row and 3rd column we have value 0.12 which basically means about 12% readings of the class sitting is misclassified as standing.

**Linear SVC**

**Plot-7**

Normalized confusion matrix for Linear SVC Model

<img src="https://github.com/KavitaPK/MachineLearningUsingPython/blob/master/Human%20Activity%20Recognition/images/plot7_7.png" >

In this model also the diagonal elements, we have value 1 for rows corresponding to 'Laying' and 'Walking'.<br>
Again row corresponding to 'sitting' has value of only 0.87. In the row 2nd row and 3rd column we have value 0.12 which basically means about 12% readings of the class sitting is misclassified as standing.<br>
<br>
It is not a surprise as in the t-sne plot (plot-5) we saw that 'sitting' and 'Standing' class readings are overlapping.

For detailed code of all the ML models check the [HumanActivityRecognitionPrediction Notebook](https://github.com/KavitaPK/MachineLearningUsingPython/blob/master/Human%20Activity%20Recognition/HumanActivityRecognitionPrediction.ipynb)


------------------------------------------------------------------------------------------------

References:
-----------

https://en.wikipedia.org/wiki/Gyroscope <br>
https://scikit-learn.org/stable/supervised_learning.html#supervised-learning <br>
https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc
