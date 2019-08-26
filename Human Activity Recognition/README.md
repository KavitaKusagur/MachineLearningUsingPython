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

dataset is also included in the Repository with in the folder UCIML_HAR_Dataset 

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
    - 'UCIML_HAR_dataset/train.csv'
  - __Test Data__ (2947 readinds)
    - 'UCIML_HAR_dataset/test.csv'
------------------------------------------

Analysis
-----------

For detailed code of this section you can always check the [HumanActivityRecognitionPrediction Notebook](https://github.com/KavitaPK/MachineLearningUsingPython/edit/master/Human%20Activity%20Recognition/HumanActivityRecognitionPrediction)
<br>
##### Check for Imbalanced class
if some class have too little or too large numbers of values compared to rest of the classes than the dataset is imbalanced.
**Plot-1**
<br>
<img src="https://github.com/srvds/Human-Activity-Recognition/blob/master/plots/plot1.png" height=500 width=700>
<br><br>
