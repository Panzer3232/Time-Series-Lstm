# Time-Series Analysis for Rainfall Data using LSTM Model

## Pipeline For the Project 

### 1) Data Preprocessing :- 
a) All necessary libraries are imported  for data manipulation, model creation and visualization.<br>
b) CSV data is loaded and "Year" column is dropped and dataset is split into training and testing sets.<br>
c) Normalization of dataset is done to scale data between 0 and 1 using MinMaxScalar.<br>
d) Input and target sequences for LSTM model is created.<br>

### 2) Creation of LSTM Model
a) LSTm based neural network, that takes input size (number of features), hidden size of LSTM units, number of output features. <br>
b) Model contains a LSTM layer and Linear Fully connected layer.<br>
c) Forward function process the input and produce the output. <br>

### 3) Train the Neural Network
a) Train the model using <b> train_test.py </b> script <br> python train_test.py <br>
b) The <b>train_model()</b> prints the loss at every 10th epoch. <br>
c) Model uses the Mean Squared error as the loss function and also uses Adam optimizer <br>

