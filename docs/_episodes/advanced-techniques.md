---
title:  Recurrent Neural Networks (RNNs) Models
teaching: 1
exercises: 0
questions:
- "What are the primary advantages of using Recurrent Neural Networks (RNNs) for time series forecasting over traditional statistical methods and other machine learning algorithms?"
- "What are the key differences between traditional RNNs and advanced RNN models such as LSTMs and GRUs?"
- "What are some common challenges faced when training LSTM models and how can they be mitigated?"
- "How do Long Short-Term Memory (LSTM) networks and Gated Recurrent Units (GRUs) enhance the capability of RNNs in learning and remembering temporal dependencies in sequential data?"

- "What recent advancements in RNN variants, such as the Temporal Fusion Transformer (TFT), have contributed to improved time series forecasting in physical sciences applications?"
objectives:
- "To identify the advantages of Recurrent Neural Networks (RNNs) in time series forecasting compared to traditional statistical methods and other machine learning algorithms."
- "To understand the role of Long Short-Term Memory (LSTM) networks and Gated Recurrent Units (GRUs) in enhancing the memory and temporal dependency learning capabilities of RNNs."
- "To explore the recent advancements in RNN variants, such as the Temporal Fusion Transformer (TFT), and their impact on time series forecasting in physical sciences."
keypoints:
- "LSTMs and GRUs are advanced RNN architectures designed to handle long-term dependencies in sequential data."
- "The application of deep learning, particularly through RNNs and their variants like LSTM, GRU, and TFT, holds significant promise for time series forecasting in the physical sciences"
- "LSTM, GRU and TFT models leverage advanced mechanisms for superior predictive performance in physical sciences applications."

---

<!-- MathJax -->

<script type="text/javascript"

  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML">

</script>


# Time Series Modeling 

Time series data consists of observations recorded at regular intervals, and time series forecasting aims to predict future values based on historical patterns and trends. This type of forecasting is critical across various domains, making it highly pertinent for machine learning practitioners. Applications such as risk scoring, fraud detection, and weather forecasting all benefit significantly from accurate time series predictions.

Deep learning, particularly neural networks, has only recently started to outperform traditional methods in time series forecasting, albeit by a smaller margin compared to its successes in image and language processing. Despite this, deep learning presents distinct advantages over other machine learning techniques, such as gradient-boosted trees. The primary advantage lies in the ability of neural network architectures to inherently understand time, automatically linking temporally close data points.

A Recurrent Neural Network (RNN) is a type of artificial neural network that includes memory or feedback loops, allowing it to recognize patterns in sequential data more effectively. The recurrent connections enable the network to consider both the current data sample and its previous hidden state, enhancing its ability to capture complex temporal dependencies. Some RNNs, like Long Short-Term Memory (LSTM) networks and Gated Recurrent Units (GRUs), have specialized memory mechanisms to remember significant events over extended time periods, which is essential for sequence learning tasks.

The distinguishing feature of sequence learning, compared to other regression and classification tasks, is the necessity to utilize models that can learn temporal dependencies in input data. LSTMs and GRUs excel in this regard, offering the ability to capture intricate time dependencies. Furthermore, neural networks employ continuous activation functions, making them particularly adept at interpolation in high-dimensional spaces, thereby optimizing inputs effectively.

Recent advancements in RNN variants, such as the Temporal Fusion Transformer (TFT), have further enhanced the capabilities of time series forecasting in physical science applications. These state-of-the-art models leverage advanced attention mechanisms and complex architectural innovations to provide superior predictive performance.


## Time Series EDA

- Let's start with calculating the rolling mean and standard deviation, adds upper and lower bounds, and visualizes the \\( CO_2 \\)levels data trends over time in a plot with enhanced aesthetics using seaborn:

~~~
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
co2_levels = pd.read_csv('data/co2_levels.csv')

# Convert the 'timestamp' column to datetime format
co2_levels['datestamp'] = pd.to_datetime(co2_levels['datestamp'])

# Set the 'timestamp' column as the index
co2_levels.set_index('datestamp', inplace=True)

# Compute the 52 weeks rolling mean of the co2_levels DataFrame
ma = co2_levels.rolling(window=52).mean()

# Compute the 52 weeks rolling standard deviation of the co2_levels DataFrame
mstd = co2_levels.rolling(window=52).std()

# Add the upper bound column to the ma DataFrame
ma['upper'] = ma['co2'] + (mstd['co2'] * 2)

# Add the lower bound column to the ma DataFrame
ma['lower'] = ma['co2'] - (mstd['co2'] * 2)

# Plot the content of the ma DataFrame
ax = ma.plot(linewidth=0.8, fontsize=6)

# Specify labels, legend, and show the plot
ax.set_xlabel('Date', fontsize=10)
ax.set_ylabel(r'$CO_2$ levels', fontsize=10)
ax.set_title(r'Rolling mean and variance of $CO_2$ levels\n from 1958 to 2001', fontsize=10)
plt.show();

~~~
{: .python}


![](../fig/co2_rolling_mean.png)


The increasing rolling mean of \\( CO_2 \\)  levels suggests a long-term upward trend in atmospheric \\( CO_2 \\) concentrations, which is a key indicator of climate change. This trend is likely driven by human activities and has important implications for the environment and global climate.



> ## Exercise: Plot the \\(CO_2\\) time series with a vertical line
> 
> - Plot the \\(CO_2\\) levels over time using the `plot()` method
> - Add a red vertical line at the date '1960-01-01' using `axvline()`
> - Set the x-axis label to 'Date' and the title to 'Number of Monthly CO2'
> 
> > ## Solution
> > ~~~
> > # Plot the time series in your dataframe
> > ax = co2_levels.plot(color='blue', fontsize=12)
> > # Add a red vertical line at the date 1960-01-01
> > ax.axvline('1960-01-01', color='red', linestyle='--')
> > # Specify the labels in your plot
> > ax.set_xlabel('Date', fontsize=12)
> > ax.set_title(r'Number of Monthly $CO_2$', fontsize=12)
> > plt.show()
> > ~~~
> > {: .python}
> ![](../fig/co2_tends.png)
> {: .solution}
> 
{: .challenge}



## Treand, Seasonality and Noise

In time series analysis, data often comprises three main components: seasonality, trend, and noise. Seasonality refers to recurring patterns at regular intervals, trend indicates long-term direction, and noise represents random fluctuations. Understanding these components is essential for accurate analysis and forecasting. The code snippet below demonstrates time series decomposition using `seasonal_decompose` from `statsmodels` to extract and visualize the trend component of the \\(CO_2\\) time-series data.

~~~
# Import statsmodels.api as sm
import statsmodels.api as sm
# Perform time series decompositon
decomposition = sm.tsa.seasonal_decompose(co2_levels)
# Extract the trend component
trend = decomposition.trend
# Plot the values of the trend
ax = trend.plot(figsize=(12, 6), fontsize=6)
# Specify axis labels
ax.set_xlabel('Date', fontsize=10)
ax.set_title(r'Trend component of  $CO_2$ time-series', fontsize=10)
plt.show()
~~~
{: .python}


![](../fig/co2_trends_component.png)


> ## Exercise 1: Plot the seasonal components of CO2 time series
> 
> - Import the `statsmodels.api` module as `sm`
> - Perform time series decomposition using `seasonal_decompose` on the `co2_levels` DataFrame
> - Extract the seasonal components from the decomposition
> - Plot the seasonal component values using `plot()` 
> - Set the x-axis label to 'Date' and the title to 'Seasonal component of CO2 time-series'
> 
> > ## Solution
> > ~~~
> > import statsmodels.api as sm
> > # Perform time series decomposition
> > decomposition = sm.tsa.seasonal_decompose(co2_levels)
> > # Extract the seasonal components
> > seasonal = decomposition.seasonal
> >
> > # Plot the values of the seasonality 
> > ax = seasonal.plot(figsize=(12, 6), fontsize=6)
> > # Specify axis labels
> > ax.set_xlabel('Date', fontsize=10)
> > ax.set_title(r'Seasonal component of $CO_2$ time-series', fontsize=10)
> > plt.show()
> > ~~~
> > {: .python}
> ![](../fig/co2_seasonal_component.png)
> {: .solution}
> 
{: .challenge}


> ## Exercise 2: Decompose and Plot \\(CO2\\) Time Series with Log Scale
> In this exercise, you will perform time series decomposition on CO2 levels data, extract the trend and seasonal components, and plot the decomposed time series using a log scale.
> - Decompose \\(CO_2\\) time series, extract the trend and seasonal components, and plot the decomposed time series using a log scale on the y-axis.
>
> > ## Solution
> > ~~~
> > from statsmodels.tsa.seasonal import seasonal_decompose
> > import numpy as np
> > import pandas as pd
> > import matplotlib.pyplot as plt
> > import seaborn as sns
> > plt.style.use('ggplot')
> > co2_levels = pd.read_csv('data/co2_levels.csv')
> > # Convert the 'timestamp' column to datetime format
> > co2_levels['datestamp'] = pd.to_datetime(co2_levels['datestamp'])
> > # Set the 'timestamp' column as the index
> > co2_levels.set_index('datestamp', inplace=True)
> > # Perform time series decomposition
> > result = seasonal_decompose(co2_levels, model='additive', period=12)
> > # Plot the values of the df_decomposed DataFrame
> > ax = result.plot(figsize=(12, 6), fontsize=15, logy=True)
> > # Specify axis labels
> > ax.set_xlabel('Year', fontsize=15)
> > plt.legend(fontsize=15)
> > plt.show()
> > ~~~
> > {: .python}
> ![](../fig/co2_seasonal_trends.png)
> {: .solution}
{: .challenge}
 

> ## Exercise 3: Decompose and Plot CO2 Time Series Components
> - Load CO2 levels data and set the 'Year' column as the index.
> -  Perform time series decomposition with an additive model and a period of 12.
> -  Plot the observed, trend, seasonal, and residual components on separate subplots.
> - Adjust subplot spacing and display the plot.
> 
> > ## Solution
> > ~~~
> > from statsmodels.tsa.seasonal import seasonal_decompose
> > import numpy as np
> > import pandas as pd
> > import matplotlib.pyplot as plt
> > import seaborn as sns
> > plt.style.use('ggplot')
> > co2_levels = pd.read_csv('data/co2_levels.csv')
> > # Convert the 'timestamp' column to datetime format
> > co2_levels['Year'] = pd.to_datetime(co2_levels['datestamp'])
> > # Set the 'Year' column as the index
> > co2_levels.set_index('Year', inplace=True)
> > # Perform time series decomposition
> > result = seasonal_decompose(co2_levels['co2'], model='additive', period=12)
> > # Plot the decomposed components on individual subplots
> > fig, axes = plt.subplots(4, 1, figsize=(12, 8))
> > result.observed.plot(ax=axes[0])
> > axes[0].set_title('Observed')
> > result.trend.plot(ax=axes[1])
> > axes[1].set_title('Trend')
> > result.seasonal.plot(ax=axes[2])
> > axes[2].set_title('Seasonal')
> > result.resid.plot(ax=axes[3])
> > axes[3].set_title('Residual')
> > plt.tight_layout()
> > plt.show()
> > ~~~
> > {: .python}
> ![](../fig/co2_seasonal_trends_noise.png)
> {: .solution}
> 
{: .challenge}




## Time Series Modeling with RNN

In recent years, deep learning has emerged as a powerful tool for time series forecasting, offering significant advantages over traditional statistical methods. While classical approaches like ARIMA and exponential smoothing rely heavily on assumptions about the data's structure, deep learning models, particularly neural networks, can automatically capture complex patterns __without extensive manual feature engineering__.

Among the various types of neural networks, Recurrent Neural Networks (RNNs) have shown exceptional promise for time series tasks due to their inherent ability to process sequences of data. RNNs, with their internal memory and feedback loops, excel at recognizing temporal dependencies, making them well-suited for forecasting tasks where past observations are crucial for predicting future values.

However, standard RNNs face challenges such as vanishing gradients, which can hinder their performance on long sequences. To address these issues, advanced architectures like Long Short-Term Memory (LSTM) networks and Gated Recurrent Units (GRUs) have been developed. These models incorporate mechanisms to maintain and update memory over longer periods, thereby improving the model's ability to learn from and retain long-term dependencies in the data.


Recurrent Neural Networks (RNNs) are a type of neural network specifically designed to handle sequential data by incorporating feedback loops that allow information to persist. This architecture enables RNNs to capture temporal dependencies and patterns within time series data, making them particularly effective for tasks such as language modeling, speech recognition, and time series forecasting. Unlike traditional feedforward neural networks, RNNs maintain a hidden state that is updated at each time step, providing a dynamic memory that can process sequences of varying lengths. Advanced variants like Long Short-Term Memory (LSTM) networks and Gated Recurrent Units (GRUs) further enhance this capability by mitigating issues like vanishing gradients, thus enabling the modeling of long-term dependencies more effectively.


## LSTM

Long Short-Term Memory (LSTM) networks are a specialized type of Recurrent Neural Network (RNN) designed to effectively capture long-term dependencies in sequential data by incorporating memory cells that can maintain and update information over extended time periods. The LSTM architecture addresses the vanishing gradient problem prevalent in standard RNNs, allowing for better training and performance on long sequences. In the provided LSTM model, the network consists of an LSTM layer followed by a fully connected layer, which processes the output of the last time step to produce the final prediction. This structure enables the model to learn complex temporal patterns and make accurate forecasts or classifications based on sequential inputs. 


This training material provides a structured approach to implementing both singel-variable and multi-variable time series prediction using Long Short-Term Memory (LSTM) networks. LSTM networks are a type of recurrent neural network (RNN) that are particularly effective for sequence prediction tasks due to their ability to capture long-term dependencies in data.

### LSTM for Single-Variable Time Series Forecasting

Long Short-Term Memory (LSTM) networks are a type of recurrent neural network (RNN) that excel in learning from sequential data, making them particularly useful for time series forecasting. In this tutorial, we will explore how to implement an LSTM network to predict future values in a one-variable time series dataset.

We begin by loading necessary libraries such as NumPy, Matplotlib, and Pandas for data manipulation and visualization, along with PyTorch for building and training our neural network model. The dataset we'll use consists of monthly airline passenger numbers, a classic example in time series analysis.

First, we load the dataset and display the initial few rows to understand its structure:

~~~
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import random
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
plt.style.use("ggplot")

# Set seed for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


# Load the dataset
df = pd.read_csv('data/co2_levels.csv')
# Convert 'datestamp' column to datetime format
df['datestamp'] = pd.to_datetime(df['datestamp'])
# Reindex the DataFrame before splitting
df = df.set_index('datestamp')
# Display the first few rows of the DataFrame
print(df.head())
~~~
{: .python}

The dataset comprises two columns: `datestamp` and \\(CO_2\\). For our analysis, we will focus on the \\(CO_2\\) column as our variable of interest.

~~~
            co2
datestamp        
1958-03-29  316.1
1958-04-05  317.3
1958-04-12  317.6
1958-04-19  317.5
1958-04-26  316.4
~~~
{: .output}

Next, we extract the passenger data and visualize it to get an initial sense of the time series trend.

~~~
# Plot the CO2 levels over time
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['co2'], label='CO2 Levels')
plt.title('CO2 Levels Over Time')
plt.xlabel('Date')
plt.ylabel('CO2 Levels')
plt.legend()
plt.show()
~~~
{: .python}

![](../fig/co2_time_series.png)

This visualization step helps us understand the overall trend and seasonality in the data, setting the stage for building our LSTM model. Through this tutorial, you will learn how to preprocess the data, construct the LSTM network, and evaluate its performance in forecasting future passenger numbers.


#### Data Windowing for Time Series

To effectively train an LSTM, it is crucial to organize the time series data into sequences that the network can learn from. This involves creating sliding windows of a fixed length, where each window represents a sequence of past values that will be used to predict the next value in the series.  RNN models makes predictions based on this sliding window of consecutive data samples. key features of input sliding windows include:

1. **Width**: Number of time steps in the input and label windows.
2. **Offset**: Time gap between input and label windows.
3. **Features**: Selection of features as inputs, labels, or both.

We will construct various models (Linear, DNN, CNN, RNN) for:
- Single-output and multi-output predictions.
- Single-time-step and multi-time-step predictions.

Depending on the task and type of model you may want to generate a variety of data windows. For instance, to make a single prediction 24 hours into the future, given 24 hours of history you migh define a window like this:

![](../fig/raw_window_24h.png)


Forthermore, to make a prediction one hour into the future, given six hours of history would need a windos of input length six(6) with offset 1 as shown below:


![](../fig/raw_window_1h.png)


In the remainder of this section, we define a `sliding_windows` class. This class can:

- Manage indexes and offsets.
- Split windows of features into feature (X) and label (y) pairs.
- Efficiently generate batches of these windows from the training and test data.


The sliding window generator class is crucial for preparing data for time series forecasting. The code below demonstrates its implementation:

~~~
# Sliding window generation class
class SlidingWindowGenerator:
    def __init__(self, seq_length, label_width, shift, df, label_columns=None, dropnan=True):
        self.df = df
        self.label_columns = label_columns
        self.dropnan = dropnan
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(df.columns)}
        self.seq_length = seq_length
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = seq_length + shift
        self.input_slice = slice(0, seq_length)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])
    def sliding_windows(self):
        data = self.df.values
        X, y = [], []
        for i in range(len(data) - self.total_window_size + 1):
            input_window = data[i:i + self.seq_length]
            label_window = data[i + self.seq_length:i + self.total_window_size]
            X.append(input_window)
            if self.label_columns is not None:
                label_window = label_window[:, [self.column_indices[name] for name in self.label_columns]]
            y.append(label_window)
        X, y = np.array(X), np.array(y)
        if self.dropnan:
            X = X[~np.isnan(X).any(axis=(1, 2))]
            y = y[~np.isnan(y).any(axis=(1, 2))]
        return X, y.reshape(-1,1)
~~~
{: .python}

The code initializes a sliding window generator with specified parameters, including input width, label width, and shift. Below is an example demonstrating how to create and use a sliding window generator with a DataFrame:
~~~
# Initialize the generator
swg = SlidingWindowGenerator(seq_length=4, label_width=5, shift=1, df=df, label_columns=['co2'])
print(swg)
~~~
{: .python}


~~~
Total window size: 5
Input indices: [0 1 2 3]
Label indices: [4]
Label column name(s): ['co2']
~~~
{: .output}



~~~
# Generate windows
X, y = swg.sliding_windows()
X.shape, y.shape
~~~
{: .python}


~~~
((2280, 4, 1), (2280, 1))
~~~
{: .output}


The arrays `X` and `y` store these windows and targets, respectively, and are converted to NumPy arrays for efficient computation.

By setting `seq_length = 4`, we generate sequences length of 4 with offset 1 where each input sequence consists of four time steps, and the corresponding target is the value immediately following this sequence.

This preprocessing step prepares the data for the LSTM network, enabling it to learn from the sequential patterns in the time series and predict future \\(CO_2\\) levels based on past observations.

Next, we will proceed to construct tensor format preprocessed data  and the LSTM model to train it, ultimately evaluating its performance in forecasting future values.


First, we need to split the dataset into training and testing sets and convert them into tensors, which are the primary data structure used in PyTorch.


~~~
# Normalize the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_shape = X.shape
y_shape = y.shape

X_flat = X.reshape(-1, X_shape[-1])
y_flat = y.reshape(-1, y_shape[-1])

X = scaler_X.fit_transform(X_flat).reshape(X_shape)
y = scaler_y.fit_transform(y_flat).reshape(y_shape)

# train and test data loading in tensor format
train_size = int(len(y) * 0.7)
test_size = len(y) - train_size

X_train = Variable(torch.Tensor(np.array(X)))
y_train = Variable(torch.Tensor(np.array(y)))

X_train = Variable(torch.Tensor(np.array(X[0:train_size])))
y_train = Variable(torch.Tensor(np.array(y[0:train_size])))

X_test = Variable(torch.Tensor(np.array(X[train_size:len(X)])))
y_test = Variable(torch.Tensor(np.array(y[train_size:len(y)])))
~~~
{: .python}

Here, train_size is set to 70% of the dataset, while test_size is the remaining 30%. We convert the respective segments of X and y into PyTorch tensors using Variable.

Next, we define our LSTM model by creating a class that inherits from nn.Module. This class includes the initialization of the LSTM and a forward method to define the forward pass of the network.

~~~
# the LSTM model building

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
~~~
{: .python}

Now, we train the LSTM model. We set the number of epochs, learning rate, and other hyperparameters. We use mean squared error (MSE) as the loss function and Adam optimizer for training.
~~~
# training the model
num_epochs = 2000
learning_rate = 0.01

input_size = X.shape[2] # feature vectors 
hidden_size = 3
num_layers = 1
output_size = 1

lstm = LSTM(input_size, hidden_size, num_layers, output_size)

criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    outputs = lstm(X_train)
    optimizer.zero_grad()

    # obtain the loss function
    loss = criterion(outputs, y_train)

    loss.backward()

    optimizer.step()
    if epoch % 100 == 0:
      print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
~~~
{: .python}

~~~
Epoch: 0, loss: 0.06325
Epoch: 100, loss: 0.00028
Epoch: 200, loss: 0.00018
Epoch: 300, loss: 0.00017
Epoch: 400, loss: 0.00016
Epoch: 500, loss: 0.00015
Epoch: 600, loss: 0.00014
...
Epoch: 1500, loss: 0.00008
Epoch: 1600, loss: 0.00007
Epoch: 1700, loss: 0.00007
Epoch: 1800, loss: 0.00007
Epoch: 1900, loss: 0.00006
~~~
{: .output}
The training loop runs for 2000 epochs, and the loss is printed every 100 epochs to monitor the training process.

After training, we evaluate the model's performance on the test data. We set the model to evaluation mode and generate predictions for the test set. These predictions and the actual values are then inverse-transformed to their original scale for visualization.

~~~
# Testing the model performance
lstm.eval()
test_predict = lstm(X_test)

# Convert predictions to numpy arrays and reshape
data_predict = test_predict.data.numpy().reshape(-1, 1)
dataY_plot = y_test.data.numpy().reshape(-1, 1)

# Inverse transform the predictions and actual values
data_predict = scaler_y.inverse_transform(data_predict)
dataY_plot = scaler_y.inverse_transform(dataY_plot)

# Compute MSE and R²
mse = mean_squared_error(dataY_plot, data_predict)
r2 = r2_score(dataY_plot, data_predict)

# Get the test datestamps
test_size = len(dataY_plot)
test_dates = df.index[-test_size:]

# Plot observed and predicted values
plt.figure(figsize=(12, 6))
plt.axvline(x=test_dates[0], c='r', linestyle='--', label='Train/Test Split')
plt.plot(test_dates, dataY_plot, label='Observed')
plt.plot(test_dates, data_predict, label='Predicted')
plt.suptitle('Time-Series Prediction')
plt.xlabel('Date')
plt.ylabel(r'$CO_2$')
plt.legend()

# Add MSE and R2 values as annotations
plt.text(0.5, 0.9, f'MSE: {mse:.5f}', ha='center', va='center', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
plt.text(0.5, 0.8, f'R²: {r2:.5f}', ha='center', va='center', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
plt.tight_layout()
plt.show()
~~~
{: .python}

![](../fig/X_test2_predict.png)


In the plot, the red vertical line separates the training data from the test data. The model's performance in predicting the time series is visualized by plotting the predicted values against the actual values.

By following these steps, you will preprocess the data, construct an LSTM network, train it, and evaluate its performance in forecasting future \\(CO_2\\) levels.

> ## Exercise: How to impove the model performance?
> - Modify the sequence length  to 5 used in the LSTM model and observe its impact on the model's performance.
> - Modify the hidden layers of the LSTM model to 5 and retain the model and plot the observed vs prediction for `X_test`
> 
> > ## Solution
> > ~~~
> > # Modify the sequence length to 5
> > windows = SlidingWindowGenerator(seq_length=5, label_width=1, shift=1, df=df, label_columns=['co2'])
> > X, y = windows.sliding_windows(training_data)
> > # Train and test data loading in tensor format
> > train_size = int(len(y) * 0.70)
> > test_size = len(y) - train_size
> > X_train = Variable(torch.Tensor(np.array(X[0:train_size])))
> > y_train = Variable(torch.Tensor(np.array(y[0:train_size])))
> > X_test = Variable(torch.Tensor(np.array(X[train_size:len(X)])))
> > y_test = Variable(torch.Tensor(np.array(y[train_size:len(y)])))
> > # LSTM model building
> > # Training the model
> > num_epochs = 2000
> > learning_rate = 0.01
> > input_size = 1
> > hidden_size = 5
> > num_layers = 1
> > num_classes = 1
> > lstm = LSTM(input_size, hidden_size, num_layers, output_size)
> > criterion = torch.nn.MSELoss()    # Mean-squared error for regression
> > optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
> > # Train the model
> > for epoch in range(num_epochs):
> >    outputs = lstm(X_train)
> >    optimizer.zero_grad()   
> >    loss = criterion(outputs, y_train)
> >    loss.backward()
> >    optimizer.step()    
> >    if epoch % 100 == 0:
> >        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
> > # Testing the model performance
> > from sklearn.metrics import mean_squared_error, r2_score
> > lstm.eval()
> > train_predict = lstm(X_test)
> > data_predict = train_predict.data.numpy()
> > dataY_plot = y_test.data.numpy()
> > # Inverse transform the predictions and actual values
> > data_predict = scaler_y.inverse_transform(data_predict)
> > dataY_plot = scaler_y.inverse_transform(dataY_plot)
> > # Compute MSE and R²
> > mse = mean_squared_error(dataY_plot, data_predict)
> > r2 = r2_score(dataY_plot, data_predict)
> > # Get the test datestamps
> > test_dates = df.index[train_size + seq_length + 1:]
> > # Plot observed and predicted values
> > plt.figure(figsize=(12, 6))
> > plt.axvline(x=test_dates[0], c='r', linestyle='--', label='Train/Test Split')
> > plt.plot(df.index[train_size + seq_length + 1:], dataY_plot, label='Observed')
> > plt.plot(df.index[train_size + seq_length + 1:], data_predict, label='Predicted')
> > plt.suptitle('Time-Series Prediction')
> > plt.xlabel('Date')
> > plt.ylabel(r'$CO_2$')
> > plt.legend()
> > # Add MSE and R2 values as annotations
> > plt.text(0.5, 0.9, f'MSE: {mse:.5f}', ha='center', va='center', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
> > plt.text(0.5, 0.8, f'R²: {r2:.5f}', ha='center', va='center', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
> > plt.show()
> > ~~~
> > {: .python}
> > ![](../fig/X_test3_predict.png)
> {: .solution}
>
{: .challenge}




### LSTM for Multi-Variable Time Series Forecasting

For this session, we will use the Jena Climate dataset, compiled by the Max Planck Institute for Biogeochemistry, contains 14 meteorological features, such as temperature, pressure, and humidity, recorded every 10 minutes from January 10, 2009, to December 31, 2016.The  key features of the ata dataset are:

| Column Name     | Description                          |
|-----------------|--------------------------------------|
| `Date Time`     | Timestamp (every 10 minutes)         |
| `T (degC)`      | Temperature in degrees Celsius       |
| `p (mbar)`      | Atmospheric pressure in millibars    |
| `rh (%)`        | Relative humidity in percentage      |
| `VPmax (mbar)`  | Maximum vapor pressure in millibars  |
| `VPact (mbar)`  | Actual vapor pressure in millibars   |
| `VPdef (mbar)`  | Vapor pressure deficit in millibars  |
| `sh (g/kg)`     | Specific humidity in grams per kg    |
| `H2OC (mmol/mol)`| Water vapor concentration            |
| `rho (g/m**3)`  | Air density                          |
| `wv (m/s)`      | Wind velocity                        |
| `max. wv (m/s)` | Maximum wind velocity                |
| `wd (deg)`      | Wind direction                       |


This dataset is ideal for various time series analysis applications, including forecasting, trend analysis, and anomaly detection. Given its high frequency and rich feature set, it provides a robust platform for modeling and predicting following climatic conditions:

- **Temperature Forecasting:** Predict future temperatures.
- **Pressure Trends:** Analyze atmospheric pressure trends.
- **Humidity Patterns:** Investigate changes in relative humidity.


#### Raw Data Visualization

To understand the dataset, each feature is plotted below, showing patterns from 2009 to 2016. These plots reveal distinct trends and highlight anomalies, which we will address during normalization.

~~~
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Set seed for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

plt.style.use("ggplot")

# Load the dataset
df = pd.read_csv('data/weather_forecast.csv')
# Titles and feature keys
titles = [
    "Pressure", "Temperature", "Temperature in Kelvin", "Temperature (dew point)",
    "Relative Humidity", "Saturation vapor pressure", "Vapor pressure",
    "Vapor pressure deficit", "Specific humidity", "Water vapor concentration",
    "Airtight", "Wind speed", "Maximum wind speed", "Wind direction in degrees"
]

feature_keys = [
    "p(mbar)", "T(degC)", "Tpot(K)", "Tdew(degC)", "rh(%)", "VPmax(mbar)",
    "VPact(mbar)", "VPdef(mbar)", "sh(g/kg)", "H2OC(mmol/mol)", "rho(g/m**3)",
    "wv(m/s)", "max.wv(m/s)", "wd(deg)"
]

colors = [
    "blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"
]

date_time_key = "DateTime"

def show_raw_visualization(data):
    time_data = data[date_time_key]
    fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(15, 20), dpi=80)
    axes = axes.flatten()  # Flatten axes array for easy indexing
    
    for i, (key, title) in enumerate(zip(feature_keys, titles)):
        color = colors[i % len(colors)]
        ax = axes[i]
        data[key].plot(ax=ax, color=color, title=f"{title} - {key}", rot=25)
        ax.set_xlabel("")  # Remove x-axis label for clarity
        ax.legend([title])
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()

# Show the raw data visualization
show_raw_visualization(df)
~~~
{: .python}

![](..fig/weather_forecastig_raw)


### Data Preparation

We can see from the correlation heatmap, few parameters like Relative Humidity and Specific Humidity are redundant. Hence we will be using select features, not all. 

In this section, we begin by loading the dataset and performing initial preprocessing steps. The dataset, stored in a CSV file, contains various meteorological measurements such as pressure, temperature, and wind direction. We extract relevant features and reformat the data to facilitate further processing. Additionally, we normalize the data to ensure that all features are on a similar scale, which is crucial for training neural networks effectively.

~~~

# Clean the 'DateTime' column by removing malformed entries
df = df[df['DateTime'].str.match(r'\d{4}-\d{2}-\d{2}.*')]
# Convert 'DateTime' column to datetime format, allowing pandas to infer the format
df["DateTime"] = pd.to_datetime(df["DateTime"], errors='coerce')
# Drop rows where the 'DateTime' conversion resulted in NaT (not-a-time)
df.dropna(subset=["DateTime"], inplace=True)

# Reindex the DataFrame before splitting
df.set_index('DateTime', inplace=True)
                                                                                                                                                               
# select only important features
features = ['p(mbar)','T(degC)', 'VPmax(mbar)','VPdef(mbar)', 'sh(g/kg)', 'rho(g/m**3)',  'wv(m/s)', 'wd(deg)' ]
df = df[features]

# Resample the DataFrame by day and compute the mean for each day 
df_daily = df.resample('D').mean() 
# Display the first few rows of the resampled DataFrame 
df_daily.head()
~~~
{: .python}


~~~
               p(mbar)   T(degC)  ...   wv(m/s)     wd(deg)
DateTime                          ...                      
2009-01-01  999.145594 -6.810629  ...  0.778601  181.863077
2009-01-02  999.600625 -3.728194  ...  1.419514  125.072014
2009-01-03  998.548611 -5.271736  ...  1.250903  190.383333
2009-01-04  988.510694 -1.375208  ...  1.720417  213.069861
2009-01-05  990.405694 -4.867153  ...  3.800278  118.287361

[5 rows x 8 columns]
~~~
{: .output}


### Sliding Window Generator

The sliding window generator is a crucial component for preparing time series data for training predictive models. It segments the time series into input-output pairs, where the input consists of past observations (window) and the output is the next observation(s) to be predicted. This approach enables the LSTM model to learn from sequential patterns in the data. The generator's parameters, such as sequence length and label width, dictate the temporal context and prediction horizon, respectively.

~~~
import numpy as np

class SlidingWindowGenerator:
    def __init__(self, seq_length, label_width, shift, df, label_columns=None, dropnan=True):
        self.df = df
        self.label_columns = label_columns
        self.dropnan = dropnan

        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(df.columns)}

        self.seq_length = seq_length
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = seq_length + shift

        self.input_slice = slice(0, seq_length)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def sliding_windows(self):
        data = self.df.values
        X, y = [], []

        for i in range(len(data) - self.total_window_size + 1):
            input_window = data[i:i + self.seq_length]
            label_window = data[i + self.seq_length:i + self.total_window_size]

            # Check for nan values in input_window and label_window by flattening them
            if np.isnan(input_window.flatten()).any() or np.isnan(label_window.flatten()).any():
                continue  # Skip this window if it contains nan values

            X.append(input_window)

            if self.label_columns is not None:
                label_window = label_window[:, [self.column_indices[name] for name in self.label_columns]]
            y.append(label_window)

        X, y = np.array(X), np.array(y)

        return X, y.reshape(-1, 1)

# Initialize the generator
swg = SlidingWindowGenerator(seq_length=30, label_width=7, shift=1, df=df, label_columns=['wv(m/s)'])
print(swg)
# Generate windows
X, y = swg.sliding_windows()
print("-----------------------------")
print(X.shape)
print(y.shape)


~~~
{: .python}
~~~
Total window size: 31
Input indices: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25 26 27 28 29]
Label indices: [24 25 26 27 28 29 30]
Label column name(s): ['wv(m/s)']
-----------------------------
(420521, 30, 8)
(420521, 1)
~~~
{: .output}





### Normalize the data

To prepare the data for training the neural network, we first normalize it using MinMaxScaler to ensure that all features have a common scale. This step helps in accelerating the training process and improving the model's performance. The normalized data is then converted into PyTorch tensors and moved to the GPU if available. Finally, we create TensorDataset instances and split them into training and testing sets, which are loaded into DataLoader objects for efficient batch processing.

~~~

# Normalize the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_shape = X.shape
y_shape = y.shape

X_flat = X.reshape(-1, X_shape[-1])
y_flat = y.reshape(-1, y_shape[-1])

X = scaler_X.fit_transform(X_flat).reshape(X_shape)
y = scaler_y.fit_transform(y_flat).reshape(y_shape)

# train and test data loading in tensor format
train_size = int(len(y) * 0.7)
test_size = len(y) - train_size

X_train = Variable(torch.Tensor(np.array(X[0:train_size])))
y_train = Variable(torch.Tensor(np.array(y[0:train_size])))

X_test = Variable(torch.Tensor(np.array(X[train_size:len(X)])))
y_test = Variable(torch.Tensor(np.array(y[train_size:len(y)])))

# Create TensorDataset instances for training and testing data
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

# Initialize DataLoader objects for both datasets with batch size 256
batch_size = 256
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

~~~
{: .python}


### LSTM Model Architecture

The model architecture defines the structure of the LSTM neural network. In this section, we specify the number of input features, hidden units, layers, and output size of the LSTM model. The LSTM layer is a fundamental building block that processes sequential data by learning long-term dependencies. Additionally, we define a fully connected (dense) layer to map the LSTM output to the desired prediction output size.


~~~

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
~~~
{: .python}


This architecture allows the LSTM to learn and remember long-term dependencies in the input sequences, making it particularly effective for time series forecasting and sequential data analysis. The constructor initializes the LSTM model.

- input_size: The number of features in the input data.
- hidden_size: The number  hidden state in the model.
- num_layers: The number of stacked LSTM layers.
- output_size: The size of the output.

The LSTM  model also consists of layers followed by a fully connected (linear) layer. The  layer processes input sequences, maintaining hidden and cell states to capture long-term dependencies. Initialized with zero states, the LSTM layer produces an output for each time step. The fully connected layer then takes the output from the last time step of the LSTM and maps it to the final prediction. This structure enables the model to effectively handle sequential data, making it ideal for tasks like time series forecasting.



### GRU Model Architecture

The GRU (Gated Recurrent Unit) model comprises a GRU layer followed by a fully connected (linear) layer. The GRU layer is designed to capture sequential dependencies while addressing the vanishing gradient problem. It operates similarly to an LSTM but with fewer parameters, making it computationally efficient. In the provided architecture, the GRU layer takes input sequences and produces output at each time step, maintaining hidden states to retain information across time. The fully connected layer then maps the final hidden state to the output prediction. This streamlined architecture enables effective modeling of sequential data, making the GRU model suitable for tasks such as time series forecasting. The model structure of GRU is as given below:

~~~
# Define the GRU model
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out
~~~
{: .python}



### Training Process

The training process involves iteratively optimizing the LSTM model's parameters to minimize a predefined loss function. We utilize the Mean Squared Error (MSE) loss and the Adam optimizer for training. During each epoch, the model is trained on mini-batches of the training dataset, and the gradients are computed and updated using backpropagation. We monitor the training and testing losses to assess the model's performance and convergence.

~~~

# Check for GPU availability including CUDA and Apple's MPS GPU

device = torch.device('cuda:0' if torch.cuda.is_available() else 'mps:0' if torch.backends.mps.is_available() else 'cpu')
# Training the model
num_epochs = 20
learning_rate = 0.01
input_size = X.shape[2] # feature fecture 
hidden_size = 5
num_layers = 2
output_size = 1
lstm = LSTM(input_size, hidden_size, num_layers, output_size)

criterion = torch.nn.MSELoss()    # Mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

# Train the model
train_losses = []
test_losses = []
for epoch in range(num_epochs):
    # Train
    lstm.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = lstm(inputs)
        train_loss = criterion(outputs, targets)
        train_loss.backward()
        optimizer.step()
        train_losses.append(train_loss.item())
    
    # Test
    lstm.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            test_outputs = lstm(inputs)
            test_loss = criterion(test_outputs, targets)
            test_losses.append(test_loss.item())
    
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Train Loss: {np.mean(train_losses[-len(train_loader):]):.5f}, Test Loss: {np.mean(test_losses[-len(test_loader):]):.5f}")

~~~
{: .python}

~~~
Epoch: 0, Train Loss: 0.57703, Test Loss: 0.23651
Epoch: 5, Train Loss: 0.00306, Test Loss: 0.00215
Epoch: 10, Train Loss: 0.00003, Test Loss: 0.00098
Epoch: 15, Train Loss: 0.00000, Test Loss: 0.00097
Epoch: 20, Train Loss: 0.00000, Test Loss: 0.00097
~~~
{: .output}



### Model Evaluation


After training the LSTM model, we evaluate its performance on both the training and testing datasets. We compute metrics such as Mean Squared Error (MSE) and R² score to quantify the model's predictive accuracy. By comparing the model's predictions with the actual values, we gain insights into its ability to capture the underlying patterns in the time series data.


~~~
# Compute final MSE and R² for train and test sets
train_predict = lstm(X_train).data.numpy()
test_predict = lstm(X_test).data.numpy()
trainY_plot = y_train.data.numpy()
testY_plot = y_test.data.numpy()
train_predict = scaler_y.inverse_transform(train_predict)
trainY_plot = scaler_y.inverse_transform(trainY_plot)
test_predict = scaler_y.inverse_transform(test_predict)
testY_plot = scaler_y.inverse_transform(testY_plot)
train_mse = mean_squared_error(trainY_plot, train_predict)
train_r2 = r2_score(trainY_plot, train_predict)
test_mse = mean_squared_error(testY_plot, test_predict)
test_r2 = r2_score(testY_plot, test_predict)
# Plot the training and testing loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Testing Loss Over Epochs')
# Add MSE and R² values as annotations
plt.text(0.5, 0.9, f'MSE: {train_mse:.5f}', ha='center', va='center', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
plt.text(0.5, 0.8, f'R²: {train_r2:.5f}', ha='center', va='center', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
plt.tight_layout()
plt.show()

~~~
{: .python}


These sections and corresponding code snippets provide a comprehensive guide to implementing and evaluating a multi-variable time series prediction model using LSTM networks. Each step is essential for understanding and applying LSTM-based forecasting techniques effectively.


> ## Exercise: How to Improve the Model Performance?
> - Modify the sequence length to 50 used in the LSTM model and observe its impact on the model's performance.
> - Modify the hidden layers of the LSTM model to 10 and train the model again. Plot the observed vs prediction for the test dataset.
> 
> > ## Solution
> > ~~~
> > input_size = X.shape[2]
> > hidden_size = 10  # Modify hidden layers to 10
> > num_layers = 2
> > output_size = 1
> > lstm = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
> > criterion = torch.nn.MSELoss()
> > optimizer = torch.optim.Adam(lstm.parameters(), lr=0.01)
> > num_epochs = 2000
> > train_losses = []
> > test_losses = []
> >for epoch in range(num_epochs):
> >    lstm.train()
> >    for inputs, targets in train_loader:
> >        inputs, targets = inputs.to(device), targets.to(device)
> >        optimizer.zero_grad()
> >        outputs = lstm(inputs)
> >        train_loss = criterion(outputs, targets)
> >        train_loss.backward()
> >        optimizer.step()
> >        train_losses.append(train_loss.item())
> >    
> >    lstm.eval()
> >    with torch.no_grad():
> >        for inputs, targets in test_loader:
> >            inputs, targets = inputs.to(device), targets.to(device)
> >            test_outputs = lstm(inputs)
> >            test_loss = criterion(test_outputs, targets)
> >            test_losses.append(test_loss.item())
> >    
> >    if epoch % 100 == 0:
> >        print(f"Epoch: {epoch}, Train Loss: {np.mean(train_losses[-len(train_loader):]):.5f}, Test Loss: {np.mean(test_losses[-len(test_loader):]): .5f}")
> >
> > train_predict = lstm(X_tensor[:train_size].to(device)).cpu().detach().numpy()
> > test_predict = lstm(X_tensor[train_size:].to(device)).cpu().detach().numpy()
> > 
> > train_predict = scaler_y.inverse_transform(train_predict.reshape(-1, 1))
> > test_predict = scaler_y.inverse_transform(test_predict.reshape(-1, 1))
> >
> > train_mse = mean_squared_error(y[:train_size], train_predict)
> > test_mse = mean_squared_error(y[train_size:], test_predict)
> >
> > train_r2 = r2_score(y[:train_size], train_predict)
> > test_r2 = r2_score(y[train_size:], test_predict)
> >
> > plt.figure(figsize=(10, 5))
> > plt.plot(train_losses, label='Train Loss')
> > plt.plot(test_losses, label='Test Loss')
> > plt.xlabel('Epoch')
> > plt.ylabel('Loss')
> > plt.legend()
> > plt.title('Training and Testing Loss Over Epochs')
> > plt.text(0.5, 0.9, f'Train MSE: {train_mse:.5f}', ha='center', va='center', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
> > plt.text(0.5, 0.8, f'Test MSE: {test_mse:.5f}', ha='center', va='center', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
> > plt.text(0.5, 0.7, f'Train R²: {train_r2:.5f}', ha='center', va='center', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
> > plt.text(0.5, 0.6, f'Test R²: {test_r2:.5f}', ha='center', va='center', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
> > plt.tight_layout()
> > plt.show()
> > plt.figure(figsize=(12, 6))
> > plt.plot(df_daily.index[train_size:], scaler_y.inverse_transform(y[train_size:]), label='Actual')
> > plt.plot(df_daily.index[train_size:], test_predict, label='Predicted')
> > plt.xlabel('Date')
> > plt.ylabel('wv(m/s)')
> > plt.title('Observed vs Predicted Temperature')
> > plt.legend()
> > plt.show()
> > ~~~
> > {: .python}
> {: .solution}
>
{: .challenge}




> ## Exercise: Forecasting with GRU Model 
> - Modify the sequence length to 50 used in the LSTM model and observe its impact on the model's performance.
> - Modify the hidden layers of the LSTM model to 10 and train the model again. Plot the observed vs prediction for the test dataset.
> 
> > ## Solution
> > ~~~
> > # Define the GRU model
> > class GRU(nn.Module):
> >     def __init__(self, input_size, hidden_size, num_layers, output_size):
> >         super(GRU, self).__init__()
> >         self.hidden_size = hidden_size
> >         self.num_layers = num_layers
> >         self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
> >         self.fc = nn.Linear(hidden_size, output_size)
> > 
> >     def forward(self, x):
> >         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
> >         out, _ = self.gru(x, h0)
> >         out = self.fc(out[:, -1, :])
> >         return out
> > input_size = X.shape[2]
> > hidden_size = 8  # Modify hidden layers to 8
> > num_layers = 2
> > output_size = 1
> > gru = GRU(input_size, hidden_size, num_layers, output_size).to(device)
> > 
> > criterion = torch.nn.MSELoss()
> > optimizer = torch.optim.Adam(gru.parameters(), lr=0.01)
> > num_epochs = 2000
> > train_losses = []
> > test_losses = []
> > for epoch in range(num_epochs):
> >     gru.train()
> >     for inputs, targets in train_loader:
> >         inputs, targets = inputs.to(device), targets.to(device)
> >         optimizer.zero_grad()
> >         outputs = gru(inputs)
> >         train_loss = criterion(outputs, targets)
> >         train_loss.backward()
> >         optimizer.step()
> >         train_losses.append(train_loss.item())
> >     
> >     gru.eval()
> >     with torch.no_grad():
> >         for inputs, targets in test_loader:
> >             inputs, targets = inputs.to(device), targets.to(device)
> >             test_outputs = gru(inputs)
> >             test_loss = criterion(test_outputs, targets)
> >             test_losses.append(test_loss.item())
> >     
> >     if epoch % 100 == 0:
> >         print(f"Epoch: {epoch}, Train Loss: {np.mean(train_losses[-len(train_loader):]):.5f}, Test Loss: {np.mean(test_losses[-len(test_loader):]): .5f}")
> > 
> > train_predict = gru(X_tensor[:train_size].to(device)).cpu().detach().numpy()
> > test_predict = gru(X_tensor[train_size:].to(device)).cpu().detach().numpy()
> > 
> > train_predict = scaler_y.inverse_transform(train_predict.reshape(-1, 1))
> > test_predict = scaler_y.inverse_transform(test_predict.reshape(-1, 1))
> > 
> > train_mse = mean_squared_error(y[:train_size], train_predict)
> > test_mse = mean_squared_error(y[train_size:], test_predict)
> > 
> > train_r2 = r2_score(y[:train_size], train_predict)
> > test_r2 = r2_score(y[train_size:], test_predict)
> > 
> > plt.figure(figsize=(10, 5))
> > plt.plot(train_losses, label='Train Loss')
> > plt.plot(test_losses, label='Test Loss')
> > plt.xlabel('Epoch')
> > plt.ylabel('Loss')
> > plt.legend()
> > plt.title('Training and Testing Loss Over Epochs')
> > plt.text(0.5, 0.9, f'Train MSE: {train_mse:.5f}', ha='center', va='center', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
> > plt.text(0.5, 0.8, f'Test MSE: {test_mse:.5f}', ha='center', va='center', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
> > plt.text(0.5, 0.7, f'Train R²: {train_r2:.5f}', ha='center', va='center', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
> > plt.text(0.5,
 0.6, f'Test R²: {test_r2:.5f}', ha='center', va='center', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
> > plt.tight_layout()
> > plt.show()
> > 
> > plt.figure(figsize=(12, 6))
> > plt.plot(df_daily.index[train_size:], scaler_y.inverse_transform(y[train_size:]), label='Actual')
> > plt.plot(df_daily.index[train_size:], test_predict, label='Predicted')
> > plt.xlabel('Date')
> > plt.ylabel('T(degC)')
> > plt.title('Observed vs Predicted Temperature')
> > plt.legend()
> > plt.show()
> > ~~~
> > {: .python}
> {: .solution}
> 
{: .challenge}



## Transformers for Timeseries

Time series analysis for the prediction of multi-variable  data stands as a significant challenge, crucial for various applications ranging from finance to weather forecasting. Leveraging cutting-edge techniques like Transforms for Time Series (TFT), researchers and practitioners aim to enhance predictive accuracy and uncover intricate temporal patterns embedded within multidimensional data streams.


This sectionfocuses on leveraging Transforms for Time Series (TFT) to predict multi-variable time series data, a critical task in various domains such as finance, healthcare, and environmental science. Through a series of concise code snippets and explanations, participants will gain a solid understanding of implementing TFT for accurate forecasting. The goal  is to illustrate the use of a transformer for timeseries prediction. 

###  Preparing the Dataset
~~~
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as  plt
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# File path for the energy dataset
file_path = "data/energydata_complete.csv"

df = pd.read_csv(file_path, index_col='date', parse_dates=['date'])
df.head()
~~~
{: .python}





### Sliding Window Generator

~~~

# Initialize the generator
# if label_width=1 it will be single-step forecasting
swg = SlidingWindowGenerator(seq_length=30, label_width=7, shift=1, df=df, label_columns=['rv2']])
print(swg)
# Generate windows
X, y = swg.sliding_windows()
print("-----------------------------")
print(X.shape)
print(y.shape)


~~~
{: .python}
~~~
Total window size: 31
Input indices: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25 26 27 28 29]
Label indices: [28 29 30]
Label column name(s): ['rv2']

--------------------------
(420521, 30, 8)
(420521, 1)

~~~
{: .output}

