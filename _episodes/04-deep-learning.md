---
title: Deep Learing Fundamentals
teaching: 1
exercises: 0
questions:
- "What are the basic timeseries I can use in pandas ?"
- "How do I write documentation for my Python code?"
- "How do I install and manage packages?"
objectives:
- "Brief overview of basic datatypes like lists, tuples, & dictionaries."
- "Recommendations for proper code documentation."
- "Installing, updating, and importing packages."
- "Verify that everyone's Python environment is ready."
keypoints:
- "Deep Learning algorithms are often represented as graph computation"
- "We have different non-linear activation functions that help in learning different relationships to solve handle non-linearity in nn problems."
- ""
---
<!-- MathJax -->

<script type="text/javascript"

  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML">

</script>



 Advances in computational power, experimental techniques, and simulations are producing vast quantities of data across fields like particle physics, cosmology, atomospherica science, materials science, and quantum computing. However, traditional analysis methods struggle to keep up with the scale and complexity of this "big data". Machine learning algorithms excel at finding patterns and making predictions from large, high-dimensional datasets. By training on this wealth of data, ML models can accelerate scientific discovery, optimize experiments, and uncover hidden insights that would be difficult for humans to discern alone. As the pace of data generation continues to outstrip human processing capabilities, machine learning will only grow in importance for driving progress in the physical sciences.


Ultimately, machine learning represents a valuable addition to the climate scientist's toolbox, but should be applied to gain the most robust insights about the Physical sciences.



##  Neural Networks

* Deep Learning algorithms are often represented as graph computation.
* Some of the activities to build high level neural network are as follows:
     * Building a data pipeline
     * Building a network architecture
     * Evaluating the architecture using a lost function
     * Optimizing the network architurcture weights using an optimization algorithm

### Layer
Layers are the building block of neural network. **Linear layers** are called by different names, such as **dense or fully connected** layers across different frameworks. This model in nn has the form:

\\[ y = \sigma(w\cdot x+ b) \\]

where 

1.   y is the predict variable
2.   x is the predictor variable
3.   b is the bias and
4.   w is the weights on the neural network
5.   \\(\sigma\\) is the activation function



ANNs consists of multiple nodes (the circles) and layers that are all connected and using basic math gives out a result. These are called feed forward networks. 

<img src="../fig/ANN_forward.png" width="500">


In each individual node the values coming in are weighted and summed together and bias term is added

~~~
import torch
from torch.autograd import Variable

inp = Variable(torch.randn(1,10)) # input data
model = nn.Linear(in_features=10,out_features=5,bias=True) ## linear model layer
model(inp)
model.weight
~~~
{: .python}
 
~~~
Parameter containing:
tensor([[ 0.3034,  0.2425, -0.1914, -0.2280, -0.3050,  0.0394,  0.0196,  0.2530,
          0.1539,  0.1212],
        [ 0.2260,  0.2431,  0.0817, -0.0612,  0.1539, -0.1220, -0.2194,  0.1102,
          0.2031, -0.1362],
        [-0.2060,  0.0617, -0.2007, -0.2809, -0.2511, -0.2009,  0.1967,  0.0988,
          0.0728, -0.0911],
        [ 0.0710,  0.2536, -0.1963,  0.2167,  0.2653, -0.1034, -0.1948,  0.2978,
          0.0614, -0.0122],
        [ 0.2486,  0.0924, -0.1496, -0.2745,  0.1828, -0.0443, -0.1161,  0.2778,
          0.1709, -0.1165]], requires_grad=True)
~~~
{: .output}

~~~
# Bias of the model
myLinear.bias
~~~
{: .python}

~~~
Parameter containing:
tensor([-0.1909,  0.2449,  0.1723,  0.0486,  0.2384], requires_grad=True)
~~~
{: .output}

<img src="../fig/ANN_activation.png" width="500">



### Activation functions

Activation function determines, if information is moving forward from that specific node.
This is the step that allows for nonlinearity in these algorithms, without activation all we would be doing is linear algebra. Some of the common activation functions are indicated in figure below:



<img src="../fig/ANN_activation2.png" width="500">


We have different **non-linear activation functions** that help in learning different relationships to solve handle non-linearity in nn problems. There are many different non-linear activation functions available in deep learning.These are:

1. Sigmoid
2. Tanh
3. ReLU
4. Leaky ReLU


So training of the network is merely determining the weights "w" and bias/offset "b"  with the addition of nonlinear activation function. Goal is to determine the best function so that the output is as  correct as possible; typically involves choosing "weights". 



### Sigmod function

When the output of the sigmoid function is close to zero or one, the gradients for the layers before the sigmoid function are close to zero and, hence, the learnable parameters of the previous layer get gradients close to zero and the weights do not get adjusted often, resulting in dead neurons. The mathematical form of sigmoid activation function is:

\\[ \delta(x) = \frac{1}{1 + e^{-x}} \\]

~~~
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
x = np.linspace(-10,10)
y = 1/(1+np.exp(x))
plt.plot(x,y)
plt.show()
~~~
{: .python}


#### Tanh

The tanh non-linearity function squashes a real-valued number in the range of -1 and 1. The tanh also faces the same issue of saturating gradients when tanh outputs extreme values close to -1 and 1.

~~~
x = np.linspace(-10,10)
y = np.tanh(x)
plt.plot(x,y)
plt.show()
~~~
{: .python}


#### ReLU 

ReLU has become more popular in the recent years; we can find either its usage or one of its
variants' usages in almost any modern architecture. It has a simple mathematical
formulation:
 \\[f(x) = max(x,0)\\]

#### Leaky ReLU

Leaky ReLU is an attempt to solve a dying problem where, instead of saturating to zero, we saturate to a very small number such as 0.001.

\\[f(x) = max(x,0.001)\\]


#### Loss Function

You know the data and the goal you're working towards, so you know the best, which loss function to use. Basic MSE or MAE works well for regression tasks. The basic MSE and MAE works well for regression task is given by:


\\[ \text{Loss} = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}_i - y_{i})^2 \\]


The quantinty you want ot determine("loss") help to determine the best weights and bias terms in the model. Gradient descent is a technique to find the weight that minimizes the loss function.  This is done by starting with a random point, the gradient (the black lines) is calculated at that point. Then the negative of that gradient is followed to the next point and so on. This is repeated until the minimum is reached.


<img src="../fig/loss_function.png" width="500">

The gradeint descent formula tells us that the next location depends on the negative gradient of J multiplied by the learning rate \\(\lambda\\).

\\[ J_{i+1} = J_{i} - \lambda \nabla J_{t} \\]


As the loss function depends on the linear function and its weights \(w_0\) and \(w_1\), the gradient is calculated as parital derviatives with relation to the weights.


<img src="../fig/loss_function2.png" width="500">


The only other thing one must pay attention to is the learning rate \\(lambda\\) (how big of a step to take). Too small and finding the right weights takes forever, too big and you might miss the minimum.

\\[ w_{i+1} = w_i - \lambda \frac{\partial J}{\partial w_i} \\]


Backpropagation is a technique used to compute the gradient of the loss function when its functional form is unknown. This method calculates the gradient with respect to the neural network's weights, allowing for the optimization of these weights to minimize the loss. A critical requirement for the activation functions in this process is that they must be differentiable, as this property is essential for the gradient computation necessary in backpropagation.

\\[ \frac{\partial J}{\partial w_k} = \frac{\partial}{\partial w_k}\left( \frac{1}{2m}\sum_{i=1}^{m}(\hat{y}_i - y_i)^2 \right) = \frac{1}{m}\sum_{i=1}^{m}(\hat{y}_i - y_i) \frac{\partial \hat{y}_i}{\partial w_{k}} \\]



## ANN Model

Let's start with our imports. Here we are importing Pytorch and calling it tf for ease of use. We then import a library called numpy, which helps us to represent our data as lists easily and quickly. The framework for defining a neural network as a set of Sequential layers is called keras, so we import that too.

~~~
 Import necessary libraries

# Import PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
omport matplotlib.pyplot as plt
import pandas as pd
# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

~~~
{: .python}

### Getting Started With Pytorch


Welcome to the world of PyTorch! This section serves as a comprehensive introduction to PyTorch, a powerful deep learning framework widely used for research and production. Whether you're new to deep learning or transitioning from another framework, this guide will help you get started with PyTorch's basics.

#### Initializing Tensors

Tensors are the fundamental building blocks of PyTorch. They are similar to NumPy arrays but come with additional features optimized for deep learning tasks. Let's begin by understanding how to create and manipulate tensors.

~~~
import torch

# Initialize a tensor of size 5x3 filled with zeros
x = torch.Tensor(5, 3)
print(x)
~~~
{: .python}


In the above code snippet, we create a 5x3 tensor initialized with zeros using the `torch.Tensor` constructor.

~~~
tensor([[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]])
~~~
{: .output}


~~~
# Create a tensor of size 5x4 with random values sampled from a uniform distribution
x = torch.rand(5, 4)
print(x)
~~~
{: .python}

Here, we create a 5x4 tensor filled with random values sampled from a uniform distribution using the `torch.rand` function.


~~~
tensor([[0.4294, 0.8854, 0.5739, 0.2666],
        [0.6274, 0.2696, 0.4414, 0.2969],
        [0.8317, 0.1053, 0.2695, 0.3588],
        [0.1994, 0.5472, 0.0062, 0.9516],
        [0.0753, 0.8860, 0.5832, 0.3376]])
~~~
{: .output}

#### Basic Tensor Operations

PyTorch supports a wide range of tensor operations, making it easy to perform computations on tensors. Let's explore some common operations.

~~~
# Element-wise addition of two tensors
x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
y = torch.tensor([5, 6, 7, 8], dtype=torch.float32)

result_add = x + y
print(result_add)
~~~
{: .python}

In this snippet, we perform element-wise addition of two tensors `x` and `y`.

~~~
tensor([ 6.,  8., 10., 12.])
~~~
{: .output}


~~~
# Matrix multiplication (dot product)
matrix1 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
matrix2 = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)
result_matmul = torch.mm(matrix1, matrix2)
print(result_matmul)
~~~
{: .python}

Here, we calculate the matrix multiplication between two tensors `matrix1` and `matrix2`.


~~~
tensor([[19., 22.],
        [43., 50.]])
~~~
{: .output}

#### Reshaping Tensors

Sometimes, we need to reshape tensors to match the required input shapes for neural networks. Let's see how to reshape tensors.

~~~
x_reshape = x.view(1, 4)
print(x_reshape.shape)
print(x.shape)
~~~
{: .python}

This code reshapes the tensor `x` into a 1x4 matrix and prints the shapes of the reshaped and original tensors.

~~~
torch.Size([1, 4])
torch.Size([4])
~~~
{: .output}


```python
result_matmul3 = torch.matmul(matrix1.view(1,-1), matrix2.view(-1,1))
print(result_matmul3)
```

Here, we compute the matrix multiplication between two tensors with reshaping.

~~~
tensor([[70.]])
~~~


#### GPU Acceleration (if available)

PyTorch provides support for GPU acceleration, which significantly speeds up computations for deep learning tasks. Let's explore how to leverage GPU acceleration if available. This code snippet checks for GPU availability and performs tensor addition either on CUDA, Metal, or CPU. It demonstrates PyTorch's flexibility in utilizing available hardware resources for computation.



~~~
if torch.cuda.is_available():
    device = torch.device("cuda")
    x_cuda = x.to(device)
    y_cuda = y.to(device)
    result_add_cuda = x_cuda + y_cuda
    print("The Cuda addition:",result_add_cuda)
elif torch.backends.mps.is_available(): 
    device = torch.device("mps")
    x_mps = x.to(device)
    y_mps = y.to(device)
    result_add_mps = x_mps + y_mps
    print("The MPS addition:",result_add_mps)
else:
    device = torch.device("cpu")
    x_cpu = x.to(device)
    y_cpu = y.to(device)
    result_add_cpu = x_cpu + y_cpu
    print("Using CPU addition:",result_add_cpu)
~~~
{: .python}

~~~
Using MPS addition: tensor([ 6.,  8., 10., 
~~~
{: .output}

### Computational Graph

PyTorch uses a dynamic computational graph, which means that the graph is built on-the-fly as operations are performed. This dynamic nature makes it easy to work with variable-sized inputs and dynamic control flow, unlike static computational graphs used by some other deep learning frameworks like TensorFlow 1.x.


~~~
import torch
import torch.nn as nn
import torchviz
from torch.autograd import Variable

# Define some input data
x = Variable(torch.randn(1, 2), requires_grad=True)

# Define a simple computation
y = x *  2 
z = y.sum()

# Visualize the computation graph
dot = torchviz.make_dot(z, params={"x": x})
dot.render("computational_graph", format="png")

# Print the computation graph
#print(dot)
~~~
{: .python}


The computational graph is dynamic and depends on the actual operations performed during execution. You can create more complex graphs by composing various operations and functions. When you perform backward propagation (backward()), PyTorch automatically computes gradients and updates the model parameters based on this dynamic graph.

~~~
import torch
import torch.autograd
import torchviz

# Create tensors in PyTorch
x = torch.tensor(2.0, dtype=torch.float32, requires_grad=True)
y = torch.tensor(3.0, dtype=torch.float32, requires_grad=True)

# Perform multiple operations
a = x * y
b = torch.sin(a)
c = torch.exp(b)
d = c / (x + y)

# Manually create a PyTorch computation graph
d.backward()

# Visualize the entire computational graph
dot = torchviz.make_dot(d, params={"x": x, "y": y, "a": a, "b": b, "c": c})
dot.render("computational_graph2", format="png")

# Print the results directly
print("x:", x.item())
print("y:", y.item())
print("d:", d.item())

~~~
{: .python}

~~~
x: 2.0
y: 3.0
d: 0.1512451320886612
~~~
{: .output}

![](../fig/omputational_graph2.png)


## Building Artificial Neural Networks Model

In this session, we aim to create an Artificial Neural Network (ANN) that learns the relationship between a set of input features (Xs) and corresponding output labels (Ys). This process involves several steps outlined below:

1. **Instantiate a Sequential Model**: We begin by creating a Sequential model, which allows us to stack layers one after the other.

2. **Build the Input and Hidden Layers**: Following the architecture depicted in the provided diagram:

   - We start with an input layer, which receives the input features (Xs) and passes them to the subsequent layers.
   - Next, we add a hidden layer, where the network performs transformations on the input data to learn relevant patterns and representations.

3. **Add the Output Layer**: Finally, we incorporate the output layer, which produces the predicted outputs based on the learned relationships from the input data.

By systematically following these steps, we construct a sequential neural network capable of discerning and modeling the underlying relationship between the input features and output labels.


<div> <img src="../fig/ANN2.png" alt="Drawing" style="width: 500px;"/></div>


~~~
import torch
import torch.nn as nn

# Define the neural network architecture
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.hidden_layer = nn.Linear(1, 3)  # Input size: 1, Output size: 3
        self.output_layer = nn.Linear(3, 1)  # Input size: 3, Output size: 1

    def forward(self, x):
        x = torch.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x

# Instantiate the model
model = ANN()
print(model)
~~~
{: .python}


~~~
ANN(
  (hidden_layer): Linear(in_features=1, out_features=3, bias=True)
  (output_layer): Linear(in_features=3, out_features=1, bias=True)
)
~~~

In PyTorch, you define your neural network architecture by subclassing nn.Module and implementing the `__init__` and forward methods. In this code, we define a simple neural network with one hidden layer and one output layer. The nn.Linear module represents a fully connected layer, and torch.relu is the rectified linear unit activation function. Finally, we instantiate the model and print its structure.


This code defines the original model ANN and then converts it into a sequential format using `nn.Sequential`. Each layer is added sequentially with the appropriate input and output sizes, along with activation functions where necessary. Finally, it prints both the original and sequential models for comparison.
~~~
# Convert to Sequential format
sequential_ANN = nn.Sequential(
    nn.Linear(1, 3),  # Input size: 1, Output size: 3
    nn.ReLU(),
    nn.Linear(3, 1)   # Input size: 3, Output size: 1
)
print("\nSequential Model:\n", sequential_ANN)
~~~


### Regression Model with Neural Networks

Because it is not directly compatible with PyTorch, we cannot simply feed the data to our PyTorch neural network. For doing so, it needs to be prepared. This is actually quite easy: we can create a PyTorch Dataset for this purpose.

~~~
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split 
from sklearn.datasets import fetch_california_housing 
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset 
from tqdm.notebook import tqdm
import warnings
import seaborn as sns
sns.set()
warnings.filterwarnings("ignore")
# Set fixed random number seed
torch.manual_seed(42);

dataset = fetch_california_housing()
print(dataset.data)
~~~
{: .python}

~~~
array([[   8.3252    ,   41.        ,    6.98412698, ...,    2.55555556,
          37.88      , -122.23      ],
       [   8.3014    ,   21.        ,    6.23813708, ...,    2.10984183,
          37.86      , -122.22      ],
       [   7.2574    ,   52.        ,    8.28813559, ...,    2.80225989,
          37.85      , -122.24      ],
       ...,
       [   1.7       ,   17.        ,    5.20554273, ...,    2.3256351 ,
          39.43      , -121.22      ],
       [   1.8672    ,   18.        ,    5.32951289, ...,    2.12320917,
          39.43      , -121.32      ],
       [   2.3886    ,   16.        ,    5.25471698, ...,    2.61698113,
          39.37      , -121.24      ]])
~~~
{: .output}


When training on a machine that has a GPU, you need to tell PyTorch you want to use it • You’ll see the following at the top of most PyTorch code:

~~~
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
~~~
{: .python}


### Representing the Dataset as Tensor

Because it is not directly compatible with PyTorch, we cannot simply feed the data to our PyTorch neural network. For doing so, it needs to be prepared. This is actually quite easy: we can create a PyTorch Dataset for this purpose.

~~~
class MLP(nn.Module): 
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(8, 24)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(24, 12)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(12, 6)
        self.relu3 = nn.ReLU()
        self.layer4 = nn.Linear(6, 1)
    def forward(self, x): 
        x = self.layer1(x) 
        x = self.relu1(x) 
        x = self.layer2(x) 
        x = self.relu2(x) 
        x = self.layer3(x) 
        x = self.relu3(x) 
        x = self.layer4(x) 
        return x

model = MLP()
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
# Check the selected device
print("Selected device:", device)
model.to(device)
~~~
{: .python}

~~~
Selected device: mps
Out[13]: 
MLP(
  (layer1): Linear(in_features=8, out_features=24, bias=True)
  (relu1): ReLU()
  (layer2): Linear(in_features=24, out_features=12, bias=True)
  (relu2): ReLU()
  (layer3): Linear(in_features=12, out_features=6, bias=True)
  (relu3): ReLU()
  (layer4): Linear(in_features=6, out_features=1, bias=True)
)
~~~
{: .output}

The above output describes the architecture of a Multi-Layer Perceptron (MLP) model defined using PyTorch. This model is intended for use on an Apple device with a Metal Performance Shaders (MPS) backend, as indicated by "Selected device: mps". 


The following code snippet demonstrates how to read data into a pandas DataFrame and display the first few rows of the dataset. This is a common practice in data preprocessing and exploration. Let's break down each part of the code:

~~~
# Read data
df = pd.DataFrame(dataset.data, columns=[dataset.feature_names])
df.head()
~~~
{: .python}


~~~
   MedInc HouseAge  AveRooms AveBedrms Population  AveOccup Latitude Longitude
0  8.3252     41.0  6.984127  1.023810      322.0  2.555556    37.88   -122.23
1  8.3014     21.0  6.238137  0.971880     2401.0  2.109842    37.86   -122.22
2  7.2574     52.0  8.288136  1.073446      496.0  2.802260    37.85   -122.24
3  5.6431     52.0  5.817352  1.073059      558.0  2.547945    37.85   -122.25
4  3.8462     52.0  6.281853  1.081081      565.0  2.181467    37.85   -122.25
~~~
{: .output}


~~~
df["y"] = dataset.target
X = df.drop('y',axis=1)
y = df['y'].values
y = y.reshape(-1,1)


# train-test split for model evaluation
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

# Standardizing data
scaler = StandardScaler()
scaler.fit(X_train_raw)
X_train = scaler.transform(X_train_raw)
X_test = scaler.transform(X_test_raw)
# Convert to 2D PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
~~~
{: .python}


Let's  demonstrates how to train a neural network in PyTorch using a Mean Squared Error (MSE) loss function and the Adam optimizer. The key steps involve initializing the model, defining the loss function and optimizer, iterating over multiple epochs, and processing data in batches. The code also tracks the best model based on the validation loss and restores the best weights after training. Additionally, it plots the training loss over epochs for visualization.

1. **Setup:** Define the model, loss function, and optimizer.
2. **Training Loop:** Iterate over multiple epochs, processing data in batches, and updating model weights.
3. **Evaluation:** Calculate the validation loss at the end of each epoch to track the model's performance.
4. **Best Model Tracking:** Save and restore the best model weights based on validation loss.
5. **Plotting:** Visualize the training loss over epochs.

~~~
# Loss function and optimizer
loss_fn = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.Adam(model.parameters(), lr=0.0001)

n_epochs = 100  # Number of epochs to run
batch_size = 10  # Size of each batch
batch_start = torch.arange(0, len(X_train), batch_size)
# Hold the best model
best_mse = np.inf  # Initialize to infinity
best_weights = None
history = []

for epoch in range(n_epochs):
    model.train()
    with tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
        bar.set_description(f"Epoch {epoch}")
        for start in bar:
            # Take a batch
            X_batch = X_train[start:start+batch_size].to(device)
            y_batch = y_train[start:start+batch_size].to(device)
            # Forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            # Update weights
            optimizer.step()
            # Print progress
            bar.set_postfix(mse=float(loss.item()))
    # Evaluate accuracy at end of each epoch
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test.to(device))
        mse = loss_fn(y_pred, y_test.to(device))
        mse = float(mse)
        history.append(mse)
        if mse < best_mse:
            best_mse = mse
            best_weights = copy.deepcopy(model.state_dict())

# Restore model to best weights and print final accuracy
model.load_state_dict(best_weights)
print("MSE: %.2f" % best_mse)
print("RMSE: %.2f" % np.sqrt(best_mse))

# Plot training history
plt.plot(history)
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Training Loss')
plt.savefig("fig/house_train_loss.png")
plt.show()
~~~
{: .python}

![](../fig/house_train_loss.png)


> ## Exercise: Training a Neural Network with PyTorch
> In this exercise, we will enhance the neural network training process with PyTorch by implementing the following steps:
> 
> 1. Apply LeakyReLU and tanh activation functions and compare the results with the previous implementation.
> 2. Introduce a dropout rate of 0.2 between the layers to improve model generalization.
> 3. Experiment with different optimizers listed at the end of the slide (MLP).
> 4. Implement torch DataLoader techniques to handle batch processing for large datasets.
> 
> > ## Solution
> > 
> > ~~~
> > import torch
> > import torch.nn as nn
> > import torch.optim as optim
> > import numpy as np
> > import copy
> > from tqdm import tqdm
> > import matplotlib.pyplot as plt
> > from torch.utils.data import DataLoader, TensorDataset
> > 
> > # Define the neural network architecture
> > class NeuralNetwork(nn.Module):
> >     def __init__(self):
> >         super(NeuralNetwork, self).__init__()
> >         self.layer1 = nn.Linear(8, 24)
> >         self.leakyrelu1 = nn.LeakyReLU()
> >         self.dropout1 = nn.Dropout(0.2)
> >         self.layer2 = nn.Linear(24, 12)
> >         self.leakyrelu2 = nn.LeakyReLU()
> >         self.dropout2 = nn.Dropout(0.2)
> >         self.layer3 = nn.Linear(12, 6)
> >         self.leakyrelu3 = nn.LeakyReLU()
> >         self.dropout3 = nn.Dropout(0.2)
> >         self.layer4 = nn.Linear(6, 1)
> > 
> >     def forward(self, x):
> >         x = self.dropout1(self.leakyrelu1(self.layer1(x)))
> >         x = self.dropout2(self.leakyrelu2(self.layer2(x)))
> >         x = self.dropout3(self.leakyrelu3(self.layer3(x)))
> >         x = self.layer4(x)
> >         return x
> > 
> > # Define loss function and optimizers
> > loss_fn = nn.MSELoss()  # Mean Squared Error loss
> > optimizers = [optim.Adam, optim.SGD, optim.RMSprop]  # Different optimizers to try
> > n_epochs = 100  # Number of epochs to run
> > batch_size = 64  # Size of each batch
> > 
> > # Create DataLoader
> > train_dataset = TensorDataset(X_train, y_train)
> > test_dataset = TensorDataset(X_test, y_test)
> > train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
> > test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
> >
> > device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
> > for optimizer_class in optimizers:
> >     # Instantiate the model and move it to the device
> >     model = NeuralNetwork().to(device)
> >     optimizer = optimizer_class(model.parameters(), lr=0.0001)
> >     best_mse = np.inf  # Initialize to infinity
> >     best_weights = None
> >     history = []
> >
> >     # Training loop
> >     for epoch in range(n_epochs):
> >         model.train()
> >         for X_batch, y_batch in tqdm(train_loader, unit="batch", mininterval=0, disable=True, desc=f"Epoch {epoch}"):
> >             # Move batch data to device
> >             X_batch, y_batch = X_batch.to(device), y_batch.to(device)
> >             optimizer.zero_grad()
> >             y_pred = model(X_batch)
> >             loss = loss_fn(y_pred, y_batch)
> >             loss.backward()
> >             optimizer.step()
> >
> >         # Evaluate accuracy at the end of each epoch
> >         model.eval()
> >         with torch.no_grad():
> >             mse = 0
> >             for X_batch, y_batch in test_loader:
> >                 X_batch, y_batch = X_batch.to(device), y_batch.to(device)
> >                 y_pred = model(X_batch)
> >                 mse += loss_fn(y_pred, y_batch).item()
> >             mse /= len(test_loader)
> >
> >         history.append(mse)
> >         if mse < best_mse:
> >             best_mse = mse
> >             best_weights = copy.deepcopy(model.state_dict())
> >
> >     # Restore model and return best accuracy
> >     model.load_state_dict(best_weights)
> >     print(f"Optimizer: {optimizer_class.__name__}, Best MSE: {best_mse:.2f}, RMSE: {np.sqrt(best_mse):.2f}")
> >
> >     # Plot training history
> >     plt.plot(history, label=optimizer_class.__name__)
> >
> > plt.xlabel("Epochs")
> > plt.ylabel("MSE")
> > plt.legend()
> > plt.savefig("fig/house_train_loss.png")
> > plt.show()
> > ~~~
> > {: .python}
> {: .solution}
{: .challenge}

the R² score provides a measure of how well the regression model explains the variance in the target variable relative to a baseline model (usually the mean of the target variable). The R2 (coefficient of determination) is calculated as follows:

1. **Total Sum of Squares (TSS)**: This represents the total variance in the target variable (y). It is calculated as the sum of squared differences between each observed value and the mean of all observed values.
   \\[ \text{TSS} = \sum_{i=1}^{n}(y_{i} - \bar{y})^2 \\]

2. **Residual Sum of Squares (RSS)**: This represents the unexplained variance in the target variable after fitting the regression model. It is calculated as the sum of squared differences between each observed value and its corresponding predicted value.
   \\[ \text{RSS} = \sum_{i=1}^{n}(y_{i} - \hat{y}_{i})^2 \\] 
3. **R2 Score**: The R2 score is then calculated as the proportion of the variance in the target variable that is explained by the regression model. It is defined as:
   \\[ R^2 = 1 - \frac{\text{RSS}}{\text{TSS}} \\]  
##  ANN Classification with ANN

The Multilayer Perceptron (MLP) was developed to overcome the limitations of simple perceptrons. Unlike the linear mappings in perceptrons, MLPs utilize non-linear mappings between inputs and outputs. An MLP consists of an input layer, an output layer, and one or more hidden layers, each containing multiple neurons. While neurons in a perceptron typically employ threshold-based activation functions like ReLU or sigmoid, neurons in an MLP can use a variety of arbitrary activation functions, enhancing the network's ability to model complex relationships.

![](../fig/MLP.png)


### Loading Libraries

~~~
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data/wine-quality-white-and-red.csv')
df.head()
~~~
{: .python}



~~~
    type  fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  free sulfur dioxide  total sulfur dioxide  density    pH  sulphates  alcohol  quality
0  white            7.0              0.27         0.36            20.7      0.045                 45.0                 170.0   1.0010  3.00       0.45      8.8        6
1  white            6.3              0.30         0.34             1.6      0.049                 14.0                 132.0   0.9940  3.30       0.49      9.5        6
2  white            8.1              0.28         0.40             6.9      0.050                 30.0                  97.0   0.9951  3.26       0.44     10.1        6
3  white            7.2              0.23         0.32             8.5      0.058                 47.0                 186.0   0.9956  3.19       0.40      9.9        6
4  white            7.2              0.23         0.32             8.5      0.058                 47.0                 186.0   0.9956  3.19       0.40      9.9        6
~~~
{: .output}


### Data Preprocessing

~~~
X = df.drop('type', axis=1)
y = df['type']

# Convert categorical values to numerical values using LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

sc = StandardScaler()
X = sc.fit_transform(X)


trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2)


# Convert target variables to NumPy arrays and reshape
trainY = np.array(trainY).reshape(-1, 1)
testY = np.array(testY).reshape(-1, 1)

# Convert data to PyTorch tensors with the correct data type
X_train = torch.Tensor(trainX)
y_train = torch.Tensor(trainY)  
X_test = torch.Tensor(testX)
y_test = torch.Tensor(testY)  
~~~
{: .python}



~~~
# Define the ANN model
class ANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

input_size = X_train.shape[1]
hidden_size = 64
output_size = 1
model = ANN(input_size, hidden_size, output_size)

device = torch.device("cuda:0" if torch.cuda.is_available() else "mps:0" if torch.backends.mps.is_available() else "cpu")
model.to(device)
# move the tensor to GPU device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)
print(model)
~~~
{: .python}

This simple neural network architecture is suitable for binary classification tasks where the input data has 12 features, and the output is a probability indicating the likelihood of belonging to the positive class.


~~~
ANN(
  (fc1): Linear(in_features=12, out_features=64, bias=True)
  (relu): ReLU()
  (fc2): Linear(in_features=64, out_features=1, bias=True)
  (sigmoid): Sigmoid()
)
~~~
{: .output}

The output is a summary of the architecture of an Artificial Neural Network (ANN) model implemented using PyTorch:

1. **Input Layer**:
   - The network expects an input with 12 features. This correspond to 12 different measurements or attributes in the dataset.

2. **First Fully Connected Layer (fc1)**:
   - This layer performs a linear transformation on the 12 input features, producing 64 output features. The transformation can be represented as:
     \\[
     \text{output} = \text{input} \times \text{weight} + \text{bias}
     \\]
   - The weight is a matrix with dimensions (12, 64) and the bias is a vector with 64 elements.

3. **ReLU Activation Function**:
   - After the first linear transformation, the ReLU activation function is applied. This introduces non-linearity to the model, enabling it to capture more complex patterns in the data. The ReLU function is defined as:
     \\[
     \text{ReLU}(x) = \max(0, x)
     \\]
   - This means that any negative values in the output from `fc1` are set to 0, while positive values remain unchanged.

4. **Second Fully Connected Layer (fc2)**:
   - The second fully connected layer takes the 64 features produced by the ReLU activation and transforms them into a single output feature using another linear transformation. This is typically the final layer in a binary classification network.

5. **Sigmoid Activation Function**:
   - Finally, the Sigmoid activation function is applied to the output of the second fully connected layer. The Sigmoid function maps the output to a value between 0 and 1, which can be interpreted as the probability of the positive class in a binary classification problem. The Sigmoid function is defined as:
     \\[
     \sigma(x) = \frac{1}{1 + e^{-x}}
     \\]



~~~
# Define the loss function and optimizer
criterion = nn.BCELoss() # binary cross-entropy loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    predictions = np.round(predictions.cpu().numpy()).astype(int).reshape(-1)
    accuracy = np.mean(predictions == y_test_tensor.cpu().numpy().reshape(-1))
    print(f'Accuracy: {accuracy:.4f}')
~~~
{: .python}

~~~
Epoch [10/100], Loss: 0.4843
Epoch [20/100], Loss: 0.0108
Epoch [30/100], Loss: 0.0026
Epoch [40/100], Loss: 0.0033
Epoch [50/100], Loss: 0.0012
Epoch [60/100], Loss: 0.0005
Epoch [70/100], Loss: 0.0015
Epoch [80/100], Loss: 0.0010
Epoch [90/100], Loss: 0.0007
Epoch [100/100], Loss: 0.0006
Accuracy: 0.9969
~~~
{: .output}

> ## Exercise 1: Training and Evaluating an ANN Classifier
>
> In this exercise, we will enhance the neural network training process with PyTorch by implementing the following steps:
>
> 1. Train an ANN model using PyTorch on a wine quality dataset.
> 2. Evaluate the model using classification metrics.
> 3. Plot the confusion matrix to visualize the performance of the classifier.
>
> > ## Solution
> > 
> > ~~~
> > from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
> > input_size = X_train.shape[1]
> > hidden_size = 64
> > output_size = 1
> > model = ANN(input_size, hidden_size, output_size)
> > 
> > # Move the model to the available device
> > device = torch.device("cuda:0" if torch.cuda.is_available() else "mps:0" if torch.backends.mps.is_available() else "cpu")
> > model.to(device)
> > # Move the tensors to the device
> > X_train, y_train = X_train.to(device), y_train.to(device)
> > X_test, y_test = X_test.to(device), y_test.to(device)
> > 
> > # Define the loss function and optimizer
> > criterion = nn.BCELoss()
> > optimizer = optim.Adam(model.parameters(), lr=0.001)
> > # Convert data into PyTorch datasets and dataloaders
> > train_dataset = TensorDataset(X_train, y_train)
> > train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
> > 
> > # Train the model
> > num_epochs = 100
> > for epoch in range(num_epochs):
> >     model.train()  # Set the model to training mode
> >     for inputs, targets in train_loader:
> >        inputs, targets = inputs.to(device), targets.to(device)
> >         outputs = model(inputs)
> >         loss = criterion(outputs, targets)
> >         optimizer.zero_grad()
> >         loss.backward()
> >         optimizer.step()
> > 
> >     if (epoch + 1) % 10 == 0:
> >         print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
> > # Evaluation
> > model.eval()  # Set the model to evaluation mode
> > with torch.no_grad():
> >     predY = model(X_test)
> >     predY = np.round(predY.cpu().numpy()).astype(int).reshape(-1)  # Ensure predictions are integers
> > # Calculate classification metrics
> > accuracy = np.mean(predY == testY.reshape(-1))
> > conf_matrix = confusion_matrix(testY, predY)
> > class_report = classification_report(testY, predY, target_names=le.classes_)
> > print(f'Accuracy: {accuracy:.4f}')
> > print('Confusion Matrix:')
> > print(conf_matrix)
> > print('\nClassification Report:')
> > print(class_report)
> > # Plot the confusion matrix
> > disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=le.classes_)
> > disp.plot(cmap=plt.cm.Blues)
> > plt.title('Confusion Matrix')
> > plt.savefig("fig/wine_quality_confusion_matrix.png")
> > plt.show()
> > ~~~
> > {: .python}
> > ![](../fig/wine_quality_confusion_matrix.png)
> {: .solution}
{: .challenge}


The ROC curve provides a visual representation of a model's performance across different thresholds, and the AUC quantifies this performance into a single value for easier comparison.

### ROC Curve
- **ROC Curve (Receiver Operating Characteristic Curve)**: A graphical plot used to evaluate the performance of a binary classification model.
- **Axes**: 
  - **X-axis (False Positive Rate, FPR)**: The proportion of negative instances incorrectly classified as positive (calculated as \\( \text{FPR} = \frac{\text{False Positives}}{\text{False Positives} + \text{True Negatives}} \\)).
  - **Y-axis (True Positive Rate, TPR)**: The proportion of positive instances correctly classified (calculated as \\( \text{TPR} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}} \\)).
- **Purpose**: The ROC curve shows the trade-off between TPR and FPR at different classification thresholds. 

### AUC
- **AUC (Area Under the Curve)**: A single scalar value that summarizes the performance of the model by measuring the entire two-dimensional area underneath the ROC curve.
- **Range**: 0 to 1.
  - **AUC = 1**: Perfect model.
  - **AUC = 0.5**: Model with no discriminative power (random guessing).
  - **AUC < 0.5**: Model performing worse than random guessing.
- **Purpose**: A higher AUC indicates a better performing model, as it reflects higher TPRs at lower FPRs across all possible thresholds.


> ## Exercise 2: Evaluating Model Performance with ROC Curve
> In this exercise, you will evaluate the performance of the trained ANN model using an ROC curve.
> 
> 1. Plot the ROC curve for the MLP classifier.
> 2. Calculate the area under the ROC curve (AUC) to quantify the model's performance.
>
> > ## Solution
> > 
> > ~~~
> > from sklearn.metrics import roc_auc_score, roc_curve
> > import matplotlib.pyplot as plt
> > import torch
> > # Calculate predicted probabilities using sigmoid activation
> > 
> > # Calculate predicted probabilities using sigmoid activation
> > with torch.no_grad():
> >     ypred_proba = torch.sigmoid(model(X_test)).cpu().numpy()
> > 
> > # Calculate ROC AUC score
> > roc_auc = roc_auc_score(testY, ypred_proba)
> > 
> > # Compute ROC curve
> > fpr, tpr, _ = roc_curve(testY, ypred_proba)
> > 
> > # Plot ROC curve in a single figure
> > plt.figure(figsize=(8, 6))
> > 
> > # Plot ROC curve
> > plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
> > plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
> > plt.xlim([0.0, 1.0])
> > plt.ylim([0.0, 1.05])
> > plt.xlabel('False Positive Rate')
> > plt.ylabel('True Positive Rate')
> > plt.title('Receiver Operating Characteristic (ROC) Curve')
> > plt.legend(loc='lower right')
> > plt.grid(True)
> > plt.tight_layout()  # Tighten layout to prevent overlap
> > plt.savefig("fig/wine_quality_roc_auc.png")
> > plt.show()
> > ~~~
> > {: .python}
> > ![](../fig/wine_quality_roc_auc.png)
> {: .solution}
{: .challenge}


Training a good model can take a lot of time. And I mean weeks, months or even years. So, let’s make sure that you know how you can save your precious work. Saving is easy:

~~~
# save the trained model
model_path = 'model.pth'
torch.save(model, model_path)

# Restoring your model is easy too

mpl_model = torch.load(model_path)
~~~



## Convolutional Neural Networks

Convolutional Neural Networks (CNNs) offer a powerful alternative to fully connected neural networks, especially for handling spatially structured data like images. Unlike fully connected networks, where each neuron in one layer is connected to every neuron in the next, CNNs employ a unique architecture that addresses two key limitations. Firstly, fully connected networks result in a large number of parameters, making the models complex and computationally intensive. Secondly, these networks do not consider the order of input features, treating them as if their arrangement does not matter. This can be particularly problematic for image data, where spatial relationships between pixels are crucial.

In contrast, CNNs introduce local connectivity and parameter sharing. Neurons in a CNN layer connect only to a small region of the previous layer, known as the receptive field, preserving the spatial structure of the data. Moreover, CNNs apply the same set of weights (filters or kernels) across different parts of the input through a process called convolution, significantly reducing the number of parameters compared to fully connected networks. This approach not only enhances computational efficiency but also enables CNNs to capture hierarchical patterns in data, such as edges, textures, and more complex structures in images. For instance, a simple 3x3 filter sliding over a 5x5 image can create a feature map that highlights specific patterns, effectively learning from the spatial context of the image.


Now let's take a look at convolutional neural networks (CNNs), the models people really use for classifying images.

~~~
# import PyTorch and its related packages
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

# set default device based on GPU's availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "mps:0" if torch.backends.mps.is_available() else "cpu")
print(device)
~~~
{: .python}

~~~
'mps'
~~~
{: .output}


Download the CIFAR10 dataset from `torchvision` libarary
~~~
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 32
image_size = (32, 32, 3)

train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
train_loader = T.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
test_loader = T.utils.data.DataLoader(test_set, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
~~~
{: .python}

~~~
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./data/cifar-10-python.tar.gz
100%|█████████████████████████████| 170498071/170498071 [06:48<00:00, 416929.38it/s]
Extracting ./data/cifar-10-python.tar.gz to ./data
Files already downloaded and verified
~~~
{: .output}


Let's define the loss function as cross entropy as:

~~~
criterion = nn.CrossEntropyLoss()
~~~
{: .python}

Now, let's define the `ConvNet` class as our CNN model for image classification tasks with 10 output classes. This network comprises a feature extraction module followed by a classifier. The feature extractor includes two convolutional layers, each followed by ReLU activation and max pooling, capturing spatial hierarchies and reducing dimensionality. The classifier consists of a dropout layer to prevent overfitting, a fully connected layer to transform the features into a 512-dimensional space with ReLU activation, and a final fully connected layer that maps to the 10 output classes. The `forward` method orchestrates the data flow through these layers, ensuring the input is processed correctly for classification.

~~~
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 9 * 9, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 128 * 9 * 9)
        x = self.classifier(x)
        return x
net = ConvNet()
net.to(device)
print(net)
~~~
{: .python}


~~~
~~~
{: .output}


~~~
# also the optimizer
optimizer = optim.Adam(net.parameters(), lr=0.001)

train_loss = []
test_loss = []
train_acc = []
test_acc = []

for epoch in range(1, 33):  # loop over the dataset multiple times
    
    running_loss = .0
    correct = 0
    total = 0
    for i, data in enumerate(train_loader):
        # get the inputs
        inputs, labels = data
        if device == 'cuda':
            inputs, labels = inputs.to(device), labels.to(device)

        # reset the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        # backward
        loss.backward()
        # optimize
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = T.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    running_loss /= len(train_loader)
    train_loss.append(running_loss)
    running_acc = correct / total
    train_acc.append(running_acc)
    
    if epoch % 4 == 0:
        print('\nEpoch: {}'.format(epoch))
        print('Train Acc. => {:.3f}%'.format(100 * running_acc), end=' | ')
        print('Train Loss => {:.5f}'.format(running_loss))
    
    # evaluate on the test set
    # note this is usually performed on the validation set
    # for simplicity we just evaluate it on the test set
    with T.no_grad():
        correct = 0
        total = 0
        test_running_loss = .0
        for data in test_loader:
            inputs, labels = data
            if device == 'cuda':
                inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            test_running_loss += loss.item()
            _, predicted = T.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        test_running_loss /= len(test_loader)
        test_loss.append(test_running_loss)
        test_running_acc = correct / total
        test_acc.append(test_running_acc)
        
        if epoch % 4 == 0:
            print('Test Acc.  => {:.3f}%'.format(100 * test_running_acc), end=' | ')
            print('Test Loss  => {:.5f}'.format(test_running_loss))

print('Finished Training')
~~~
{: .python}

~~~
Epoch: 4
Train Acc. => 59.444% | Train Loss => 1.14581
Test Acc.  => 60.100% | Test Loss  => 1.11926

Epoch: 8
Train Acc. => 69.432% | Train Loss => 0.87285
Test Acc.  => 67.090% | Test Loss  => 0.93443

Epoch: 12
Train Acc. => 75.260% | Train Loss => 0.70139
Test Acc.  => 70.550% | Test Loss  => 0.85366

Epoch: 16
Train Acc. => 81.078% | Train Loss => 0.54445
Test Acc.  => 71.850% | Test Loss  => 0.83311

Epoch: 20
Train Acc. => 85.854% | Train Loss => 0.41174
Test Acc.  => 72.440% | Test Loss  => 0.82984

Epoch: 24
Train Acc. => 90.186% | Train Loss => 0.28792
Test Acc.  => 73.930% | Test Loss  => 0.84632

Epoch: 28
Train Acc. => 93.288% | Train Loss => 0.19497
Test Acc.  => 73.710% | Test Loss  => 0.91641

Epoch: 32
Train Acc. => 95.684% | Train Loss => 0.13074
Test Acc.  => 74.170% | Test Loss  => 0.99424
Finished Training
~~~
{: .putput}


Now, it is time to plot training and test losses and accuracies over 32 epochs using `matplotlib`. The plot has two subplots: one for the loss and one for the accuracy. The first subplot displays the train and test losses, while the second subplot shows the train and test accuracies. Both plots include labels, titles, legends, and grids for clarity. The layout is adjusted to prevent overlap.

~~~
# Plotting train and test loss
plt.figure(figsize=(12, 5))

# Subplot for Loss
plt.subplot(1, 2, 1)
plt.plot(range(1, 33), train_loss, label='Train Loss')
plt.plot(range(1, 33), test_loss, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and Test Loss over Epochs')
plt.legend()
plt.grid(True)

# Subplot for Accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, 33), train_acc, label='Train Accuracy')
plt.plot(range(1, 33), test_acc, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train and Test Accuracy over Epochs')
plt.legend()
plt.grid(True)
# Display the plots
plt.tight_layout()
plt.savefig("fig/loss_accuracy_CIFAR10.png")
plt.show()
~~~
{: .python}

![](../fig/loss_accuracy_CIFAR10.png")


The visualizations and the provided metrics clearly highlight the overfitting trend, emphasizing the need for strategies to enhance the model's robustness and generalization capabilities.




Let's define a function to visualize a batch of images from the CIFAR-10 dataset. It first transforms a tensor into a numpy array suitable for plotting, denormalizes the images for correct display, and plots them using matplotlib. The function imshow displays the images with optional titles.

The class names for CIFAR-10 are defined in a list. The code then retrieves a batch of training data, selects the first 10 images and their corresponding labels, and creates a grid of these images using torchvision.utils.make_grid. Finally, it displays the grid with the correct class labels as the title and saves the figure as follows:

~~~
# Define a function to show images
def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

# Define the class names for CIFAR-10
class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Get a batch of training data
inputs, classes = next(iter(train_loader))

# Select only the first 10 images and labels
inputs = inputs[:10]
classes = classes[:10]

# Make a grid from the selected batch
out = torchvision.utils.make_grid(inputs, nrow=10)

# Display the images with correct titles
imshow(out, title=' '.join([class_names[x] for x in classes]))
plt.savefig("fig/class_labels_CIFAR10.png")
plt.show()
~~~
{: .python}

![](../fig/class_labels_CIFAR10.png"))




To control overfitting there are  several strategies during the training process such as data augmentation, dropout, and early stopping. Additionally, I can also use L2 regularization to the optimizer and a learning rate scheduler to adjust the learning rate during training as follows:

~~~

import torch as T
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Data augmentation
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load CIFAR-10 dataset
train_set = datasets.CIFAR10(root='./data', train=True,
                             download=True, transform=transform_train)
train_loader = DataLoader(train_set, batch_size=batch_size,
                          shuffle=True, num_workers=2)

test_set = datasets.CIFAR10(root='./data', train=False,
                            download=True, transform=transform_test)
test_loader = DataLoader(test_set, batch_size=batch_size,
                         shuffle=False, num_workers=2)

# Define your network (with dropout layers added)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 9 * 9, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 128 * 9 * 9)
        x = self.classifier(x)
        return x

# Initialize the network
net = ConvNet(num_classes=10)
device = 'cuda' if T.cuda.is_available() else 'cpu'
net.to(device)

# Define the criterion and optimizer with L2 regularization
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

train_loss = []
test_loss = []
train_acc = []
test_acc = []

early_stopping_threshold = 5
no_improvement_count = 0
best_test_loss = float('inf')

for epoch in range(1, 33):  # loop over the dataset multiple times
    
    running_loss = .0
    correct = 0
    total = 0
    net.train()
    for i, data in enumerate(train_loader):
        # get the inputs
        inputs, labels = data
        if device == 'cuda':
            inputs, labels = inputs.to(device), labels.to(device)

        # reset the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        # backward
        loss.backward()
        # optimize
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = T.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    running_loss /= len(train_loader)
    train_loss.append(running_loss)
    running_acc = correct / total
    train_acc.append(running_acc)
    
    if epoch % 4 == 0:
        print('\nEpoch: {}'.format(epoch))
        print('Train Acc. => {:.3f}%'.format(100 * running_acc), end=' | ')
        print('Train Loss => {:.5f}'.format(running_loss))
    
    # evaluate on the test set
    net.eval()
    with T.no_grad():
        correct = 0
        total = 0
        test_running_loss = .0
        for data in test_loader:
            inputs, labels = data
            if device == 'cuda':
                inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            test_running_loss += loss.item()
            _, predicted = T.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        test_running_loss /= len(test_loader)
        test_loss.append(test_running_loss)
        test_running_acc = correct / total
        test_acc.append(test_running_acc)
        
        if epoch % 4 == 0:
            print('Test Acc.  => {:.3f}%'.format(100 * test_running_acc), end=' | ')
            print('Test Loss  => {:.5f}'.format(test_running_loss))

    scheduler.step()

    # Early stopping
    if test_running_loss < best_test_loss:
        best_test_loss = test_running_loss
        no_improvement_count = 0
    else:
        no_improvement_count += 1
        if no_improvement_count >= early_stopping_threshold:
            print('Early stopping at epoch {}'.format(epoch))
            break

print('Finished Training')
~~~
{: .python}


~~~
# Plotting train and test loss
plt.figure(figsize=(12, 5))

# Subplot for Loss
plt.subplot(1, 2, 1)
plt.plot(range(1, 33), train_loss, label='Train Loss')
plt.plot(range(1, 33), test_loss, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and Test Loss over Epochs')
plt.legend()
plt.grid(True)

# Subplot for Accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, 33), train_acc, label='Train Accuracy')
plt.plot(range(1, 33), test_acc, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train and Test Accuracy over Epochs')
plt.legend()
plt.grid(True)
# Display the plots
plt.tight_layout()
plt.savefig("fig/loss_accuracy2_CIFAR10.png")
plt.show()
~~~


![](../fig/class_labels2_CIFAR10.png")

## Transfer Learning 


Transfer learning involves leveraging knowledge gained from solving one problem and applying it to a different, but related, problem. This approach can significantly improve learning efficiency, especially when labeled data is limited for the target task. There are two common strategies for transfer learning: fine-tuning and feature extraction.

Fine-tuning begins with a pretrained model, typically trained on a large dataset, and updates all of the model's parameters to adapt it to the new task. Essentially, the entire model is retrained using the new dataset, allowing it to learn task-specific features while retaining the general knowledge learned from the original task.

On the other hand, feature extraction involves starting with a pretrained model and keeping its parameters fixed, except for the final layer weights responsible for making predictions. This approach treats the pretrained model as a fixed feature extractor, extracting useful features from the input data, and only trains a new classifier on top of these extracted features.

Both fine-tuning and feature extraction are valuable techniques in transfer learning, offering flexibility in adapting pretrained models to new tasks with varying amounts of available data. Fine-tuning allows for more adaptation to the new task, while feature extraction can be faster and requires less computational resources, particularly when dealing with limited data or computational constraints.



~~~
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                         download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32,
                                           shuffle=True, num_workers=2)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32,
                                          shuffle=False, num_workers=2)

# Define pretrained ResNet model
model = models.resnet18(pretrained=True)

# Freeze model parameters
for param in model.parameters():
    param.requires_grad = False

# Replace the final fully connected layer with new layer for CIFAR-10 classification
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Train the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 10

train_loss = []
test_loss = []
train_acc = []
test_acc = []

for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item() * inputs.size(0)  # Update running train loss
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    train_loss.append(running_train_loss / len(train_loader.dataset))  # Append epoch's train loss
    train_acc.append(correct / total)
    
    # Validation
    model.eval()
    test_running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_running_loss += loss.item() * inputs.size(0)  # Update running test loss
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_loss.append(test_running_loss / len(test_loader.dataset))  # Append epoch's test loss
    test_acc.append(correct / total)
    # Print epoch statistics
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss[-1]:.4f}, Train Acc: {train_acc[-1]:.4f}, Test Loss: {test_loss[-1]:.4f}, Test Acc: {test_acc[-1]:.4f}')

print('Finished Training')
~~~
{: .python}

~~~
Epoch [1/10], Train Loss: 0.8026, Train Acc: 0.7322, Test Loss: 0.6058, Test Acc: 0.7934
Epoch [2/10], Train Loss: 0.6374, Train Acc: 0.7806, Test Loss: 0.6237, Test Acc: 0.7908
Epoch [3/10], Train Loss: 0.6161, Train Acc: 0.7862, Test Loss: 0.5727, Test Acc: 0.8025
Epoch [4/10], Train Loss: 0.6004, Train Acc: 0.7913, Test Loss: 0.5847, Test Acc: 0.8018
Epoch [5/10], Train Loss: 0.5951, Train Acc: 0.7942, Test Loss: 0.5852, Test Acc: 0.8011
Epoch [6/10], Train Loss: 0.5907, Train Acc: 0.7956, Test Loss: 0.5838, Test Acc: 0.8027
Epoch [7/10], Train Loss: 0.5942, Train Acc: 0.7946, Test Loss: 0.5909, Test Acc: 0.8009
Epoch [8/10], Train Loss: 0.5919, Train Acc: 0.7954, Test Loss: 0.6120, Test Acc: 0.7938
Epoch [9/10], Train Loss: 0.5812, Train Acc: 0.7986, Test Loss: 0.5728, Test Acc: 0.8050
Epoch [10/10], Train Loss: 0.5874, Train Acc: 0.7989, Test Loss: 0.5765, Test Acc: 0.8030
Finished Training
~~~
{: .output}

~~~
import matplotlib.pyplot as plt

# Plotting train and test loss
plt.figure(figsize=(12, 5))

# Subplot for Loss
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_loss, label='Train Loss')
plt.plot(range(1, num_epochs+1), test_loss, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and Test Loss over Epochs')
plt.legend()
plt.grid(True)

# Subplot for Accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), train_acc, label='Train Accuracy')
plt.plot(range(1, num_epochs+1), test_acc, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train and Test Accuracy over Epochs')
plt.legend()
plt.grid(True)

# Display the plots
plt.tight_layout()
plt.savefig("fig/loss_accuracy3_CIFAR10.png")
plt.show()

~~~
{: .python}

![](../fig/class_labels3_CIFAR10.png")

