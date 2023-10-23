> ***NOTE!***
>
> *This is my personal note from the course, and related sources. Before this course Highly recommend [Computer Aided Solution Processing](https://github.com/gabboraron/Computer_Aided_Solution_Processing) and [Artificial Intelligence - Intelligent agents paradigm](https://github.com/gabboraron/artificial_intelligence-intelligent_agents_paradigm) courses.*

# Intro to Deep Learning
Use TensorFlow and Keras to build and train neural networks for structured data.

## A Single Neuron
file:[exercise-a-single-neuron.ipynb](https://github.com/gabboraron/Intro_to_Deep_Learning-Kaggle_course/blob/main/exercise-a-single-neuron.ipynb)

### What is Deep Learning?
> Some of the most impressive advances in artificial intelligence in recent years have been in the field of deep learning. Natural language translation, image recognition, and game playing are all tasks where deep learning models have neared or even exceeded human-level performance.
>
> So what is deep learning? Deep learning is an approach to machine learning characterized by deep stacks of computations. This depth of computation is what has enabled deep learning models to disentangle the kinds of complex and hierarchical patterns found in the most challenging real-world datasets.
>
> Through their power and scalability neural networks have become the defining model of deep learning. Neural networks are composed of neurons, where each neuron individually performs only a simple computation. The power of a neural network comes instead from the complexity of the connections these neurons can form.

### The Linear Unit
> So let's begin with the fundamental component of a neural network: the individual neuron. As a diagram, a neuron (or unit) with one input looks like:

![neuron](https://storage.googleapis.com/kaggle-media/learn/images/mfOlDR6.png) 

The Linear Unit: $y=wx+b$ Does the formula looks familiar?  It's an equation of a line! It's the slope-intercept equation, where $w$ is the slope and $b$ is the y-intercept. 

## Example - The Linear Unit as a Model
Though individual neurons will usually only function as part of a larger network, it's often useful to start with a single neuron model as a baseline. ***Single neuron models are linear models.***

Based on the [80 Cereals](https://www.kaggle.com/datasets/crawford/80-cereals) dataset we could estimate the calorie content of a cereal with 5 grams of sugar per serving like this:

![Computing with the linear unit](https://storage.googleapis.com/kaggle-media/learn/images/yjsfFvY.png)
*Training a model with 'sugars' (grams of sugars per serving) as input and 'calories' (calories per serving) as output, we might find the bias is b=90 and the weight is w=2.5.*

### Multiple Inputs
What if we wanted to expand our model to include things like fiber or protein content? That's easy enough. We can just add more input connections to the neuron, one for each additional feature. To find the output, we would multiply each input to its connection weight and then add them all together.

![A linear unit with three inputs](https://storage.googleapis.com/kaggle-media/learn/images/vyXSnlZ.png)

The formula for this neuron would be $y=w0x0+w1x1+w2x2+b$. A linear unit with two inputs will fit a plane, and a unit with more inputs than that will fit a [hyperplane](https://mathworld.wolfram.com/Hyperplane.html).

### Linear Units in [Keras](https://keras.io)
> *"[Keras](https://keras.io) is an API designed for human beings, not machines. Keras follows best practices for reducing cognitive load: it offers consistent & simple APIs, it minimizes the number of user actions required for common use cases, and it provides clear & actionable error messages. Keras also gives the highest priority to crafting great documentation and developer guides."*


The easiest way to create a model in Keras is through `keras.Sequential`, which creates a neural network as a stack of layers. We can create models like those above using a dense layer (which we'll learn more about in the next lesson).

We could define a linear model accepting three input features (`'sugars'`, `'fiber'`, and `'protein'`) and producing a single output (`'calories'`) like [so](https://www.kaggle.com/code/ryanholbrook/a-single-neuron?scriptVersionId=126574232&cellId=3):

````Python
from tensorflow import keras
from tensorflow.keras import layers

# Create a network with 1 linear unit
model = keras.Sequential([
    layers.Dense(units=1,         # define how many outputs we want.
                 input_shape=[3]) # we tell Keras the dimensions of the inputs. Setting input_shape=[3] ensures the model will accept three features as input ('sugars', 'fiber', and 'protein').
])
````

> The data we'll use in this course will be tabular data, like in a Pandas dataframe. We'll have one input for each feature in the dataset. The features are arranged by column, so we'll always have `input_shape=[num_columns]`. The reason Keras uses a list here is to permit use of more complex datasets. Image data, for instance, might need three dimensions: `[height, width, channels]`. 
>
> Internally, Keras represents the weights of a neural network with **tensors**. Tensors are basically TensorFlow's version of a Numpy array with a few differences that make them better suited to deep learning. One of the most important is that tensors are compatible with [GPU](https://www.kaggle.com/docs/efficient-gpu-usage) and [TPU](https://www.kaggle.com/docs/tpu)) accelerators. TPUs, in fact, are designed specifically for tensor computations.
>
> A model's weights are kept in its `weights` attribute as a list of tensors. Get the weights of the model you defined above. (If you want, you could display the weights with something like: `print("Weights\n{}\n\nBias\n{}".format(w, b))`).

You can get the used weights like this:
```Python
w, b = model.weights
print("Weights\n{}\n\nBias\n{}".format(w, b))
```

*Keras represents weights as tensors, but also uses tensors to represent data. When you set the input_shape argument, you are telling Keras the dimensions of the array it should expect for each example in the training data. Setting `input_shape=[3]` would create a network accepting vectors of length 3, like [0.2, 0.4, 0.6].*









### 
