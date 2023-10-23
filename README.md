# Intro to Deep Learning
Use TensorFlow and Keras to build and train neural networks for structured data.

## A Single Neuron

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

The easiest way to create a model in Keras is through `keras.Sequential`, which creates a neural network as a stack of layers. We can create models like those above using a dense layer (which we'll learn more about in the next lesson).






### 
