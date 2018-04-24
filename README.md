# Intro to PyTorch

Welcome to an introduction to PyTorch hosted by [Expero](https://experoinc.com) and [Global Data Geeks](http://globaldatageeks.org/)!

Your instructors today are [Graham Ganssle](https://www.linkedin.com/in/grahamganssle/) and [Ryan Brady](https://www.experoinc.com/author/ryan-brady). Please don't hesitate to get up and scribble a question on the whiteboard!

---

### Requirements

**Before you begin,** you should have the following installed:

* [PyTorch](https://github.com/pytorch/pytorch/blob/master/README.md#installation) (CPU or GPU flavor, depending on your hardware)

* [Pandas](https://pandas.pydata.org/pandas-docs/stable/install.html)

* [TensorboardX](https://github.com/lanpa/tensorboard-pytorch)

And optionally:

* [Bokeh](https://bokeh.pydata.org/en/latest/docs/installation.html)

* [tqdm](https://github.com/tqdm/tqdm#installation)

---

### Syllabus

When you leave today you should know how to build and train simple PyTorch models. We'll build a neural network in the course today, which is not the only model type PyTorch is capable of representing. In fact, the library is full of goodies which you should play with at home after this course! Here's what we'll be working on today in chronological order:

1. Tensors - what are they?

2. Gradients - how do gradients play a role in the world of deep learning?

3. Optimization - how to use tensors (**1.**) and gradients (**2.**) to find function extrema.

4. Neural networks (regression) - we'll train a neural network to approximate a continuous, differentiable function.

5. Neural networks (classification) - we'll discuss how to train a neural network to bucket customers into various customer cohorts based on their product interests and demographic information. We'll set up the code and hand it off to you for training on a GPU at home.

6. [time permitting] Matrix Factorization - if there's enough interest (and time), we'll show you how we used PyTorch to build the training data we're working on in this course.

---

### Data Source

Thanks to UC Irvine for providing the [data](http://archive.ics.uci.edu/ml/datasets/online+retail), and Chen, et. al for collecting and open sourcing it:
>Daqing Chen, Sai Liang Sain, and Kun Guo, Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining, Journal of Database Marketing and Customer Strategy Management, Vol. 19, No. 3, pp. 197â€“208, 2012 (Published online before print: 27 August 2012. doi: 10.1057/dbm.2012.17).
