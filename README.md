# PMLPP

This is an experimental Machine Learning module for the Pandemonium Engine. 

Based on https://github.com/novak-99/MLPP .

## Status

- Reworked codestyle to match the engine's.
- Reworked classes to work well with the engine.
- Added custom MLPPVector and MLPPMatrix that store data in a single continuous array. Their downside is push back, and remove uses realloc().
- Added binds for most of the classes.
- Added unit tests.

## Todos

### Saves

Reimplement saving.

### Bind remaining methods

Go through and bind all methods. Also add properties as needed.

### Add initialization api to all classes that need it

The original library used contructors to initialize everything, but with the engine scripts can't rely on this,
make sure all classes have initializations apis, and they bail out when they are in an uninitialized state.

### Rework remaining apis.

Rework and bind the remaining apis, so they can be used from scripts.

### Error handling

Make error macros usage consistent. Also a command line option should be available that disables them for math operations.

### Crashes

There are still likely lots of crashes, find, and fix them.

### Unit tests

- Add more unit tests
- Also use the engine's own unit test module. It still needs to be fininshed, would be a good idea doing it alongside this modules's tests.
- They should only be built when you want them. Command line option: `mlpp_tests=yes`

### Old classes

- Remove all old classes once they are no longer needed.
- Old classes should only be built when you want them. Command line option: `mlpp_tests=yes` or maybe something else?
- Make old classes use old utilities.
- Remove old apis from new classes that has been ported.

### std::random

Replace remaining std::random usage with engine internals.

### Tensor

Add a tensor class. Same as MLPPVector and MLPPMatrix, except it's n-d.

### More algos

Add more machine learning algorithms.


## Contents of the Library
1. ***Regression***
    1. Linear Regression 
    2. Logistic Regression
    3. Softmax Regression
    4. Exponential Regression
    5. Probit Regression
    6. CLogLog Regression
    7. Tanh Regression
2. ***Deep, Dynamically Sized Neural Networks***
    1. Possible Activation Functions
        - Linear
        - Sigmoid
        - Softmax
        - Swish
        - Mish
        - SinC
        - Softplus
        - Softsign
        - CLogLog
        - Logit
        - Gaussian CDF
        - RELU
        - GELU
        - Sign
        - Unit Step 
        - Sinh
        - Cosh
        - Tanh
        - Csch
        - Sech
        - Coth
        - Arsinh
        - Arcosh
        - Artanh
        - Arcsch
        - Arsech
        - Arcoth
    2. Possible Optimization Algorithms
        - Batch Gradient Descent
        - Mini-Batch Gradient Descent 
        - Stochastic Gradient Descent 
        - Gradient Descent with Momentum
        - Nesterov Accelerated Gradient
        - Adagrad Optimizer 
        - Adadelta Optimizer 
        - Adam Optimizer 
        - Adamax Optimizer 
        - Nadam Optimizer 
        - AMSGrad Optimizer 
        - 2nd Order Newton-Raphson Optimizer*
        - Normal Equation*
        <p></p>
        *Only available for linear regression
    3. Possible Loss Functions
        - MSE
        - RMSE 
        - MAE
        - MBE
        - Log Loss
        - Cross Entropy
        - Hinge Loss
        - Wasserstein Loss
    4. Possible Regularization Methods
        - Lasso
        - Ridge
        - ElasticNet
        - Weight Clipping
    5. Possible Weight Initialization Methods
        - Uniform 
        - Xavier Normal
        - Xavier Uniform
        - He Normal
        - He Uniform
        - LeCun Normal
        - LeCun Uniform
    6. Possible Learning Rate Schedulers
        - Time Based 
        - Epoch Based
        - Step Based
        - Exponential 
3. ***Prebuilt Neural Networks***
    1. Multilayer Peceptron
    2. Autoencoder
    3. Softmax Network
4. ***Generative Modeling***
    1. Tabular Generative Adversarial Networks
    2. Tabular Wasserstein Generative Adversarial Networks
5. ***Natural Language Processing***
    1. Word2Vec (Continous Bag of Words, Skip-Gram)
    2. Stemming
    3. Bag of Words
    4. TFIDF
    5. Tokenization 
    6. Auxiliary Text Processing Functions
6. ***Computer Vision***
    1. The Convolution Operation
    2. Max, Min, Average Pooling
    3. Global Max, Min, Average Pooling
    4. Prebuilt Feature Detectors
        - Horizontal/Vertical Prewitt Filter
        - Horizontal/Vertical Sobel Filter
        - Horizontal/Vertical Scharr Filter
        - Horizontal/Vertical Roberts Filter
        - Gaussian Filter
        - Harris Corner Detector
7. ***Principal Component Analysis***
8. ***Naive Bayes Classifiers***
    1. Multinomial Naive Bayes
    2. Bernoulli Naive Bayes 
    3. Gaussian Naive Bayes
9. ***Support Vector Classification***
    1. Primal Formulation (Hinge Loss Objective) 
    2. Dual Formulation (Via Lagrangian Multipliers)
10. ***K-Means***
11. ***k-Nearest Neighbors***
12. ***Outlier Finder (Using z-scores)***
13. ***Matrix Decompositions***    
    1. SVD Decomposition
    2. Cholesky Decomposition
        - Positive Definiteness Checker 
    3. QR Decomposition
14. ***Numerical Analysis***
    1. Numerical Diffrentiation 
        - Univariate Functions 
        - Multivariate Functions 
    2. Jacobian Vector Calculator
    3. Hessian Matrix Calculator
    4. Function approximator
        - Constant Approximation
        - Linear Approximation 
        - Quadratic Approximation
        - Cubic Approximation
    5. Diffrential Equations Solvers 
        - Euler's Method 
        - Growth Method
15. ***Mathematical Transforms***
    1. Discrete Cosine Transform
16. ***Linear Algebra Module***
17. ***Statistics Module***
18. ***Data Processing Module***
    1. Setting and Printing Datasets 
    2. Available Datasets
        1. Wisconsin Breast Cancer Dataset
            - Binary
            - SVM 
        2. MNIST Dataset
            - Train
            - Test
        3. Iris Flower Dataset
        4. Wine Dataset
        5. California Housing Dataset
        6. Fires and Crime Dataset (Chicago)
    3. Feature Scaling 
    4. Mean Normalization
    5. One Hot Representation
    6. Reverse One Hot Representation
    7. Supported Color Space Conversions 
        - RGB to Grayscale
        - RGB to HSV
        - RGB to YCbCr
        - RGB to XYZ
        - XYZ to RGB
19. ***Utilities***
    1. TP, FP, TN, FN function
    2. Precision
    3. Recall 
    4. Accuracy
    5. F1 score


## Citations
Various different materials helped me along the way of creating ML++, and I would like to give credit to several of them here. [This](https://www.tutorialspoint.com/cplusplus-program-to-compute-determinant-of-a-matrix) article by TutorialsPoint was a big help when trying to implement the determinant of a matrix, and [this](https://www.geeksforgeeks.org/adjoint-inverse-matrix/) article by GeeksForGeeks was very helpful when trying to take the adjoint and inverse of a matrix.

