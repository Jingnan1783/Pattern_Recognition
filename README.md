# Pattern_Recognition
Build a system to classify between clear and blurry images using linear classifiers.
## High level overview
![](https://courses.engr.illinois.edu/ece544na/fa2017/_site/homework/mp1/img/train_test_pipeline.png)
## Numpy Implementation
### Reading in data
Revelant File: 
> utils/io_tools.py
### Data processing
> utils/data_tools.py
The feature extraction process involes simple image manipulation, such as mean removal, filtering, element-wise scaling.
### Linear model implementation
> models/linear_model.py
Implement an abstract base class for linear modles (e.g linear regression, logistic regression, and support vector machine).
#### Forward operation
Forward operation is the function which takes an input and outputs a score. In this case, for linear models, it is F = wTx + b.
For simplicity, we will rewrite x = [x,1], and w = [w,b], then equivalently, F = wTx.
#### Loss Functions
Loss function takes in a score, and ground-truth label and outputs a scalar. The loss function indicates how good the model’s predicted score fits to the ground-truth.
#### Backward operation
Backward opretaion is the operation for computing gradient of the loss function w.r.t to the model parameters. This is computed after the forward operation to update the model.
#### Predict operation
The prediction operation is a function which takes a score as input and outputs a prediction ∈Y, in the case of binary classification, Y = {−1,1}.
> models/linear_model.py, models/linear_regression.py, models/logistic_regression.py, models/support_vector_machine.py
## Gradient descent implementation
Gradient descent is a optimization algorithm, where you adjust the model in the direction of the negative gradient of L.
Repeat until convergence:
> w(t) = w(t−1) − η∇L(t−1)
The above equation is refered as an update step, which consists of one pass of the forward and backward operation.
> models/train_eval_model.py
## Model selection
Observe that dataset is divided into three parts, train, eval, and test, the splits were constructed to perform the following:
### Training
The training set is used to train your model, meaning you use this set to find the optimal model parameters.
### Validation
The validation set is used to evalute the quality of the trained models, and determine which set of hyperparameters to use. In our case, the hyperparameters are the feature extractions, loss function, the learning rate, and the number of training steps.
The idea is to pick the best hyperparamters based on the validation performance, this prevents overfitting to the training set.
### Testing
Lastly, the testing set is never used to determine anything of the trained model. This set is purely used for evaluation. In the test split,we did not provie the label, you will be able to compute the testing performace through the Kaggle competition.
> models/train_eval_model.py
### Running Experiments
Experiment with different features, weights initialization and learning rate. We will not grade the main.py file, feel free to change it.
> models/main.py
