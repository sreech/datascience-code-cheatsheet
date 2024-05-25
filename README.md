#  Basic Code
```
import pandas as pd
df = pd.read_csv("test.csv")
df['TotalMiles']
```
# Normal distribution aka Gaussian
one of most probability distribution in stats, bellshaped curve symmetric around mean. contains mean and standard dev. to find probablity of student scoring above 120 etc..
```
from scipy import stats, where mu is mean, sd =standard dev
inventory = 1000
fill_rate = stats.norm.cdf(inventory, loc=mu, scale=sd)
```

# One Bernoulli trial (aka coin toss)
tells probability of random variable that takes one of 2 possible values (success or failure). binary outcome
```
import numpy as np
p=0.5 #probability
np.random.choice(['Heads', 'Tails'], p=[p, 1 - p])
x = np.random.choice(['Heads', 'Tails'], p=[p, 1 - p], size=50)
pd.Series(x).value_counts()
p = 0.5  # Probability of "success".
n = 1000 # Sample size.
x = stats.bernoulli.rvs(p, size = n)
bernoulli_sample = pd.Series(x).value_counts() / n
bernoulli_pmf = stats.bernoulli.pmf([0, 1], p)
```

# Probability dist of the number of successes in a fixed number of independent Bernoulli trials,
```
import scipy
from scipy.stats import binom
k = 5   # Number of "successes".
n = 10  # Number of Bernoulli trials.
p = 0.5 # Probability of "success".
stats.binom.pmf(k, n, p)
```
# PDF: 
describe the probability distribution of a continuous random variable. It gives the relative likelihood of the random variable taking on a specific value or falling within a particular range. In normal dist, pdf gives the bell shaped curve representing likelihood different values occuring. PDF for single point is 0, pdf represents probabilities as areas under curve. Area under te entire curve is 1.
# CDF:
gives the cumulative probability that a random variable takes on a value less than or equal to a given value. For example, if u hve bag of marbles, each marble written with number 1 thro 10, let's say you want to know the probability that the number on the marble is less than or equal to 5. The CDF gives you this information. If you find out that the CDF at 5 is 0.7, it means there's a 70% chance that the number on the marble is less than or equal to 5.
# PMF:
describe the probability mass distribution of a discrete random variable. It gives the probability that the variable takes on a specific value. Six sided die, PMF assigns prob of 1/6 of each possible outcome
# PPF:
Percent Point Fuction, inverse cdf. let's say you're told that there's a 0.7 probability that the number on the marble is less than or equal to 5. You want to find out what number corresponds to this probability.

# Plotting
```
import matplotlib.pyplot as plt
plots = plt.bar(r_values, dist)
fig = plt.scatter(data1, data2)
```
# 
Covariance: covariance measures the extent to which two variables change together. Covariance +ve or -ve
```
import numpy as np
hours_studied = [3, 4, 5, 6, 7, 8]  # Hours spent studying
exam_scores = [60, 65, 70, 75, 80, 85]  # Corresponding exam scores
```
# Calculate covariance
```
covariance = np.cov(hours_studied, exam_scores)[0, 1]
```
#Covariance between hours studied and exam scores: 12.5, suggests a positive linear relationship between the two variables. Magnitude or strength cant be determined but correlation coefficient gives comprehensive view or could be represented via covar matrix
```
X = np.stack((hours_studied, exam_scores), axis=0)
```
# Calculate the covariance matrix
covMatrix = np.cov(X, bias=False)
[[15.8  9.6]
[ 9.6 21.7]]

# Correlation:
is a standardized measure showing correlation coefficients between variables range from -1 to 1
```
import numpy as np
data = np.array([
    [10, 20, 30],  # Variable 1
    [15, 25, 35],  # Variable 2
    [25, 15, 5]    # Variable 3
])
correlation_matrix = np.corrcoef(data)
[[ 1.          0.98198051 -0.98198051] Rows: V1, V2, V3 and Columns V1, V2, V3. Now check the correlation
 [ 0.98198051  1.         -1.        ]
 [-0.98198051 -1.          1.        ]]
```
#Both covariance and correlation measure relationship between variables, correlation provide standarized measure

# Normalization technique known as Standardization
to transform data into a common scale,. With standadization we will transform the data so each feature has a mean of 0 and deviation of 1.
# min-max scaling: 
features has a definite maximum and minimum value
# z-scale(standardization)
transforms the features of a dataset to have a mean of 0 and a standard deviation of 1.
```
from sklearn.preprocessing import MinMaxScaler
data = np.array([[10, 20])
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data)
[0.33333333 0.33333333]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
standardized_data = scaler.fit_transform(data)
[-1.06904497 -1.06904497]
```
# KMeans Clustering: 
K-means clustering is an iterative algorithm used to partition a dataset into k clusters. The algorithm aims to minimize the variance within each cluster by iteratively optimizing cluster centroids. Its a popular unsupervised machine learning algorithm used for clustering data into groups or clusters based on similarity. Choosing correct value of K using (elbow method and silhouette score)
```
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# Apply KMeans clustering for different values of K
x_input = data.loc[:, ['Age', 'SpendingScore']].values
k_values = range(2, 10)
inertia_values = []
silhouette_scores = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))
k_means=KMeans(n_clusters=4)
k_means.fit(x_input)
centroids = k_means.cluster_centers_
cluster_assignments = k_means.labels_
k_means.predict(new_data_points)
```
# Euclidean distance 
is commonly used in clustering algorithms like K-means to measure the similarity between data points. Its used in classification algorithms like k-nearest neighbors (KNN) to find the nearest neighbors of a query point.Euclidean is distance between data points and cluster centroids
``` distance = np.linalg.norm(point - centroid)  # Euclidean distance```
# Hierarchial Clustering
```import seaborn as sns
sns.clustermap(x)
``` 
# Principle Component Analyis(PCA) 
is dimensionality reduction technique used to transform high-dimensional data into a lower-dimensional space while preserving most of the variability in the data. PCA achieves this by identifying the principal components, which are linear combinations of the original features that capture the maximum variance in the data.
PCA computes the covariance matrix of the standardized data, which represents the pairwise relationships between the features.
PCA then performs eigenvalue decomposition on the covariance matrix to find its eigenvectors and eigenvalues. The eigenvectors represent the directions (principal components) of maximum variance in the data. The principal components with the highest eigenvalues capture the most variance in the data.
```
from sklearn.decomposition import PCA
x_input = dataframe.loc[:, ['driving_properties', 'interior', 'technology', 'comfort', 'household']].values
pca_num_components = 2
reduced_data = PCA(n_components=pca_num_components).fit_transform(dataframe)
results = pd.DataFrame(reduced_data,columns=['pca1','pca2'])
```
# Linear Regression:R-squared,MSE, RMSE, OOSRsquare 
commonly used metrics in regression analysis to evaluate performance of predictive models
# R-square: 
measure proportion of variance in dependent variable to independent variables between 0-1 (1 is good)
# MSE:
avg sqaured difference between predicted value to actual value. (Take avg of squared residuals[errors])
# RMSE:
square root of mse. if RMSE of housing price prediction is 100, it means avg diff is $100
# OOSRsquare: 
how well reg model generalizes to new or unseen data. High OOSRsquare means model is good for new
```
from skimpy import skim
from statsmodels.formula.api import ols
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import datasets #optional as its used to load publicly available dataset 
skim(df) #visualize all the columns, summary more like profiling
categorical_variables = ['month','day','hour','day_of_week','weekend'] #convert variables as categorical
df[categorical_variables] = df[categorical_variables].astype('category')
est = ols(formula="rentals ~ temp + rel_humidity", data=df).fit()
print(est.summary()) #provides the ols regression results with coeff, p values (<0.05), r-square etc
df_train, df_test = train_test_split(df, test_size=0.3) #split dataset
est_train = ols(formula="rentals ~ temp + rel_humidity + precipitation", data=df_train).fit()
test_pred = est_train.predict(df_test)
print('OOS R-squared: '+str(r2_score(df_test['rentals'],test_pred))) #out of sample r-square
```
# Gradient Descent:
It is an iterative optimization algorithm that efficiently finds the optimal coefficients for linear regression models. Its used to minimize the cost function (also known as the loss function). Iteratively updating the coefficients until convergence. Find the optimal coefficients for a LR model that best fits data
```
X = 2 * np.random.rand(100, 1)  # Feature (input) variable
y = 4 + 3 * X + np.random.randn(100, 1)  # Target variable
```
# Gradient Descent Parameters
```
eta = 0.1  # Learning rate: The learning rate is a hyperparameter that controls the size of the steps taken during the gradient descent optimization process
n_iterations = 1000
m = 100  # Number of instances
# Initialize coefficients
theta = np.random.randn(2, 1)
```
# Gradient Descent
```
for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta -= eta * gradients
# Print the final coefficients
print("Final Coefficients (Intercept, Slope):", theta.ravel())
```
# Avoid multicollinearity
(2 or more independent variables highly correlated, remove highly corr attributes to avoid errors, for instance if temp and temp_wb have high corr, then remove one of them to avoid errors)

# Linear Regression: 
Dependent variable is continuous and has a linear relation with the ind variables -houseprice
# Logistic Regression:
Dependent variable is categorical and represents binary outcomes (email is spam or not)
```
from statsmodels.formula.api import logit
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
df_train, df_test = train_test_split(df, test_size=0.3)
formula="default ~ installment + log_income + fico_score + rev_balance + inquiries + records"
est = logit(formula=formula, data=df_train).fit()
# apply the model (est) to the test data and make predictions
preds = est.predict(df_test)
df_test['predicted_probability'] = preds
# test for 'predicted_probability > 0.5, if yes assign will_default to 1, otherwise to 0
df_test['will_default'] = np.where(df_test['predicted_probability']>0.5, 1, 0)
```
# ConfusionMatrix:
Evaluate performance of classification model by summarizing the counts of true positive, true negative, false positive, and false negative predictions made by the model on a test dataset.
```
#Profit = (True Negatives × Profit per True Negative) − (False Positives × Loss per False Positive)
ConfusionMatrixDisplay.from_predictions(df_test['default'], df_test['will_default'], display_labels = ['No Default', 'Default'])
#print accuracy
print('Accuracy:'+str(accuracy_score(df_test['default'], df_test['will_default'])))
cm = confusion_matrix(df_test['default'], df_test['will_default'])
cm_profit = 1000*cm[0,0]-5000*cm[1,0]
print('Profit at Threshold of 0.5 = $'+str(cm_profit))
precision = tp / (tp + fp) # True positives vs all predicted positives.
recall = tp / (tp + fn) # True positives vs all actual positives.
f1_score = 2 * precision * recall / (precision + recall) # Harmonic mean
```
# Collaborative filtering: 
is a technique used in recommendation systems to make predictions or recommendations about user preferences based on the preferences. netflix, spotify etc. Userbased or Itembased
# Linear Optimization Models:
Maximize or Minimize the obj function 
```
#excel: Create Decision Variables, Constraints and Objective Functions and use Simplex LP (Data->Solver) 
from scipy.optimize import linprog
opt = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq)#lhs=decision variables, rhs=constraints
```
# The Gurobi Optimizer 
is a mathematical optimization software library for solving mixed-integer linear and quadratic optimization problems.
```
import gurobipy as gp
from gurobipy import GRB
# Create Gurobi model object - repository for all objects to be used in the model
mod1 = gp.Model ("price_model_1")
# Define decision variables
p = mod1.addVars(weeks, ub = 1)
#Set Objective Function #Use Linear regression demand model and get the coefficients
obj_fn = mod1.setObjective(p[w1] * (intercept + p_coeff*p[w1] + p1_coeff*1 + p2_coeff*1 + season_coeff[season[w1]]) +
                           p[w2] * (intercept + p_coeff*p[w2] + p1_coeff*p[w1] + p2_coeff*1 + season_coeff[season[w2]]) +
                           sum(p[w] * (intercept + p_coeff*p[w] + p1_coeff*p[w-1] + p2_coeff*p[w-2] + season_coeff[season[w]]) for w in weeks[2:]),
                          GRB.MAXIMIZE)
mod1.optimize()
```
# CART Model
Classification Regression Tree (Nonlinear Models for Regression and Classification):powerful tree-based algorithms commonly used in both classification and regression tasks,
```
from sklearn.tree import DecisionTreeClassifier, plot_tree
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X,y)
y_pred = tree.predict_proba(X_test)[:,1]
#compare with logit and check the results with decision tree. Mitigate overfitting, use cost complexity pruning
path = tree.cost_complexity_pruning_path(X,y)
alphas = path.ccp_alphas
trees = []
for alpha in alphas:
  tree = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
  tree.fit(X,y)
  trees.append(tree)
tree = DecisionTreeRegressor(random_state=42, ccp_alpha=alphas_small[opt_idx])
tree.fit(X,y)
tree.tree_.node_count, tree.tree_.max_depth
tree_pred = tree.predict(x_test)
tree_acc = tree.score(x_test, y_test)
from sklearn.tree import plot_tree
plt.figure(dpi=150)
plot_tree(tree, filled=True)
plt.show()

#Instead of predicting class labels, it predicts continuous target variables.
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor(random_state=42).fit(train,y_train)
y_pred = tree.predict(test)
tree = DecisionTreeRegressor(random_state=42) 
path = tree.cost_complexity_pruning_path(train, y_train)
alphas = path.ccp_alphas
```
# Random Forest 
is an ensemble learning method that constructs a multitude of decision trees during training and outputs the mode (for classification) or mean (for regression) of the predictions of the individual trees. Each decision tree in the Random Forest is trained on a bootstrap sample of the training data 
```
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=42).fit(train,y_train)
y_pred = rf.predict(test)
```
# XGBoost
is a Extreme gradient boosting algorithm that builds an ensemble of weak learners (usually decision trees) in a sequential manner. It iteratively trains new trees to correct the errors of the previous trees, focusing on the instances that were incorrectly predicted. Each new tree is trained on the residuals (the differences between the predicted and actual values) of the previous trees.
```
from xgboost import XGBRegressor
model = XGBRegressor(random_state=42)
model.fit(train, y_train)
y_pred = model.predict(test)
```
### Hyperparameter Optimization: GridSearch or RandomSearch
```
from sklearn.model_selection import GridSearchCV
param_grid = {'n_estimators': [10,50,100],'max_depth': [1,2,3,4,5],'learning_rate': [0.001, 0.01, 0.1, 0.3]}
gcv = GridSearchCV(model,param_grid=param_grid,cv=5,scoring='neg_mean_absolute_error')
gcv.fit(train, y_train)
gcv.best_params_
y_pred = gcv.best_estimator_.predict(test)
from sklearn.model_selection import RandomizedSearchCV
param_grid = {'n_estimators': range(10,1010,10),'max_depth': range(1,21),'learning_rate': np.linspace(0.001, 1.0, 50)}
rcv = RandomizedSearchCV(model,param_distributions = param_grid,cv=5,n_iter=10,scoring='neg_mean_absolute_error',random_state=42)
rcv.fit(train, y_train)
rcv.best_params_
y_pred = rcv.best_estimator_.predict(test)
```
# Neural Network
```coefficients = weights, intercepts = biases
Execute Logistic regression by combining intercept (adding biases) + coefficient of variable 1 to n(multipying inputs with weights), then running through sigmoid function to determine probability 1/1+e^-x
Transform k-dimensional input into n dimensional vector output using linear function and then feed into nonlinear (like sigmoid). Ater n transformations, you then feed into logistic regresssion function
Use linear and non linear functions repeatedly between outputs and inputs so that there are transformations, to smartly represent unstructured data
```
```
Neural network: Insert logistic or linear regression into linear functions, followed by non linear functions
Neurons: operations that involve linear function and receive inputs, add them up and send them thro non-linear functions, One neuron connected to another neuron within a network of connections
Layer: Vertical stack of neurons
Activation function: non linear functions (sigmoid) used inside each neuron including output
Input layer (inputs or variables with no transformations), output layer (final output from previuos of network procedures, sigmoid), rest is hidden layer
Dense or fully connected layer: layer with stack of neurons and numbers fed into every neuron in next layer
Deep neural network/deep learning: is a neural network with lots of hidden layers [number of layers is called depth of network] RESNET34
```
```
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, stratify=y)
### Define model in Keras
#Creating an NN  is usually just a few lines of Keras code. , We will start with a single hidden layer. 
#Since this is a *binary classification problem*, we will use a sigmoid activation in the output layer.
#get the number of columns and assign it to "num_columns"
num_columns = X_train.shape[1]
# Define the input layer. assign it to "input"
input = keras.Input(shape=num_columns)
# Feed the input vector to the hidden layer. Call it "h"
h = keras.layers.Dense(16, activation="relu", name="Hidden")(input)
# Feed the output of the hidden layer to the output layer. Call it "output"
output = keras.layers.Dense(1, activation="sigmoid", name="Output")(h)
# tell Keras that this (input,output) pair is your model. Call it "model"
model = keras.Model(input, output)
### Set Optimization Parameters
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
### Train the Model with fit
history = model.fit(X_train, y_train,epochs=100,batch_size=32,validation_split=0.2)
### Evaluate Model
score, acc = model.evaluate(X_test, y_test)
