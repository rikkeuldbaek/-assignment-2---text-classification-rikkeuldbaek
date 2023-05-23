[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10326455&assignment_repo_type=AssignmentRepo)

# **Assignment 2 - Text classification benchmarks**
## **Cultural Data Science - Language Analytics** 
#### Author: Rikke Uldbæk (202007501)
#### Date: 2nd of March 2023
<br>


# **2.1 GitHub link**
The following link is a link to the GitHub repository of assignment 2 in the course Language Analytics (F23.147201U021.A). Within the GitHub repository all necessary code are provided to reproduce the results of the assignment.

https://github.com/rikkeuldbaek/assignment-2-text-classification-rikkeuldbaek

<br>

# **2.2 Description**

For this assignment, I have trained two binary classification models on a *Fake or Real News Dataset*, in order to classify whether the news was real or fake. These two classification models are: a logistic regression classifier and a neural network classifier, and the two of them have been designed using the available tools from ```scikit-learn```. Finally, the performance of the two classifiers is evaluated in the results section. 

<br>

# **2.3 Data**
The *Fake or Real News Dataset* consists of four columns: an ID, a title (headline), text (articles), and a label (fake or real news). For this assignment I have used the text column (articles) to classify the label column (fake or real news), and the number of articles in the text column is approximately 6000. The data is available via Kaggle, please see resources for further information. 

<br>


# **2.4 Repository Structure**
The scripts require a certain folder structure, thus the table below presents the required folders and their description and content.

|Folder name|Description|Content|
|---|---|---|
|```src```|model and data scripts|```vectorizer.py```, ```logistic_regression_classifier.py```, ```neural_network.py```|
|```out```|classification reports|```LR_metrics_report.txt```, ```NN_metrics_report.txt```|
|```models```|models|```LR_classifier.joblib```, ```NN_classifier.joblib```, ```mtfidf_vectorizer```|


The ```vectorizer.py``` script located in ```src``` produces training and test data. The ```logistic_regression_classifier.py``` and the ```neural_network.py``` located in the ```src``` 
 folder produce models which are saved in the folder ```models``` and classification reports which are saved in the folder ```out```. 


<br>

# **2.5 Usage and Reproducibility**
## **2.5.1 Prerequisites** 
In order for the user to be able to run the code, please make sure to have bash and python 3 installed on the used device. The code has been written and tested with Python 3.9.2 on a Linux operating system. In order to run the provided code for this assignment, please follow the instructions below.

<br>

## **2.5.2 Setup Instructions** 
**1) Clone the repository**
```python
git clone https://github.com/rikkeuldbaek/assignment-2-text-classification-rikkeuldbaek
 ```

 **2) Setup** <br>
Setup virtual environment (```LA2_env```) and install required packages.
```python
bash setup.sh
```
<br>

## **2.5.3 Run the script** 
Please execute the command below in the terminal to run the ```logistic_regression_classifier.py``` and ```neural_network.py``` scripts and produce the results of the assignment.
```python
bash run.sh
```

<br>

## **2.5.4 Script arguments**
In order to make user-specific modifications, the vectorizer and two models have the following arguments stated below. These arguments can be modified and adjusted in the ```run.sh``` script. If no modifications is added, default parameters are run. Please write --help in continuation of the code below instead of writing an argument, if help is needed. The ```vectorizer.py``` is automatically called upon when running both model scripts, thus the arguments for ```vectorizer.py``` must be parsed through the ```logistic_reg_classifier.py``` and the ```neural_network_classifier.py``` script. Very important, an ```vectorizer.py``` argument, needs to be parsed in both model scripts simultaneously for the models to use the same data. 


The ```vectorizer.py``` takes the following arguments:

|Argument|Type|Default|
|---|---|---|
|--filename|string| "fake_or_real_news.csv"|
|--test_size|float| 0.2|
|--random_state|integer|666|
|--ngram_range|integer|(1,2)|
|--lowercase|boolean|True|
|--max_df|float|0.95|
|--min_df|float|0.05 |
|--max_features|integer|500|

<br>

The ```logistic_reg_classifier.py``` takes the following argument:

|Argument|Type|Default|
|---|---|---|
|--random_state|integer|666|

<br>

The ```neural_network_classifier.py``` takes the following arguments:

|Argument|Type|Default|
|---|---|---|
|--activation|string | "logistic"|
|--hidden_layer_sizes|integer |(30,30)|
|--max_iteration|integer |1000 |
|--random_state|integer|666|


<br>

**Important to note** <br>
The ```hidden_layer_sizes``` and the ```ngram_range``` argument must be specified without commas in the ```run.sh``` script, see examples of such:

````python 
python src/neural_network_classifier.py --hidden_layer_sizes 30 30
python src/vectorizer.py --ngram_range 1 2
````

<br>


# **2.6 Results**
From the classification reports located in the ```out``` folder, we can observe that the two classifiers have an almost identical performance. Both the logistic regression classifier and the neural network classifier have an accuracy score of 88%. The logistic regression classifier shows an F1 score of 89% for *fake news* and an F1 score of 88% for *real news*. In comparison the neural network classifier shows an F1 score of 88% for *fake news* and an F1 score of 88% for *real news*. Hence the results are pretty much identical. 

<br>

# **Resources**

**[Scikit-learn documentation - TfidfVectorizer()](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)**
(https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

**[Scikit-learn documentation - LogisticRegression()](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)**
(https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

**[Scikit-learn documentation - MLPClassifier()](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html )**
(https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html )

**[Data - Fake or Real News](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news)**
(https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news)