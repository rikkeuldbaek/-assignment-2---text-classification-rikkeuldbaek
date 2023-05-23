### Language Assignment 2
## Cultural Data Science - Language Analytics 
# Author: Rikke Uldbæk (202007501)
# Date: 2nd of March 2023

#--------------------------------------------------------#
############### NEURAL NETWORK CLASSIFIER ###############
#--------------------------------------------------------#

# (please note that some of this code has been adapted from class sessions)

# Importing packages
# System tools
import os

# Data munging tools
import pandas as pd

# Save model and report
from joblib import dump, load

# Machine learning stuff
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

# Scripting
import argparse


############# Parser function #############
def input_parse():
    #initialise the parser
    parser = argparse.ArgumentParser()
    #add arguments
    parser.add_argument("--filename", type=str, default= "fake_or_real_news.csv", help= "Specify .csv file.") 
    parser.add_argument("--test_size", type= float, default =0.2, help= "Specify the proportion of the dataset to include in the test split.")
    parser.add_argument("--random_state", type= int, default = 666, help = "Specify random state.")
    parser.add_argument("--ngram_range", nargs='+', type= int, default=(1,2), help = "Specify the lower and upper boundary of the range of n-values for different n-grams to be extracted.")
    parser.add_argument("--lowercase", type= bool, default = True, help = "Specify whether lowercase should be True or False.")
    parser.add_argument("--max_df", type= float, default = 0.95, help= "Specify maximum threshold and ignore terms that have a document frequency strictly higher than the given threshold.")
    parser.add_argument("--min_df", type= float, default = 0.05, help = "Specify minimum threshold and ignore terms that have a document frequency strictly lower than the given threshold.")
    parser.add_argument("--max_features", type= int, default = 500, help = "Specify maximum number of features to extract.")
    parser.add_argument("--activation", type= str, default= "logistic", help= "Speficy activation function for hidden layers.")
    parser.add_argument("--hidden_layer_sizes", nargs='+', type= int, default=(30,30), help="Specify hidden layer sizes. More hidden layers will increase computational time. This must be specified without commas.")
    parser.add_argument("--max_iter", type= int, default=1000, help = "Specify maximum number of iterations.")
    # parse the arguments from the command line 
    args = parser.parse_args()
    
    #define a return value
    return args #returning arguments


#############  Load data  ############# 
import vectorizer as vec
args = input_parse()
X,y = vec.load_data(args.filename)
X_train_feats, X_test_feats, y_train, y_test, vectorizer = vec.vectorize_function(X, y, args.test_size, args.random_state, args.ngram_range, args.lowercase, args.max_df, args.min_df, args.max_features)
 



############# Classifying and predicting #############
def neural_network_model(activation, hidden_layer_sizes, max_iter, random_state):

    print("Initializing neural network classifier..")

    #############  Classify  ############# 
    classifier = MLPClassifier(activation = activation, # default "logistic"
                            hidden_layer_sizes = hidden_layer_sizes, # default 20 nodes in first hidden layer, 10 in next hidden layer
                            max_iter= max_iter,  # default 1000
                            random_state = random_state) # default 666

    print("Activation =", activation)
    print("Number of hidden layer sizes =", hidden_layer_sizes)
    print("Maximum number of iterations =", str(max_iter))

    # Fitting the model
    classifier.fit(X_train_feats, y_train)

    # Extract predictions of y
    y_pred = classifier.predict(X_test_feats) 


    ############# Evaluate #############
    # Calculating metrics for model performance
    classification_metrics = metrics.classification_report(y_test, y_pred) 

    return classification_metrics, classifier



############# Save report and model #############
def save_NN_results(classifier_metrics, classifier):

    # Save the classification report in the folder "out"
    # Define out path
    outpath_metrics_report = os.path.join(os.getcwd(), "out", "NN_metrics_report.txt")

    # Save the metrics report
    file = open(outpath_metrics_report, "w")
    file.write(classifier_metrics)
    file.close()

    # Save the trained model to the folder called "models"
    # Define out path
    outpath_classifier = os.path.join(os.getcwd(), "models", "NN_classifier.joblib")

    # Save model
    dump(classifier, open(outpath_classifier, 'wb'))

    print( "Saving the neural network metrics report in the folder ´out´")
    print( "Saving the neural network model in the folder ´models´")



############# Main function #############
def main():
    # input parse
    args = input_parse()
    # pass argument to neural network function
    classification_metrics, classifier = neural_network_model(args.activation, tuple(args.hidden_layer_sizes), args.max_iter, args.random_state)
    save_NN_results(classification_metrics, classifier)

if __name__ == '__main__':
    main()



