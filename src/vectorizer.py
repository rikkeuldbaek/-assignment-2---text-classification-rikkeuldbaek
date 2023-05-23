### Language Assignment 2
## Cultural Data Science - Language Analytics 
# Author: Rikke Uldbæk (202007501)
# Date: 2nd of March 2023

#--------------------------------------------------------#
################### VECTORIZING ################### 
#--------------------------------------------------------#

# (please note that some of this code has been adapted from class sessions)

# Importing packages
# system tools
import os

# data munging tools
import pandas as pd

# Machine learning stuff
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, ShuffleSplit

# Saving vectorizer
from joblib import dump, load

# Scripting
import argparse


############# Feature extraction #############
def load_data(filename):
    
    # Read in data
    file = os.path.join("in", filename)
    data = pd.read_csv(file, index_col=0)
    print("Reading in " + filename + "...")

    # Divide data into X and y
    X = data["text"]
    y = data["label"]

    return X,y 



def vectorize_function(X, y, test_size, random_state, ngram_range, lowercase, max_df, min_df, max_features):

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X,          # texts for the model
                                                        y,          # classification labels
                                                        test_size = test_size,   # default 80/20 split
                                                        random_state=random_state) # random state for reproducibility

    print("Splitting data with test size " + str(test_size))
    

    ############# Vectorize and feature extraction (TF-IDF vectorizer) #############
    vectorizer = TfidfVectorizer(ngram_range = ngram_range,     # default unigrams and bigrams (1 word and 2 word units)
                                lowercase =  lowercase,       # default lowercase
                                max_df = max_df,           # default remove very common words
                                min_df = min_df,           # default remove very rare words
                                max_features = max_features)      # default keep only top 500 features

    print("The lower and upper boundary of the range of n-values for different n-grams = " + str(ngram_range))
    print("Converting data to lowercase = " + str(lowercase))
    print("Setting max_df threshold = " + str(max_df))
    print("Setting min_df threshold = " + str(min_df))
    print("Extracting top " + str(max_features) + " features...")

    # Extract features from train and test set using the TF-IDF vectorizer
    X_train_feats = vectorizer.fit_transform(X_train) # fitting vectorizer on training data
    X_test_feats = vectorizer.transform(X_test) # fitting vectorizer on test data

    return X_train_feats, X_test_feats, y_train, y_test, vectorizer

 
############# Save vectorizer #############

def save_results(vectorizer):
    # Define out path
    outpath_vectorizer = os.path.join(os.getcwd(), "models", "mtfidf_vectorizer.joblib")

    # Save vectorizer in the folder "models"
    dump(vectorizer, open(outpath_vectorizer, 'wb') )

    print("Saving the TF-IDF vectorizer in the folder ´models´")


############# Main function #############

# Define a main function
def main():
    # input parse
    args = input_parse()
    # pass arguments to vectorize function
    X,y = load_data(args.filename)
    vectorizer = vectorize_function(X, y, args.test_size, args.random_state, tuple(args.ngram_range), args.lowercase,  args.max_df, args.min_df,args.max_features)
    save_results(vectorizer)

if __name__ == '__main__':
    main()



