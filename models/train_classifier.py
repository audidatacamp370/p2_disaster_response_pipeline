import sys
import re
#import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords','omw-1.4'])

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix, classification_report, make_scorer, accuracy_score, f1_score, fbeta_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import pickle

def load_data(database_filepath):
    """
    Loads database from SQL 
    
    :Input 
    database_filepath: - path to the cleaned database saved by data/process_data.py script
    :Output 
    X: - Input Messages  
    Y: - Output Categories
    category_names: - Names of output categories
    """    
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df.message.values
    Y = df.iloc[:,4:]
    category_names=list(Y.columns)
    return X, Y, category_names

def tokenize(text):
    """
    normalize, tokenize and lemmatize text
    
    :Input 
    text: - row text messages
    :Output 
    clean_tokens - normalized, tokenized and lemmatized text
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text) 

    clean_tokens = []
    for tok in tokens:
        clean_tok = WordNetLemmatizer().lemmatize(tok).lower().strip()
        if clean_tok not in stopwords.words("english"):
            clean_tokens.append(clean_tok)
    
    return clean_tokens

def build_model():
    """
    building a machine learnign pipeline
    
    :Input 
    
    :Output 
    model - machine learning model
    """
    pipeline=Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
        ])),

        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
        
    
    parameters = {
        'features__text_pipeline__tfidf__smooth_idf': [True, False],
        'clf__estimator__n_estimators': [50,75],
        'clf__estimator__learning_rate': [0.75, 1]
        }

#    model = pipeline
    model = GridSearchCV(pipeline, param_grid=parameters, cv=5, n_jobs=-1, verbose=3)

    return model

def evaluate_model(model, X_test, Y_test, category_names):
    """
    evaluating the machine learnign pipeline
    
    :Input 
    model: - machine learning model 
    X_test: - test set of messages
    Y_text: - test set of categories
    category_names: - Names of output categories
    :Output 
    prints the Accuracy of the model predictions (Y_pred) for each category
    """
    Y_pred = model.predict(X_test)
    for category in category_names:
#        print(category,"\n :", classification_report(Y_test[category].values, pd.DataFrame(Y_pred, columns=category_names)[category]) )
        print("Accuray", category,"\n :", accuracy_score(Y_test[category].values, pd.DataFrame(Y_pred, columns=category_names)[category]) )

def save_model(model, model_filepath):
    """
    saving machine learning model
    
    :Input 
    model: machine learning model
    model_filepath: path to save file
    :Output 
    
    """    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()