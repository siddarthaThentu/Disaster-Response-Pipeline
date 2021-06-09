import sys
import os
import nltk
nltk.download(['punkt', 'wordnet','averaged_perceptron_tagger'])
import pandas as pd
import numpy as np
import pickle
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sqlalchemy import create_engine
import re
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenFile import tokenize,StartingVerbExtractor

url_regex = "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
db_name = 'postgresql://tqdqfmxgrgunzx:a6d3564a45d7148a5e09817cead82db91e8b431f7521be123af79c92afd0d92c@ec2-54-91-188-254.compute-1.amazonaws.com:5432/d7saa2toh2oscb'


def load_data(database_filepath):
    
    #db_name = 'sqlite:///{}'.format(database_filepath)
    engine = create_engine(db_name)
    conn = engine.connect()
    table_name = 'DisasterResponse'

    df = pd.read_sql("select * from \"DisasterResponse\"",conn)
    
    X = df['message'].values
    Y = df.iloc[:,5:]

    category_names = list(Y.columns)
    
    print(Y)
    return X,Y,category_names

# def tokenize(text):
    
#     detected_urls = re.findall(url_regex, text)
#     for url in detected_urls:
#         text = text.replace(url, "urlplaceholder")
    
#     # tokenizing
#     tokens = word_tokenize(text)
    
#     # lemmatizing
#     lemmatizer = WordNetLemmatizer()
    
#     clean_tokens = []
#     for tok in tokens:
#         clean_tok = lemmatizer.lemmatize(tok).lower().strip()
#         clean_tokens.append(clean_tok)

#     return clean_tokens


def build_model():
    
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
            ])
            ),

            ('starting_verb', StartingVerbExtractor())
        ])),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    parameters = { 'clf__estimator__n_estimators' : [50,60,70,80] }
    model = GridSearchCV(pipeline, param_grid=parameters)

    return model



def evaluate_model(model, X_test, Y_test, category_names):
    
    Y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print("Category:", category_names[i],"\n", classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))
        print('Accuracy of %25s: %.2f' %(category_names[i], accuracy_score(Y_test.iloc[:, i].values, Y_pred[:,i])))


def save_model(model, model_filepath):
    
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


def main():
    if len(sys.argv) == 2:
        database_filepath = db_name
        model_filepath = sys.argv[1]
        print("model_filepath = ",model_filepath)
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