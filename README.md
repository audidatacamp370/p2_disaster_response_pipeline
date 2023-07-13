# Udacity - Data Scientist Nanodegree Program
## Project - Disaster Response Pipeline

- [Installations](#inst)
- [Project Motivation](#promot)
- [File Description](#filedesc)
- [Instructions](#instruct)
- [Licensing, Authors and Acknowledgements](#license)

<a id='inst'></a>
## Installations
Anaconda Jupyter Notebook with Python3

### Packages
- sys
- re 
- Pandas
- Numpy
- SQLAlchemy
- Natural Language Toolkit (NLTK)
- Scikit-learn
- pickle

### Data
Disaster data from Appen (formally Figure 8)

<a id='promot'></a>
## Project Motivation
The main focus of this Project is to built a model for an API that classifies disaster messages.

<a id='filedesc'></a>
## Files

app

| - template

| |- master.html  # main page of web app

| |- go.html  # classification result page of web app

|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md

<a id='instruct'></a>
## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage


<a id='license'></a>
## Licensing, Authors and Acknowledgements

Thanks to Udacity for providing the project an the environment!

Thanks to Figure Eight for providing the dataset!
