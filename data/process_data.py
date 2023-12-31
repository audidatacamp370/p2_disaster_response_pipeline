import sys
import pandas as pd
#import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    loading massage data and categary data from csv-files
    
    :Input 
    messages_filepath: - path to disaster_messages.csv
    categories_filepath: - path to disaster_categories.csv
    :Output 
    df - merged dataframe
    """    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='inner', on='id')
    return df


def clean_data(df):
    """
    cleaning dataframe
    
    :Input 
    df: - merged dataframe of message and category data
    :Output 
    df - cleaned dataframe
    """      
    categories = df.categories.str.split(';', expand=True)
    row = categories.iloc[0]
    # rename the columns of `categories`
    categories.columns = list(map(lambda x: x[:-2], row)) 
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    
    df = pd.concat([df,categories], axis=1)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    # drop row where related column has the value 2
    df=df[df['related']!=2]
    
    # drop columns with only one unique value
    for column in df.iloc[:,4:].columns:
        if df[column].nunique()==1:
            df.drop(column, inplace=True, axis=1)

    
    return df

def save_data(df, database_filename):
    """
    saving dataframe to sql-database
    
    :Input 
    df: - cleaned dataframe
    database_filename: - name of sql database
    :Output 

    """      
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')
      


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()