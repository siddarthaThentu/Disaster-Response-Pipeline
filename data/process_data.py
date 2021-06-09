import sys
import pandas as pd
from sqlalchemy import create_engine
import psycopg2

db_name = 'postgresql://hyjeoyyisyahqy:794f9c1fa282a80697e89b0fa2cd24cc08808984c0d2280ee7cbff6f4a911a69@ec2-54-197-100-79.compute-1.amazonaws.com:5432/d6rf827scnq3ud'

def load_data(messages_filepath, categories_filepath):
    """ 
    Loading Data From Multiple Files. 
  
    Merging Them Together To Form A Single DataFrame And Returning It. 
  
    Parameters: 
    messages_filepath (str): Filepath Where Messages Are Stored.
    categories_filepath (str): Filepath Where Categories Are Stored.
    
    Returns: 
    merged_df (DataFrame): Combined DataFrame Object
  
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = messages.merge(categories,on='id')
 
    return df


def clean_data(df):
    """ 
    Data Cleaning Process. 
  
    Takes In A Pandas DataFrame And Seperates The Categories Into Individual
    Category Columns And Then Merging Them Back To The DataFrame ,Finally 
    Returning New DataFrame. 
  
    Parameters: 
    df (DataFrame): Pandas DataFrame Object Containing The Data.
  
    Returns: 
    cleaned_df (DataFrame): Pandas DataFrame Object Containing The Cleaned Data.
  
    """
    categories = df['categories'].str.split(';',expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x:x[:-2])
    categories.columns = category_colnames
    
    for column in categories:
    	# setting each value to be the last character of the string
    	categories[column] = categories[column].str[-1]
    	# converting column from string to numeric
    	categories[column] = pd.to_numeric(categories[column])

    # replacing categories column in df with new category columns.
    df = df.drop('categories',axis = 1)
    # concatenating the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis = 1)
    # removing duplicates
    cleaned_df = df.drop_duplicates(keep = 'first')
    
    return cleaned_df


def save_data(df, database_filename=db_name):
    """ 
    Save To Database Method. 
  
    Saving The Cleaned Pandas DataFrame Into A SQLite Database. 
  
    Parameters: 
    df (DataFrame): Pandas DataFrame Object Containing The Cleaned Data.
    database_filename (str) : Filename Of Database.
  
    """
    #db_name = 'sqlite:///{}'.format(database_filename)
    
    engine = create_engine(database_filename)
    conn = engine.connect()
    table_name = 'DisasterResponse'
    #df.to_sql('DisasterResponse', engine, index=False)  
    try:
        df.to_sql(table_name,conn)
    except Exception as e:
        print(e)
    else:
        print("Table created succesfully")
    finally:
        conn.close()


def main():
    if len(sys.argv) == 3:

        messages_filepath, categories_filepath = sys.argv[1:]
        database_filepath = db_name

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