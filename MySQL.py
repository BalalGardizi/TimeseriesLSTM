from sqlalchemy import create_engine
import pandas as pd
import numpy as np

# this file is specific to the laptop of Abraham.
# If you want to run the main.py you have to get a local database like MySQL. You can partly re-use this code
def dataSQL():
    SQLALCHEMY_DATABASE_URI = create_engine('mysql+pymysql://root:x^g7FU%y1IeMWa$cGw*O@localhost/pureballast')
    database = pd.read_sql_query("""SELECT * FROM shipData""", SQLALCHEMY_DATABASE_URI)
    database = database.replace('', np.nan)
    database['log_time'] = pd.to_datetime(database['log_time'])
    database['Code'] = pd.to_numeric(database['Code'])
    database['Value'] = pd.to_numeric(database['Value'])
    # database['A_W'] = database['A_W'],
    return (database)
