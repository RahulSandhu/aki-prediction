import os

import pandas as pd
import pymysql


def execute_query(con: pymysql.connections.Connection, query: str) -> pd.DataFrame:
    """
    Run a SQL query and return the results as a DataFrame.

    Inputs:
        - con: Database connection object.
        - query (str): SQL query string to execute.

    Outputs:
        - pd.DataFrame: Query results as a DataFrame.
    """
    with con.cursor() as cursor:
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
        return pd.DataFrame(rows, columns=columns)  # type: ignore[arg-type]


if __name__ == "__main__":
    # Define connection parameters
    HOST = ""
    USER = ""
    PASSWORD = ""
    DATABASE = ""
    PORT = 3306

    # Connect to database
    con = pymysql.connect(
        host=HOST,
        user=USER,
        password=PASSWORD,
        database=DATABASE,
        port=PORT,
    )

    # Read query
    with open("cohort.sql", "r", encoding="utf-8") as file:
        query = file.read()

    # Execute query
    df = execute_query(con, query)

    # Exploratory analysis
    print(df.head())
    print(df.info())
    print(df.describe())

    # Save data
    os.makedirs("../../data/raw/", exist_ok=True)
    df.to_csv("../../data/raw/cohort.csv", index=False)

    # Close connection
    con.close()
