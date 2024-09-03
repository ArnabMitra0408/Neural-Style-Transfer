#importing libraries
import numpy as np
import os

import yaml
import zipfile
import logging
import mysql.connector
from mysql.connector import Error



class sql_connect:
    def __init__(self):
        try:
            self.connection = mysql.connector.connect(
                host='localhost',
                database='neuralstyletransfer',
                user='root',
                password='root'
            )
            if self.connection.is_connected():
                self.cursor = self.connection.cursor()
                print("Connected to MySQL database")
        except Error as e:
            print(f"Cannot Connect to DB {e}")
            self.connection = None

    def execute_query(self, query, params=None):
        try:
            if self.connection is not None:
                if params:
                    self.cursor.execute(query, params)
                else:
                    self.cursor.execute(query)
                self.connection.commit()
                print(f"Query executed: {query}")
            else:
                print("No database connection.")
        except Error as e:
            print(f"Failed to execute query: {e}")

    def close_connection(self):
        if self.connection is not None and self.connection.is_connected():
            self.cursor.close()
            self.connection.close()
            print("MySQL connection is closed")


class Custom_Handler(logging.Handler):
    def __init__(self, db):
        super().__init__()
        self.db = db

    def emit(self, record):
        try:
            if record:
                query = """
                    INSERT INTO LOGS (LevelName, Message, DateCreated) 
                    VALUES (%s, %s, SYSDATE())
                """
                params = (record.levelname, record.msg)
                self.db.execute_query(query, params)
        except Exception as e:
            print(f"Failed to log message to database: {e}")


db = sql_connect()
custom_handler = Custom_Handler(db)
logger = logging.getLogger('NST')
logger.setLevel(logging.DEBUG)
logger.addHandler(custom_handler)

def read_params(config_path:str) -> dict:
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config