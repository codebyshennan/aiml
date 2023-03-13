import sqlite3 as sql
import pandas as pd


class SQLite:
    def __init__(self, conn_string, table) -> None:
        self.query = "SELECT * FROM " + table
        self.conn_string = conn_string

    def parseDb(self):
        con = sql.connect(self.conn_string)
        df = pd.read_sql_query(self.query, con)

        # Be sure to close the connection
        con.close()

        return df
