import glob
import pandas as pd
import os
import duckdb

con = duckdb.connect('formulaone.db')

files =  glob.glob('D:\\python_projects\\formulaone\\rawfiles\\*.csv')

for file in files:
    title = os.path.splitext(os.path.basename(file))[0]
    df = pd.read_csv(file)
    con.execute(f"CREATE OR REPLACE TABLE {title} AS SELECT * FROM df")

con.close()