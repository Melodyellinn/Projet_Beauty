import pandas as pd 
import sqlite3 as sql
from datetime import datetime
import shutil as sh

new_data = pd.read_csv("tmp_data/new_data.csv")
new_data["date"]= new_data["date"].apply(lambda x:  \
                                         datetime.strptime(str(x),"%Y%m%d"))


new_data["fullVisitorId"] = new_data["fullVisitorId"].astype("string")
new_data_train = new_data.copy().drop('will_buy_on_return_visit',axis = 1)

conn = sql.connect('C:/Users/Simplon/Desktop/PROJET_FINAL_BEAUTY/App_Beauty_E1/Application_prod.db')
new_data.to_sql("Test_data",conn, if_exists ="append")
new_data_train = pd.read_sql("SELECT * FROM Test_data",conn)
new_data_train.to_sql('Raw_data', conn,if_exists="replace")



conn.close()
year = str(new_data['date'].min())[:4]
month = str(new_data["date"].min())[5:7]
day = str(new_data["date"].min())[8:10]
sh.move("tmp_data/new_data.csv", f"archive_data/new_data_{year}_{month}_{day}.csv")