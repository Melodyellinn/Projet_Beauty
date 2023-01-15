import pandas as pd
from sklearn.preprocessing import LabelEncoder
import mlflow
import sqlite3 as sql
from datetime import timedelta, datetime


conn = sql.connect('Application_prod.db')

data = pd.read_sql("""SELECT * 
                      FROM Raw_data""", conn)

data["date"] = pd.to_datetime(data["date"])
date = "20221001"
#today_date  = datetime.today()

today_date = datetime.strptime(date,"%Y%m%d")
last_date = today_date - timedelta(days=7)
focus_data = data.copy()[data['date'].between(last_date,today_date)]

print("---------------------------------------------------------------------------------")

df = focus_data.set_index('fullVisitorId')
df = df.copy()[df["country"].isin(["United States", "France", 
                                                "India", "China", "Germany",
                                                "Canada", "(not set)"])]
#change categorial columns
categorial = df[['channelGrouping','deviceCategory', 'country']]

for i in categorial.columns:
    df[i]= LabelEncoder().fit_transform(df[i])
    df[i].unique()
    
df = df.drop(['medium','date',"index"], axis= 1)
df = df.dropna(axis=0)

print("---------------------------------------------------------------------------------")

logged_model = 'runs:/1489eb854fb249f388e3404ab0b9fef4/model_xgboost'
# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)
# Predict on a Pandas DataFrame.
result = loaded_model.predict(pd.DataFrame(df))

print("---------------------------------------------------------------------------------")

data_result = focus_data.copy()[focus_data.country.isin(["United States", "France", 
                                                "India", "China", "Germany",
                                                "Canada", "(not set)"])].dropna()
data_result['Predict'] = result

result_to_save = data_result[["index","fullVisitorId","date","Predict"]]


print("---------------------------------------------------------------------------------")


try:
    
    check_date = pd.read_sql("SELECT * FROM Result_data",conn)
    print("Data existing wait to compare")
    
    check_date["date"] = pd.to_datetime(check_date["date"])
    if check_date[check_date["date"].between(last_date,today_date)].count()[0]!= 0 :
        print("Already Trained Data")
        if check_date[check_date["date"].between(last_date,today_date)].count()[0]<(result_to_save.count()[0]*0.85):
            pourcent_trained = round(check_date[check_date["date"].between(last_date,today_date)].count()[0] / result_to_save.count()[0] * 100,2)
            print(f"Data partialy to append check duplicate afterward.\n {pourcent_trained} % of the data already trained")
            result_to_save.to_sql("Result_data",conn, if_exists ="append",index=False)
        else: 
            print("Data trained to more than 85% data dropped")
            
    else:
        print("New Fresh Run append")
        result_to_save.to_sql("Result_data",conn, if_exists ="append",index=False)
        
        
except:
    print("first Run of infering creating table")
    
    result_to_save.to_sql("Result_data",conn, if_exists ="append",index=False)
