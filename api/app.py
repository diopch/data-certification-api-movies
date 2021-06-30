
from fastapi import FastAPI
import pandas as pd
from joblib import load
import csv

app = FastAPI()


# define a root `/` endpoint
@app.get("/")
def index():
    return {"ok": True}


# Implement a /predict endpoint
@app.get("/predict")
def predict(original_title, title,release_date,duration_min,
            description,budget,original_language,status,
            number_of_awards_won,number_of_nominations,has_collection,
            all_genres,top_countries,number_of_top_productions,available_in_english):
    
    dictionnary = dict(
        original_title=original_title,
        title=title, 
        release_date= release_date, 
        duration_min=duration_min,
        description=description, 
        budget=budget,
        original_language =original_language, 
        status=status,
        number_of_awards_won =number_of_awards_won, 
        number_of_nominations=number_of_nominations, 
        has_collection=has_collection,
        all_genres=all_genres, 
        top_countries=top_countries, 
        number_of_top_productions=number_of_top_productions,
        available_in_english=available_in_english 
        )
    with open('mycsvfile.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
        w = csv.DictWriter(f, fieldnames =dictionnary.keys())
        w.writeheader()
        w.writerow(dictionnary)
            
    X_predict = pd.read_csv('mycsvfile.csv')  #[list(dictionnary.values())]
    model = load("model.joblib")
    popularity = model.predict(X_predict)
    # return X_predict
    return {'popularity': popularity}
