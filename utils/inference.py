import pandas as pd
from .CustomerData import CustomerData


def predict_churn (data : CustomerData, preprocessor, model) :

    # to dataframe
    df = pd.DataFrame([data.model_dump()])

    # transform the pipeline to data
    processed_data = preprocessor.transform(df)

    # prediction
    y_pred = model.predict(processed_data)
    y_prob = model.predict_proba(processed_data)
    
    return {
        "churn_prediction" : bool(y_pred[0]),
        "churn_probability" : float(y_prob[0][1])
    }