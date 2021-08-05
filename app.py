import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import r2_score, mean_squared_error

def evaluate_model(model, X_test, y_test, modelName, DataImb):
    print('------------------------------------------------')
    print("Model ", modelName, end="\n")
    print("Data Balancing Type ", DataImb)
    ### Model must be ran outside the function
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print("R2 Score", r2)
    print("RMSE", rmse)
    return [modelName, DataImb, r2, rmse]

# => Load Model
def load_model(modelfile):
    model = pickle.load(open(modelfile, "rb"))
    return model

# => Retrievng Values
def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value

def get_fvalue(val):
    feature_dict = {"yes": 1, "no": 0}
    for key, value in feature_dict.items():
        if val == key:
            return value

def get_tvalue(val):
    val = pd.to_datetime(val)
    val = round((((val.value - 1628122036000000000) / 10 ** 9) / 3600), 2)
    return val



'''
Predicting Delivery Time

Stuber Logistics company is one of the leading logistics company in Nigeria known for smooth workflow and customers satisfaction was indeed their motto. Lagos state traffic congestion has been the major challenge for a logistics company and it's getting worse every day as government is not doing enough at reducing traffic congestion.

Stuber has put in lots of strategies in place to optimize their delivery time but none seems to be working as expected which left them with no option than to follow the new trend(Artificial intelligence)
'''
'''
### How it Works?

Enter the required values and click on the **Predict** Button.
'''

## Data Columns
features = ['Order_Day_of_Month', 'Order_Week_of_Month', 'Delivery_MonthDay',
            'Delivery_Weekday', 'DistanceCovered_KM', 'Temperation',
            'Precipitation_in_millimeters_CAT', 'Time_of_Order_in_hours',
            'Time_of_Confirmation_in_hours', 'Arrival_at_Pickup_Time_in_hours',
            'Pickup_Time_in_hours', 'Delivery_Time_in_hours', 'Purpose_CAT',
            'Platform_CAT']

## Dictionary/Labels
platform_map = {'P1': 0, 'P2': 1, 'P3': 2, 'P4': 3}
purpose_map = {'Commercial': 0, 'Personal': 1}
raining_map = {"Yes": 1, "No": 0}

## Prediction Form
st.sidebar.title("Prediction Form")

### Inputs

st.sidebar.subheader("Orders")

Order_Day_of_Month = st.sidebar.number_input("Order_Day_of_Month", 1, 31, 1, 1)
Order_Week_of_Month = st.sidebar.number_input("Order_Week_of_Month", 1, 5, 1, 1)
Delivery_MonthDay = st.sidebar.number_input("Delivery_MonthDay", 1, 31, 1, 1)
Delivery_Weekday = st.sidebar.number_input("Delivery_Weekday", 1, 7, 1, 1)
DistanceCovered_KM = st.sidebar.number_input("DistanceCovered_KM", 1, 100, 1, 1)
Temperation = st.sidebar.number_input("Temperation", 1, 100, 1, 1)
Precipitation_in_millimeters_CAT = st.sidebar.radio("Was it Raining?", tuple(raining_map.keys()))

st.sidebar.subheader("Time Values [H:M:S]: Enter something like 2:40:05 AM")

Time_of_Order_in_hours = st.sidebar.text_input("Time_of_Order_in_hours", '12:00:00 AM')
Time_of_Confirmation_in_hours = st.sidebar.text_input("Time_of_Confirmation_in_hours", '12:00:00 AM')
Arrival_at_Pickup_Time_in_hours = st.sidebar.text_input("Arrival_at_Pickup_Time_in_hours", '12:00:00 AM')
Pickup_Time_in_hours = st.sidebar.text_input("Pickup_Time_in_hours", '12:00:00 AM')
Delivery_Time_in_hours = st.sidebar.text_input("Delivery_Time_in_hours", '12:00:00 AM')

st.sidebar.subheader("Vechicles")

Purpose_CAT = st.sidebar.radio("Purpose of Vehicle", tuple(purpose_map))
Platform_CAT = st.sidebar.radio("What Platform do you use?", tuple(platform_map))

## Feature Values
feature_values = [Order_Day_of_Month, Order_Week_of_Month,
        Delivery_MonthDay, Delivery_Weekday,
        DistanceCovered_KM, Temperation,
        get_value(Precipitation_in_millimeters_CAT, raining_map),
        get_tvalue(Time_of_Order_in_hours),
        get_tvalue(Time_of_Confirmation_in_hours), get_tvalue(Arrival_at_Pickup_Time_in_hours),
        get_tvalue(Pickup_Time_in_hours), get_tvalue(Delivery_Time_in_hours),
        get_value(Purpose_CAT, purpose_map), get_value(Platform_CAT, platform_map)]

## Pretty Result from JSON
pretty_results = {'Order_Day_of_Month': Order_Day_of_Month, 'Order_Week_of_Month': Order_Week_of_Month,
            'Delivery_MonthDay': Delivery_MonthDay, 'Delivery_Weekday': Delivery_Weekday,
            'DistanceCovered_KM': DistanceCovered_KM, 'Temperation': Temperation,
            'Precipitation_in_millimeters_CAT': Precipitation_in_millimeters_CAT,
            'Time_of_Order_in_hours': Time_of_Order_in_hours,
            'Time_of_Confirmation_in_hours': Time_of_Confirmation_in_hours,
            'Arrival_at_Pickup_Time_in_hours': Arrival_at_Pickup_Time_in_hours,
            'Pickup_Time_in_hours': Pickup_Time_in_hours, 'Delivery_Time_in_hours': Delivery_Time_in_hours,
            'Purpose_CAT': Purpose_CAT,
            'Platform_CAT': Platform_CAT}

"""
### Entered Values
"""
if st.checkbox("Show My Values"):
    st.json(pretty_results)

### Reshaped Values
single_sample = np.array(feature_values).reshape(1, -1)
print('B', single_sample)
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# single_sample = sc.fit(single_sample)




"""
### Likelihood of Delivery
"""

if st.button("Predict"):
    '''
    ## Results

    '''
    ## COME BACK HERE
    model = load_model('model.pk_dsn_0805')
    prediction = model.predict(single_sample)

    pretty_results["p_Delivery_in_min"] = prediction[0]
    prediction_table = pd.DataFrame(pretty_results, index=["Proba"])
    st.caption("Estimated time of delivery in Minutes")
    st.success(pretty_results['p_Delivery_in_min'])
    st.caption("Prediction Table")
    st.table(prediction_table)
    st.warning("[+-] From our model the Delivery Time would a variance of (give or take) 0.004")

## COME BACK HERE

## Explore
st.sidebar.subheader("Explore")
st.sidebar.info('''
See other predictive web applications that could help you make better decisions.

[Surviving Heart Failure](https://heartfailurepredictor-afl.herokuapp.com/)

[Likehood of Stroke](https://stroke-predictor-afl.herokuapp.com/)

[Trivia: Drug Classifier](https://drug-classifier.herokuapp.com/)

[Predicting Customer Churn](https://telecomms-churn.herokuapp.com/)

''')