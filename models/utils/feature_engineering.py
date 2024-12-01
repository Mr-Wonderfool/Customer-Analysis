import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(data_dir):
    """Loading and preprocessing of data
    - drop nan values
    - Extract the **"Age"** of a customer by the **"Year_Birth"** indicating the birth year of the respective person
    - Create another feature **"Spent"** indicating the total amount spent by the customer in various categories over the span of two years.
    - Create another feature **"Living_With"** out of **"Marital_Status"** to extract the living situation of couples.
    - Create a feature **"Children"** to indicate total children in a household that is, kids and teenagers.
    - To get further clarity of household, Creating feature indicating **"Family_Size"**
    - Create a feature **"Is_Parent"** to indicate parenthood status
    - Create three categories in the **"Education"** by simplifying its value counts.
    - Drop redundant features
    - remove outliers
    - change categorical features to integers
    """
    assert os.path.isdir(data_dir), "Invalid arguments for data directory"
    data_path = os.path.join(data_dir, os.listdir(data_dir)[0])
    assert os.path.isfile(data_path) and data_path.endswith(
        ".csv"
    ), "Invalid file within data directory"
    data = pd.read_csv(data_path, sep="\t")
    print(
        f"Extracting original data, data number: {len(data)}, feature number: {len(data.columns)}"
    )
    data = data.dropna()
    # date processing
    data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"], dayfirst=True)
    dates, days = [], []
    for i in data["Dt_Customer"]:
        i = i.date()
        dates.append(i)
    max_date = max(dates)
    for i in dates:
        delta = max_date - i
        days.append(delta)
    data["Customer_For"] = days
    data["Customer_For"] = pd.to_numeric(data["Customer_For"], errors="coerce")
    # Age of customer today
    data["Age"] = 2024 - data["Year_Birth"]

    # Total spendings on various items
    data["Spent"] = (
        data["MntWines"]
        + data["MntFruits"]
        + data["MntMeatProducts"]
        + data["MntFishProducts"]
        + data["MntSweetProducts"]
        + data["MntGoldProds"]
    )

    # Deriving living situation by marital status"Alone"
    data["Living_With"] = data["Marital_Status"].replace(
        {
            "Married": "Partner",
            "Together": "Partner",
            "Absurd": "Alone",
            "Widow": "Alone",
            "YOLO": "Alone",
            "Divorced": "Alone",
            "Single": "Alone",
        }
    )

    # Feature indicating total children living in the household
    data["Children"] = data["Kidhome"] + data["Teenhome"]

    # Feature for total members in the householde
    data["Family_Size"] = (
        data["Living_With"].replace({"Alone": 1, "Partner": 2}) + data["Children"]
    )

    # Feature pertaining parenthood
    data["Is_Parent"] = np.where(data.Children > 0, 1, 0)

    # Segmenting education levels in three groups
    data["Education"] = data["Education"].replace(
        {
            "Basic": "Undergraduate",
            "2n Cycle": "Undergraduate",
            "Graduation": "Graduate",
            "Master": "Postgraduate",
            "PhD": "Postgraduate",
        }
    )

    # For clarity
    data = data.rename(
        columns={
            "MntWines": "Wines",
            "MntFruits": "Fruits",
            "MntMeatProducts": "Meat",
            "MntFishProducts": "Fish",
            "MntSweetProducts": "Sweets",
            "MntGoldProds": "Gold",
        }
    )

    # Dropping some of the redundant features
    to_drop = [
        "Marital_Status",
        "Dt_Customer",
        "Z_CostContact",
        "Z_Revenue",
        "Year_Birth",
        "ID",
    ]
    data = data.drop(to_drop, axis=1)
    # remove outliers
    data = data[(data["Age"] < 90)]
    data = data[(data["Income"] < 600000)]
    
    # change categorical features to integers
    s = (data.dtypes == 'object')
    object_cols = list(s[s].index)
    labelEncoder = LabelEncoder()
    for i in object_cols:
        data[i]=data[[i]].apply(labelEncoder.fit_transform)

    return data

def standard_scale(data: pd.DataFrame, ):
    ds = data.copy()
    cols_del = ['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1','AcceptedCmp2', 'Complain', 'Response']
    ds = ds.drop(cols_del, axis=1)
    scaler = StandardScaler()
    scaler.fit(ds)
    return pd.DataFrame(scaler.transform(ds),columns= ds.columns)