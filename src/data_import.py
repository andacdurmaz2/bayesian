import pandas as pd

#import matplotlib.pyplot as plt
import os

df = pd.read_csv("data/df_equator.csv")
#df=pd.read_csv('/Users/pauljegen/Uni/TUM/Semester_3/Bayesian_statistics/Group_project/data/df_equator.csv')
df = df.sort_values(by=["lon", "year"])
print(os.getcwd())
df = df[(df["lon"] >= 45) & (df["lon"] <= 50)]
df.head()
data=dict()
for lon_value in df["lon"].unique():
    subset = df[df["lon"] == lon_value]
    # store a list of records, each record contains the year and mean for this longitude
    data[lon_value] = subset[["year", "mean"]]
    
