#init sunny
import matplotlib.pyplot as plt


from src.data_import import data



plt.figure(figsize=(10,6))
for lon,df in data.items():
    plt.scatter(df["year"], df["mean"], label=f"lon={lon}")
print('It worked:')
plt.title("Mean per Year for Each Longitude")
plt.xlabel("Year")
plt.ylabel("Mean")
plt.grid(True)
plt.show()