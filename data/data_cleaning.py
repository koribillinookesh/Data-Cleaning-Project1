import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

df = pd.read_csv("data/raw_data.csv")

num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(include="object").columns

df[num_cols] = df[num_cols].fillna(df[num_cols].median())

for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

df.drop_duplicates(inplace=True)

def remove_outliers(data, columns):
    for col in columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        data = data[(data[col] >= lower) & (data[col] <= upper)]
    return data

df = remove_outliers(df, num_cols)

os.makedirs("data", exist_ok=True)
os.makedirs("visuals", exist_ok=True)

df.to_csv("data/cleaned_data.csv", index=False)

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.savefig("visuals/correlation_heatmap.png")
plt.close()

for col in num_cols:
    sns.histplot(df[col], kde=True)
    plt.savefig(f"visuals/{col}_distribution.png")
    plt.close()

for col in num_cols:
    sns.boxplot(x=df[col])
    plt.savefig(f"visuals/{col}_boxplot.png")
    plt.close()

for col in cat_cols:
    sns.countplot(x=df[col])
    plt.xticks(rotation=45)
    plt.savefig(f"visuals/{col}_countplot.png")
    plt.close()

corr = df.corr(numeric_only=True)

for col in corr.columns:
    for idx in corr.index:
        if col != idx and abs(corr.loc[col, idx]) > 0.7:
            print(col, idx, corr.loc[col, idx])
