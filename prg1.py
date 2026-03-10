import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

data=fetch_california_housing(as_frame=True)
df=data.frame
print(df.head())

def plot_histogram(df):
  df.hist(bins=30,figsize=(12,10),color='skyblue',edgecolor='black')
  plt.suptitle('Histogram of California Housing Dataset',fontsize=16)
  plt.tight_layout(rect=[0,0,1,0.97])
  plt.show()

def plot_boxplot(df):
  plt.figure(figsize=(14,10))
  for i,column in enumerate(df.columns,1):
    plt.subplot(3,3,i)
    sns.boxplot(y=df[column],color='skyblue')
    plt.title(f'Boxplot of {column}',fontsize=14)
    plt.tight_layout()
    plt.show()

def analyze_features(df):
  print("Feature Analysis:")
  for column in df.columns:
    print(f"\nfeatures:{column}")
    print(f"Mean:{df[column].mean():.2f},Median:{df[column].mean():.2f}")
    q1=df[column].quantile(0.25)
    q3=df[column].quantile(0.75)
    iqr=q3-q1
    lower_bound=q1-1.5*iqr
    upper_bound=q3+1.5*iqr
    outliers=df[(df[column]<lower_bound)|(df[column]>upper_bound)]
    print(f"Outliers:{len(outliers)}")

plot_histogram(df)
plot_boxplot(df)
analyze_features(df)
