import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

data=fetch_california_housing(as_frame=True)
df=data.frame
print("Dataset Sample:")
print(df.head())

correlation_matrix=df.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

def plot_heatmap(correlation_matrix):
    plt.figure(figsize=(10,8))
    sns.heatmap(correlation_matrix,annot=True,fmt=".2f",
               cmap="coolwarm",cbar=True,square=True,linewidth=0.5)
    plt.title("Correlation Matrix Heatmap",fontsize=16)
    plt.show()
plot_heatmap(correlation_matrix)

def plot_pairplot(df):
    sns.pairplot(df,diag_kind="kde",corner=True,plot_kws={'alpha':0.5},diag_kws={"fill":True})
    plt.suptitle("Pair Plot of Numericsal Features",y=1.02,fontsize=16)
    plt.show()
plot_pairplot(df)
