import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
Wine = pd.read_csv("/workspaces/Human-vs-ML-Project/data/Wine.csv")
