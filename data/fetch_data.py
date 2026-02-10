from ucimlrepo import fetch_ucirepo 
import pandas as pd
  
# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 
  
# data (as pandas dataframes) 
X = wine_quality.data.features 
y = wine_quality.data.targets 
z = wine_quality.data.other

X.to_csv("data/Wine Features")
y.to_csv("data/Wine Target")
z.to_csv("data/Wine Other")