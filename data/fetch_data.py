from ucimlrepo import fetch_ucirepo 
import pandas as pd
  
# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 
  
# data (as pandas dataframes) 
X = wine_quality.data.features 
y = wine_quality.data.targets 
z = wine_quality.data.original

z.to_csv("data/Wine.csv")

targets = pd.concat([y, z["color"]], axis=1)

X.to_csv("data/Wine_Feature.csv")
targets.to_csv("data/Wine_Targets.csv")

