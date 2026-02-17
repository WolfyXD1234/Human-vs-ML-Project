import pandas as pd
from human_classifier import human_classify
df = pd.read_csv("data/Wine.csv")

df['human_prediction'] = df.apply(human_classify, axis = 1)
df['correct'] = df['human_prediction'] == df["color"]
accuracy = (df['human_prediction'] == df["color"]).mean()
print(f"Human classifier accuracy: {accuracy:.2%}")

conf_matrix = pd.crosstab(
    df["color"],
    df['human_prediction'],
    rownames=['Actual'],
    colnames=['Predicted']
)
print(conf_matrix)