import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from human_model.human_classifier import human_classify
df = pd.read_csv("data/Wine.csv")

columns = df.columns.tolist()

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

os.makedirs("human_model/plots", exist_ok=True)

for datapoint in columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x=datapoint,
        y='color',
        hue='correct',
        style='correct',
        s=100,
        palette={True: 'green', False: 'red'}
    )

    plt.title('Human Algorithm: Correct vs Incorrect Predictions')
    plt.xlabel(datapoint)
    plt.ylabel('color')
    plt.legend(title='Prediction Correct')
    plt.grid(True)
    plt.savefig(f'human_model/plots/human_model_training_results_{datapoint}.png', dpi=150)
    plt.close()

print("done")