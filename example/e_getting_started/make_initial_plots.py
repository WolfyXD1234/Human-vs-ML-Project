import os
from example.e_data.fetch_data import load_iris_data
import matplotlib.pyplot as plt
import seaborn as sns

# 'sepal length' is a possible factor
def make_plot(factor_1, factor_2):
    factor_1_label = factor_1.replace('_', ' ')
    factor_2_label = factor_2.replace('_', ' ')
    
    df, target_name = load_iris_data()

    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x=factor_1,
        y=factor_2,
        hue=target_name,
        style=target_name,
        s=90
    )

    plt.title(f'Iris Species: {factor_1_label} vs {factor_2_label}')
    plt.xlabel(f'{factor_1_label} (cm)')
    plt.ylabel(f'{factor_2_label} (cm)')
    plt.legend(title='Iris Species')
    plt.grid(True)
    plt.savefig(f'example/e_getting_started/plots/{factor_1_label}_v_{factor_2_label}.png', dpi=150)
    plt.close()

make_plot('sepal length', 'petal length')
make_plot('sepal width', 'petal length')
make_plot('sepal width', 'petal width')