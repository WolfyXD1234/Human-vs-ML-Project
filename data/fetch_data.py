# # import mlcroissant as mlc
# # import pandas as pd

# # # Fetch the Croissant JSON-LD
# # croissant_dataset = mlc.Dataset('https://www.kaggle.com/datasets/hojjatk/mnist-dataset/croissant/download')

# # # Check what record sets are in the dataset
# # record_sets = croissant_dataset.metadata.record_sets
# # print("---")
# # print(record_sets)
# # print("---")

# # # Fetch the records and put them in a DataFrame
# # record_set_df = pd.DataFrame(croissant_dataset.records(record_set=record_sets[0].uuid))
# # record_set_df.head()

# #hfkjahewfiuahriufhsarifuhadirughsiughsdiughiusthgiriusrtgb
# #it downloads to somewhrere else

# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("hojjatk/mnist-dataset")

# print("Path to dataset files:", path)

# Run --> pip install ucimlrepo
# Run --> pip install seaborn
from ucimlrepo import fetch_ucirepo
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

MNIST = fetch_ucirepo(id=80)


print(MNIST)
print("---")
print(type(MNIST))
# with open("data/MNIST.txt", "w") as f:
#   f.write(MNIST)

# df = pd.DataFrame(MNIST)

# df.to_csv("MNIST.csv")