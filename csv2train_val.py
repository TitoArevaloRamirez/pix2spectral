import os
import ast
import numpy as np
import pandas as pd



if __name__="__main__":
    csv_path="/Volumes/data/EstradaDataset/Dataset_with_images.csv",

    df = pd.read_csv(csv_path)

    df["Species"] = df["Species"].astype(str).str.strip().str.lower()
    df["Stages"] = df["Stages"].astype(str).str.strip().str.lower()

    species = str(species).strip().lower()
    df_avocado = df[df["Species"] == "avocado"]
    df_olive = df[df["Species"] == "olive"]
    df_grape = df[df["Species"] == "grape"]

