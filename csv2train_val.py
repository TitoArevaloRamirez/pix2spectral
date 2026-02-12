import os
import ast
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


if __name__=="__main__":
    csv_path="/Volumes/data/EstradaDataset/Dataset_with_images.csv"

    df = pd.read_csv(csv_path)

    df["Species"] = df["Species"].astype(str).str.strip().str.lower()
    df["Stages"] = df["Stages"].astype(str).str.strip().str.lower()

    df_avocado = df[df["Species"] == "avocado"]
    df_olive = df[df["Species"] == "olive"]
    df_grape = df[df["Species"] == "vineyard"]

    train_avocado, val_avocado = train_test_split(df_avocado, test_size=0.3, random_state=42)
    train_olive, val_olive = train_test_split(df_olive, test_size=0.3, random_state=42)
    train_grape, val_grape = train_test_split(df_grape, test_size=0.3, random_state=42)


    train_avocado.to_csv('/Volumes/data/EstradaDataset/train_avocado.csv')
    val_avocado.to_csv('/Volumes/data/EstradaDataset/val_avocado.csv')

    train_olive.to_csv('/Volumes/data/EstradaDataset/train_olive.csv')
    val_olive.to_csv('/Volumes/data/EstradaDataset/val_olive.csv')

    train_grape.to_csv('/Volumes/data/EstradaDataset/train_vineyard.csv')
    val_grape.to_csv('/Volumes/data/EstradaDataset/val_vineyard.csv')


