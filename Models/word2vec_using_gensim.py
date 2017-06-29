import pandas as pd
import numpy as np
import gensim #using pip3 to install
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

data_path="/home/wml/kaggle_instacart/data/instacart_2017_05_01/"

train_orders = pd.read_csv(data_path+"order_products__train.csv")
prior_orders = pd.read_csv(data_path+"order_products__prior.csv")
products = pd.read_csv(data_path+"products.csv").set_index('product_id')

train_orders["product_id"] = train_orders["product_id"].astype(str)
prior_orders["product_id"] = prior_orders["product_id"].astype(str)

train_products = train_orders.groupby("order_id").apply(lambda order: order['product_id'].tolist())
prior_products = prior_orders.groupby("order_id").apply(lambda order: order['product_id'].tolist())

sentences = prior_products.append(train_products).values

model = gensim.models.Word2Vec(sentences, size=100, window=1, min_count=5, workers=4)

model.save("product2vec.model")