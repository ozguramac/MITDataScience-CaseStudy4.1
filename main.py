import os
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

import pandas as pd
import numpy as np
import graphlab

#Prep Data
col_names = ["user_id", "item_id", "rating", "timestamp"]
data = pd.read_table("u.data", names=col_names)
data = data.drop("timestamp", 1)
data.info()

#Plot
# import matplotlib as mpl
# mpl.use('TkAgg')
# from matplotlib import pyplot as plt
# plt.hist(data["rating"])
# plt.show()

#Calc sparsity
Number_Ratings = len(data)
Number_Movies = len(np.unique(data["item_id"]))
Number_Users = len(np.unique(data["user_id"]))
Sparsity = (Number_Ratings * 100) / (Number_Movies * Number_Users)
print "Sparsity={sparsity}%".format(sparsity=Sparsity)
#Subset sparsity
# subset_data = data[len(data.user_id) > 50]
# Number_Subset_Ratings = len(subset_data)
# Number_Subset_Movies = len(np.unique(subset_data["item_id"]))
# Number_Subset_Users = len(np.unique(subset_data["user_id"]))
# Reduced_Sparsity = (Number_Subset_Ratings * 100) / (Number_Subset_Movies * Number_Subset_Users)
# print "Reduced Sparsity={sparsity}%".format(sparsity=Reduced_Sparsity)

#Set product key for GraphLab Create
graphlab.product_key.set_product_key(os.getenv("TURI_GRAPH_LAB_CREATE_PROD_KEY"))
print "Using Turi GraphLab Create API with product key {tglcpk}".format(tglcpk=graphlab.product_key.get_product_key())

#Split data
sf = graphlab.SFrame(data)
sf_train, sf_test = sf.random_split(.9, seed=5)
print(len(sf_train), len(sf_test))

#Popularity Recommender
pr_model = graphlab.popularity_recommender.create(sf_train, target="rating")
# prec = pr_model.recommend()
# prec.print_rows(18)

#Split data
# sf_train2, sf_valid2 = sf_train.random_split(.75, seed=5)
# print(len(sf_train2), len(sf_valid2), len(sf_test))
#Factorization Recommender
# fr_model1 = graphlab.factorization_recommender.create(sf_train2, target="rating", regularization=0.00005)
# fr_model2 = graphlab.factorization_recommender.create(sf_train2, target="rating", regularization=0.008)
# fr_model3 = graphlab.factorization_recommender.create(sf_train2, target="rating", regularization=0.04)
# frec = graphlab.recommender.util.compare_models(sf_valid2, [fr_model1,fr_model2,fr_model3])

#Item-Item Similarity Recommender
itit_model = graphlab.item_similarity_recommender.create(sf_train, target="rating")
# itit_rec = itit_model.recommend(k=5)
# itit_rec.print_rows(10)

#Precision/Recall
graphlab.recommender.util.compare_models(sf_test, [pr_model, itit_model], metric="precision_recall", target="rating")
