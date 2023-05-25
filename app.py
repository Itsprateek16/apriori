import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
st.set_option('deprecation.showfileUploaderEncoding', False)
def find_associatio_rule(support):
  # Load the pickled model
  model = pickle.load(open('aprioriexample.pkl','rb'))
    if uploaded_file is not None:
    dataset= pd.read_csv(uploaded_file)
else:
    dataset= pd.read_csv('retail_dataset.csv')

  #Create list 
    transactions = []
    for i in range(0, 315):
        transactions.append([str(dataset.values[i,j]) for j in range(0, 7)])

    from mlxtend.preprocessing import TransactionEncoder
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    df=df[['Bagel','Bread','Cheese','Diaper','Eggs','Meat','Milk','Pencil','Wine']]
    freq_items = apriori(df, min_support=support, use_colnames=True)
    rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)
return rules
def find_frequent_items(support):
  # Load the pickled model
  model = pickle.load(open('aprioriexample.pkl','rb'))     
     if uploaded_file is not None:
    dataset= pd.read_csv(uploaded_file)
    else:
    dataset= pd.read_csv('retail_dataset.csv')
    #Create list 
    transactions = []
    for i in range(0, 315):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 7)])

     from mlxtend.preprocessing import TransactionEncoder
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)
  df=df[['Bagel','Bread','Cheese','Diaper','Eggs','Meat','Milk','Pencil','Wine']]
     freq_items = apriori(df, min_support=support, use_colnames=True)
    rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)
  
  return freq_items
