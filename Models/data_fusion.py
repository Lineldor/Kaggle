# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 14:44:18 2017

@author: dell
"""

import pandas as pd #data frames
import numpy as np  #algebra & calculus
import nltk  #text processing & manipulation
import matplotlib.pyplot as plt #plotting
#import seaborn as sns #plotting
from functools import partial # to reduce df memory consumption by applying to_numeric
#color=sns.color_palette() # adjusting plotting style
import warnings
warnings.filterwarnings('ignore') # silence annoying warnings
data_path="C:\Users\lenovo\Desktop\DL\KAGGLE\Data\instacart_2017_05_01\"
#aisles
aisles=pd.read_csv(data_path+"aisles.csv",engine='c')
print("Total aisles: {}".format(aisles.shape[0]))
aisles.head()

#departments
departments=pd.read_csv(data_path+"departments.csv",engine='c')
print("Total departments: {}".format(departments.shape[0]))
departments.head()

#products
products = pd.read_csv(data_path+"products.csv", engine='c')
print('Total products: {}'.format(products.shape[0]))
products.head(5)

# combine aisles, departments and products (left joined to products)
goods=pd.merge(left=pd.merge(left=products, right=departments, how='left'),right=aisles,how='left')
goods.product_name=goods.product_name.str.replace(' ','_').str.lower()
#print(goods.info())

# load datasets

# train dataset
op_train = pd.read_csv(data_path+"order_products__train.csv", engine='c', 
                       dtype={'order_id': np.int32, 'product_id': np.int32, 
                              'add_to_cart_order': np.int16, 'reordered': np.int8})
print('Total ordered products(train): {}'.format(op_train.shape[0]))
op_train.head(10)

# test dataset (submission)
test = pd.read_csv(data_path+"sample_submission.csv", engine='c')
print('Total orders(test): {}'.format(test.shape[0]))
test.head()

#prior dataset
op_prior = pd.read_csv(data_path+"order_products__prior.csv", engine='c', 
                       dtype={'order_id': np.int32, 
                              'product_id': np.int32, 
                              'add_to_cart_order': np.int16, 
                              'reordered': np.int8})

print('Total ordered products(prior): {}'.format(op_prior.shape[0]))
op_prior.head()

# orders
orders = pd.read_csv(data_path+'orders.csv', engine='c', dtype={'order_id': np.int32, 
                                                           'user_id': np.int32, 
                                                           'order_number': np.int32, 
                                                           'order_dow': np.int8, 
                                                           'order_hour_of_day': np.int8, 
                                                           'days_since_prior_order': np.float16})
print('Total orders: {}'.format(orders.shape[0]))
orders.head()

# merge train and prior together iteratively, to fit into 8GB kernel RAM
# split df indexes into parts
indexes = np.linspace(0, len(op_prior), num=10, dtype=np.int32)

# initialize it with train dataset
order_details = pd.merge(
                left=op_train,
                 right=orders, 
                 how='left', 
                 on='order_id'
        ).apply(partial(pd.to_numeric, errors='ignore')) #downcast cann't meet the data's requirement

# add order hierarchy
order_details = pd.merge(
                left=order_details,
                right=goods[['product_id', 
                             'aisle_id', 
                             'department_id']].apply(partial(pd.to_numeric, 
                                                             errors='ignore' 
                                                        )),
                how='left',
                on='product_id'
)

print(order_details.shape, op_train.shape)

# delete (redundant now) dataframes
del op_train

#print order_details.head()

#%%time
# update by small portions
for i in range(len(indexes)-1):
    order_details = pd.concat(
        [   
            order_details,
            pd.merge(left=pd.merge(
                            left=op_prior.loc[indexes[i]:indexes[i+1]-1, :],
                            right=goods[['product_id', 
                                         'aisle_id', 
                                         'department_id' ]].apply(partial(pd.to_numeric, 
                                                                          errors='ignore', 
                                                                          )),
                            how='left',
                            on='product_id'
                            ),
                     right=orders, 
                     how='left', 
                     on='order_id'
                ) #.apply(partial(pd.to_numeric, errors='ignore', downcast='integer'))
        ]
    )
        
print('Datafame length: {}'.format(order_details.shape[0]))
print('Memory consumption: {:.2f} Mb'.format(sum(order_details.memory_usage(index=True, 
                                                                         deep=True) / 2**20)))
# check dtypes to see if we use memory effectively
print(order_details.dtypes)

# make sure we didn't forget to retain test dataset :D
test_orders = orders[orders.eval_set == 'test']

# delete (redundant now) dataframes
del op_prior, orders








