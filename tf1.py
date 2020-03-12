from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf

dftrain = pd.read_csv('data/train.csv') # training data
dfeval = pd.read_csv('data/eval.csv') # testing data
print(dftrain.head)
print(dfeval.head)
y_train=dftrain.pop('survived')
y_eval=dfeval.pop('survived')
fig, ax = plt.subplots() 
dftrain.age.hist(bins=20)
fig, ax = plt.subplots() 
dftrain.sex.value_counts().plot(kind='barh')
fig, ax = plt.subplots() 
dftrain['class'].value_counts().plot(kind='barh')
fig, ax = plt.subplots() 
pd.concat([dftrain,y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')
#uncomment to display charts:
#plt.show() 

categorical_columns=['sex','n_siblings_spouses','parch','class','deck','embark_town','alone']
numeric_columns=['age','fare']
feature_columns=[]
for feature_name in categorical_columns:
	vocabulary=dftrain[feature_name].unique()
	feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in numeric_columns:
	feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))
print(feature_columns)	

def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
	def input_function():
		ds=tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
		if shuffle:
			ds.shuffle(1000)
		ds=ds.batch(batch_size).repeat(num_epochs)
		return ds
	return input_function
train_input_fn=make_input_fn(dftrain, y_train)
eval_input_fn=make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)