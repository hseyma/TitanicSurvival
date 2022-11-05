from sklearn.cluster import KMeans
import os
import sys
import io
import json
import psycopg2 as psy
import datetime as dt
import numpy as np
import pandas as pd
import TitanicFunctions as tf
import seaborn as sns


pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 150)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

path = 'C:/Users/hseym/PycharmProjects/TitanicSurvival/'

os.chdir(path)


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train = tf.reduce_mem_usage(train)
test = tf.reduce_mem_usage(test)

sns.pairplot(train, hue = 'Survived', palette = 'Dark2')

train['Title'] = train['Name'].str.extract('([A-Za-z]+)\.')
test['Title'] = test['Name'].str.extract('([A-Za-z]+)\.')

train['Title'] = train['Title'].replace({'Mme': 'Mrs', 'Mlle': 'Miss', 'Ms': 'Miss', 'Dr': 'Other', 'Rev': 'Other',
                                         'Major': 'Other', 'Col': 'Other', 'Countess': 'Other', 'Capt': 'Other',
                                         'Sir': 'Other', 'Lady': 'Other', 'Don': 'Other', 'Jonkheer': 'Other'})
test['Title'] = test['Title'].replace({'Mme': 'Mrs', 'Mlle': 'Miss', 'Ms': 'Miss', 'Dr': 'Other', 'Rev': 'Other',
                                       'Major': 'Other', 'Col': 'Other', 'Countess': 'Other', 'Capt': 'Other',
                                       'Sir': 'Other', 'Lady': 'Other', 'Don': 'Other', 'Jonkheer': 'Other',
                                       'Dona': 'Other'})
