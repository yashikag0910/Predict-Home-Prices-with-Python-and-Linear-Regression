import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#load data from csv
data =pd.read_csv('home_dataset.csv')

#Extract features and target variables
h_sizes=data['HouseSize'].values
h_prices=data['HousePrice'].values

#visualize the data
plt.scatter(h_sizes,h_prices,marker='o',color='blue')
plt.title('House Prices vs House sizes')
plt.xlabel('House Size(sq.ft)')
plt.ylabel('House Prices($)')
plt.show()

#split the data into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(h_sizes,h_prices,test_size=0.2,random_state=42)

#reshape the data for numpy
x_train=x_train.reshape(-1,1)
x_test=x_test.reshape(-1,1)
#create and train the model
model= LinearRegression()
model.fit(x_train,y_train)

#predict prices for the test set
predictions=model.predict(x_test)

#visualize the predictions
plt.scatter(x_test,y_test,marker='o',color='blue',label='Actual prices')
plt.plot(x_test,predictions,color='red',label='Predictions')
plt.title('Price Prediction with linear Regression')
plt.xlabel('House Size(sq.ft)')
plt.ylabel('House prices($)')
plt.legend()
plt.show()

