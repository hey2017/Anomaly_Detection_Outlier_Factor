# Anomaly_Detection_Outlier_Factor
In this project I create an anomaly detection. Here the machine learning model is [LocalOutlierFactor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html). The main idea is to fit the x and y data to a linear regression model
to find the mean absolute error (MAE) before and after detecting the outliers. When the outliers are detected properly the MAE value decreases. In this project the code is written 
such a way that you can see the variation of MAE values vs. different contamination values of Isolation Forest model and also it recommends the best contamination value. 

# Data Set information
The data set has unkown labels but all are numerical values. It has 13 varaibales which each discusses properties of houses which eventually needs prediction on the value of houses. The data is ```Housing_Data.csv```

## Technical requirements
For this analysis you need to have python installed and have the following libraries:
- [Pandas](https://pandas.pydata.org/)
- [sklearn](https://scikit-learn.org/stable/)
- [matplotlib](https://matplotlib.org/)
- [LocalOutlierFactor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html)
- [Mean Absolute Error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html)
- [Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)

## Run the code
To run the code, you can use the file ```LOF.py```.
