Docker commands :  
- docker build -t sydres_a1 .
- docker run sydres_a1  
  
Output : 
```
=======================
model_regression_1
Mean Squared Error: 0.1469846169099422
R-squared: 0.25399806037690364
=======================

=======================
model_regression_2
Mean Squared Error: 0.14910993202209347
R-squared: 0.24321129078626968
=======================

=======================
Accuracy: 0.9784272790535838
Precision: 0.9780334728033473
Recall: 0.9894179894179894
F1 Score: 0.9836927932667017
ROC AUC Score: 0.9733675312943605
Confusion Matrix:
 [[935  10]
 [ 21 471]]
=======================
```