# Linear Models for Python
Linear regression models for Python

## OLS Linear regression

### Overview

This class allows you to OLS linear regression estimation with the ability to control for fixed effects according to some choice variables. Moreover, you can use robust methods (heteroscedasticity and cluster robust std. errors) to estimate the variance-covariance matrix. 

###  Inputs
The OLS linear regression model can be called by using the class *ols* in the main file. To call this class, you need the following inputs:

- *dataset* is the dataset where you store the data you want to pass into the ols model.
- *regressors* is the list of strings of all explanatory variables you want to use to explain y.
**It is very important that you pass a list of strings even if your explanatory variable is just one variable.**
- *dependent* is the list of the string of the column that stores your dependent variable in your dataframe.
**It is very important that you pass a list of strings even if your explanatory variable is just one variable.**
- *cons* is True by default, but if you want to regress without intercept, just declare *cons = False* .
- *fixed_effects* by default is an empty list. However, you can pass a list with the string of all variables you want to use for fixed effects. For example, if you have a panel dataset of countries years, you can pass a list of string of the variable that stores countries and a string for the variable that stores years.
- *method* is "standard" by default, but if you want to a heteroscedasticity robust variance covariance matrix, just declare *method = 'heter'*  Moreover, if you want to estimate the variance covariance matrix with clustered method, you have to pass *method = 'cluster'*.
- *cluster* by default is an empty list. However, if you decided to use a clusterd estimation, you have to pass a list containeingn the string for the variable you want to use as clusters.

### Features
- To summarize the results, just call "your object name" . *summary()*. It will print just the table with betas, std, t, p value, and confidence
interval.

- Use *coefficients()* if you want to obtain beta coefficients.
- Use *standard_dev()* if you want to obtain standard deviation for betas.
- Use *fitted()* if you want fitted values.
- Use *residuals()* if you want residuals.
- Use *summary()* if you want a table form summary of the estimation.

### Example 

To initialize the model (suppose the module is imported *impot linear_models as lm*:)

```
model  = lm.ols(dataset = df, regressors=['exper','expersq','kidslt6','kidsge6'],dependent = ['lwage'])
```

To call the informative summary:
```
model.summary()
```

And here is the output:
```
Linear regression
---------------------------------------------------------------
lwage      coefficient           se          t    p_value       low 95       high 95
-------  -------------  -----------  ---------  ---------  -----------  ------------
exper       0.0456439   0.0141809     3.21869        0      0.0178494    0.0734385
expersq    -0.00101304  0.000417998  -2.42356        0.02  -0.00183232  -0.000193768
kidslt6     0.0314494   0.0892375     0.352424       0.72  -0.143456     0.206355
kidsge6    -0.0345768   0.0283234    -1.22079        0.22  -0.0900906    0.020937
cons        0.875164    0.120301      7.27478        0      0.639374     1.11095
---------------------------------------------------------------
```



## 2SLS IV Regression

### Overview

This class allows you 2SLS IV regression estimation on Python. It provides also a final summary report where you can check second-stage results. Estimations are obtained using the OLS class for the first stage.


###  Inputs
The 2SLS IV Regression model can be called by using the class *two_sls* in the main file. To call this class, you need the following inputs:

- *dataset* is the dataset where you store the data data you want to use for the IV regression.
- *dependent* is name of the column of the dependent variable in your pandas dataset. 
**It is very important that you pass a list of strings even if there is just one variable for the category.**
- *regressors* is the list of columns' name of exogenous regressors that you want to use for the regression.
**It is very important that you pass a list of strings even if there is just one variable for the category.**
- *endogenous* is the column name of the endogenous variables in the dataset.
**It is very important that you pass a list of strings even if there is just one variable for the category.**
- *instruments* is the list of columns' name of instruments that you want to use in addition of exogenous regressors.
- *cons* is True by default, but if you want to regress without intercept, just declare *cons = False* .
- *fixed_eff* by default is an empty list. However, you can pass a list with the string of all variables you want to use for fixed effects. For example, if you have a panel dataset of countries years, you can pass a list of string of the variable that stores countries and a string for the variable that stores years.


###  Features
- To summarize the results, just call "your object name" . *summary()*. 

- Use *first_stage()* to return a dictionary of usefull elements for the first stage regression like the fitted values (*'fitted'*), and beta coefficients (*'betas'*)
- Use *second_stage()* to return a dictionary of usefull elements for the second stage regression. You can obtain beta coefficients (*'beta'*) and variance covariance matrix (*'var_matrix'*). Other elements can be obtain. Check the code at the second stage function to see dictionary's keys.
- Use *summary()* if you want a table form summary of the estimation.


### Example
To import the model 
```
import linear_models as lm
```

Initialize the model 

```
model = lm.two_sls(dataset=df, dependent = 'wage', regressors= ['exper'],
                endogenous= 'educ', instruments=['sibs'])
```

Print the results 

```
print(model.summary())
```

And here is the output:
```

2SLS Regression
-------------------------------------------------------------------------------
wage      coefficient         se      t    p_value      low 95    high 95
------  -------------  ---------  -----  ---------  ----------  ---------
exper         32.1567    7.06488   4.55      0         18.3095    46.0038
educ         139.684    28.0369    4.98      0         84.7315   194.636
cons       -1295.23    453.262    -2.86      0.004  -2183.62    -406.834
-------------------------------------------------------------------------------
----------------------------------------------------------------------------
Instrumented: ['educ']
Instruments: ['exper','sibs']
----------------------------------------------------------------------------

```

##If you want to see a complete example on a Jupyter Notebook , please see the Showcase file in the repository.

## If you want to use it, a citation is more than welcome


## References
- Stock J, Yogo M. Testing for Weak Instruments in Linear IV Regression. In: Andrews DWK Identification and Inference for Econometric Models. New York: Cambridge University Press ; 2005. pp. 80-108.
- Dataset from Wooldridge data sets: http://fmwww.bc.edu/ec-p/data/wooldridge/datasets.list.html
- Wooldridge, Jeffrey M. Econometric analysis of cross section and panel data. MIT press, 2010.

