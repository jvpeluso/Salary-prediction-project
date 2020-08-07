![](https://i.imgur.com/dSmz9mh.png)

## 1. Introduction

### Problem description

As we know, the job market is very complex and competitive. For those who want to be hired, and for the companies which want to have the best talent at their service. Knowing in advance, how much are you going to get paid, or will pay for a certain job or profile, could be helpful for both the employee and the employer.

**_Having data of the roles and employee details, ¿Can we develop a model that predicts the salary for a specific job and profile? How accurate can we get? Are the features enough for a good predictive model?_**

We will work to develop a machine learning model, that will try to predict the salaries accurately as possible, with the data at our disposal. We will use the **_Root Mean Squared Error (RMSE)_** as the metric to validate the effectiveness of our model.

### The dataset, features and target value

This dataset contains data on different work positions within a group of 100 companies with their respective salaries (expressed in thousands of dollars). The information is divided into 3 CSV files, the first one with the information of the work positions and employee details (features), and contains 1.000.000 records. The columns are the following:

* **jobId:** The ID of the job.
* **companyId:** The ID of the company.
* **jobType:** The description of the job.
* **degree:** The degree of the employee.
* **major:** The university or college field of specialization of the employee.
* **industry:** The field to which the company belongs.
* **yearsExperience:** The employee years of experience on the job.
* **milesFromMetropolis:** The distance in miles, the employee lives away from his/her place of work.

The second file includes the corresponding salary (target feature) for each of the work positions in the previous file, information of both files will be used as the _Training and validation set_. The file also contains 1.000.000 records and includes the following columns:

* **jobId:** The ID of the job.
* **salary:** Salary amount paid for that job.

The last file contains information about other positions, and have the same structure than the first file. On this data, we will use the algorithm to predict what should be the salary for each job.

## 2. Data quality check

The data files of the job description and corresponding salaries were uploaded into Pandas dataframes, being merged to be able to analyze the data. But first, we've run the consistency checks, finding no NaN or duplicated rows, but 5 rows had a **salary 0** (zero), which we've deleted in the process. Also, we loaded and checked the file with the data on which we will apply the predictive model. 

## 3. Descriptive statistics

### General overview

At first sight, after computing the basic statistics of the dataset, we can make some initial conclusions:

* The numerical features (*yearsExperience* and *milesFromMetropolis*) both have a mean exactly in the middle of their ranges and a high STD value, which indicates that they aren't normally distributed, meaning, the predictive power will be low. 

* The target value *salary*, have a mean close to the 2nd quartile, and smaller, but still high STD. There are some high salaries, but salaries are volatile, considering the max value is 300, we consider those values are not so extreme to be considered outliers. We can see the distribution in the following box plot.

![](https://i.imgur.com/z3YrJF2.png)

* The categorical features have the following unique values:


Feature | Unique values | 
--- | --- 
jobType | CFO, CEO, VICE_PRESIDENT, MANAGER, JUNIOR, JANITOR, CTO, SENIOR
degree | MASTERS, HIGH_SCHOOL, DOCTORAL, BACHELORS, NONE
major | MATH, NONE, PHYSICS, CHEMISTRY, COMPSCI, BIOLOGY, LITERATURE, BUSINESS, ENGINEERING
industry | HEALTH, WEB, AUTO, FINANCE, EDUCATION, OIL, SERVICE


## Feature distribution.

* Numerical features (*yearsExperience* and *milesFromMetropolis*) are evenly distributed within their respective ranges, the salary feature has a normal distribution, slightly left-skewed.

![](https://i.imgur.com/QtV0Lxs.png)

* 2 of the categorical features (*jobType* and *industry*) have a perfect evenly distribution, as each category has the same percentage of records.
* Feature *degree* splits into 2 groups, one with **NONE** and **HIGH_SCHOOL** with 47.4% of the records; 23.7% each, and the remaining 52.6% is evenly distributed among the 3 remaining categories.
* Feature *major* has a predominant category, **NONE**, with 53.2% of the records, the remaining 46.8% is evenly distributed among the remaining categories.

![](https://i.imgur.com/49BaLhF.png)

## Correlations

We can conclude that there is no strong correlation in the numerical features, not with the target value (-0.38 / 0.30) and not between them (0.00).

![](https://i.imgur.com/e56HYlG.png)

## 4. Exploratory data analysis

### Features statistics analysis.

The global mean salary is **_116.06_**; if we calculate the mean per company, we see the range doesn't differ that much, as the max mean is **_116.79_** and the min mean is **_115.34_**, with an inner range of only **_1.45_**. 

If we dig deep and calculate the mean salary per company and job position, the differences are minimal, the highest inner range is for the position of *CEO*(**_4.38_**) and the mean of the ranges is **_3.23_**. These calculations indicate, that the average salary per company won't help with the prediction task.

As expected, the mean salary increases together with the years of experience of the employee, but interestingly, decreases the far the employee lives from the metropolis.

![](https://i.imgur.com/jUpPHUs.png)

When we grouped the salary means by the unique values of the categorical features, we got the following insights:

![](https://i.imgur.com/kaqbwz8.png)

* **jobType:** As expected, the lowest value is for *JANITOR* and the highest for *CEO*. Overall, the values gradually increase according to the level of the job, but interestingly enough, *CTO* and *CFO* have the same value.
* **industry:** Perhaps the feature that shows a more gradual rise between categories. *EDUCATION* pays the least, *FINANCE* and *OIL* the most, and both have essentially the same salary mean.
* **major:** Having a major pays best than not having one; *NONE* has the lowest salary and is widely separated from the rest of the categories, the top major is *ENGINEERING*.
* **degree:** The values are split into 2 groups, *NONE* and *HIGH_SCHOOL* in the lower range, and *BACHELORS*, *MASTERS*, and *DOCTORAL* at the top. The gap is significant between *HIGH_SCHOOL* and *BACHELORS*.


As the position is a key feature, we've analyzed the average salary per *jobType* divided by the other 3 features. Interestingly, the differences remain stable in all cases, that is, they follow the same behaviour when we divide them by *degree*, *industry* or *major*. They all descend in the same way, through the different job types, and only becoming pronounced when we reach *JANITOR*. Here you can see the bar plot of the salary average for *degree* feature, where the trend is easily perceived. Other plots can be seen in the notebook.

![](https://i.imgur.com/pV4yp3B.png)

## 5. Data split and features engineering

We divide the train data into _80%_ train and _20%_ validation, then we apply to all 3 datasets (train, validation and test) the creation, encoding and transformation processes to prepare the data for the predictive models we will develop. The processes are the following:

**_New features_**

To enhance the performance of the models, 5 new features were created:

- **hasMajor:** As the distribution of this feature is 53% of NONE value, and the rest is equally distributed among the remaining values, a dummy feature was created.
- **mean_SalaryGroup :** The mean salary values per _**jobType**_, _**industry**_, _**degree**_, _**major**_ and _**yearsExperience**_ categories. 
- **std_SalaryGroup :** The STD salary values per _**jobType**_, _**industry**_, _**degree**_, _**major**_ and _**yearsExperience**_ categories. 
- **min_SalaryGroup :** The minimum salary values per _**jobType**_, _**industry**_, _**degree**_, _**major**_ and _**yearsExperience**_ categories.
- **max_SalaryGroup :** The maximum salary values per _**jobType**_, _**industry**_, _**degree**_, _**major**_ and _**yearsExperience**_ categories.

Note, that the process calculates the values of the group features on the train data, and then map it into the validation data, to avoid *data leakage*.

**_Categorical features encoding_**

The 4 categorical features were encoded in order based on average salaries per for each of the value in each feature. Note, that here too the process calculates the values of the group features on the train data, and then map it into the validation data, to avoid *data leakage*.

**_Features update_**

To be able to use the *companyID* feature in the machine learning algorithm we update it, filtering only the numerical code of the companies.

## 6. Model development

### Baseline

To begin with, we must define how to measure the effectiveness of the models we are going to develop. Our baseline is the mean of the salary grouped by *jobType* and *companyId*. The mean _RMSE_ error obtained is _**47.6583**_. 

### Model development and validation

For this regression task, 3 different machine learning algorithms were selected, one linear and two ensembles, to see which performs better for the problem:

* **Linear regression.**
* **Random forest regressor.**
* **XGboost regressor.**

After being carefully tuned we've created a *pipeline*, to standardize the values of the features and tested each of them with *cross_val_score* using the **train** datasets, the results were the following:

Model | RMSE | Time
--- | --- | --- 
Linear Regression | 18.8844 | 14 seconds
Random forest regressor | 18.3325 | 57 minutes 44 seconds
XGBoost regressor | 18.2978 | 7 minutes 44 seconds

The lowest *RMSE* error was obtained by the _**XGBoost regressor**_, although the score of the *Random forest regressor* was very close, and not far the *Linear regression* model. Bear in mind too, the time spent in the process by each model, being *Linear Regression* much faster than the other 2, and *Random Forest regressor* the slowest by far. Here you can see the results plotted.

![](https://i.imgur.com/VlD4YAG.png)

### Best model selection and feature importance

Now, to finally test the models, we check the performance with the **validation** datasets. The RMSE go up a little bit but the trend remains, being the best model still the **_XGBoost regressor_**, maintaining also a quite reasonable time of execution. 

Model | RMSE | Time
--- | --- | --- 
Linear Regression | 19.8220 | 2 seconds
Random forest regressor | 19.7116 | 19 minutes 54 seconds
XGBoost regressor | 19.5923 | 2 minutes 44 seconds

We also display the feature importances, which confirms that the features engineered are key in the predictive process and that the dataset itself seems not good enough to do this task, as the results are not that exciting.

![](https://i.imgur.com/Mkz3AE6.png)

Finally, we saved to disk the best model and the predictions for the unseen jobs positions.

## 7. Conclusions

### Conclusion

We've built a decent model, that can predict the salary value for a position with a given group of features, but for us, the performance is as good as the data is. What does this mean? Well, the values of the features aren't normally distributed and some of them doesn't follow a trend that can help the algorithms to predict. And above all, gathering some other position features, work specifications, or details of the employee will help to deliver more accurate results.

On the other hand, we can confirm the *feature engineering* process is crucial, the model bases almost entirely on the feature created. Also is worth to keep in mind to be careful to avoid the data leakage when building features based on the target feature.

Finally, even if the lower *RMSE* was obtained by the **_XGBoost regressor_** model, we must also take into account the execution time, so although the highest error was obtained by the *Linear regression* model, the training process is really fast, so depending on the planned use (an API as an example) the difference in performance can be compensated by the speed of execution.
