# Loan Approval Prediction in Python


## 1. PROBLEM AND OBJECTIVE

Whenever an individual/corporation applies for a loan from a bank, their credit history undergoes a rigorous credit-worthiness check to ensure that they are capable enough to pay off the loan.

Banks use 'Data Models' to decide.
Inputs to the models are like 'current financial standing', 'previous credit history' and 'some other variables'.
Output of the model is a metric which gives a measure of the risk that the issuer will potentially take on issuing the loan. 

The measure is generally in the form of a probability and is the risk that the person will default on their loan (called the probability of default) in the future.
Based on the amount of risk that the issuer is willing to take (plus some other factors) they decide on a cutoff of that score and use it to take a decision regarding whether to pass the loan or not. This is a way of managing credit risk. The whole process collectively is referred to as 'UNDERWRITING'.

The OBJECTIVE is to build a data model using past data to predict the probability of default, and choose cut-off based on what is suitable, so as to be able to decide who can be given loan in the future.
Alternatively one can also use a modeling technique which gives binary 
output. Based on the data that is available during loan application, build a model to predict default in the future. This will help the company in deciding whether or not to pass the loan.


## 2. BUSINESS SENSE & DATA DESCRIPTION

GIVEN DATA consist of:
- Dataset containing both train and test data
- Data dictionary

The text files contain complete loan data for all loans issued by the company through 2007-2015. The data contains the indicator of default, payment information, credit history, etc. (855969 records and 73 features).

DATA SPLIT INTO "TRAIN-VAL-TEST" SET
The original data file was split into train (June 2007 - May 2015) and out-of time test (June 2015 - Dec 2015) data, and further the ‘train’ data was split into ‘train’ and ‘validation’ datasets.

Moreover the Train data was divided into Train and Validation set on 80:20 ratio for better model building, and the out-of-time test data was used for final checking of selected best models.

STANDARDIZATION OF X VARIABLES
All Train and Validation X variables were scaled using 
‘sklearn:StandardScaler’. 

DOWN SAMPLING THE TRAIN DATA USING ‘RandomUnderSampler’
Under-sample the majority class by randomly picking samples without 
replacement.

BALANCING THE TRAIN DATA USING SMOTE
Synthetic Minority Over-sampling Technique generates new samples by 
interpolation within the minority sample.

The training data was used to build models/analytical solution with prediction on validation data and finally apply the model to out-of-time test data to measure the performance and robustness of the selected best models.

CREATE OUT-SAMPLE DATA FOR SUBMISSION FILE
A submission file with ‘id’, ‘loan_amnt’, ‘default_ind’ was maintained through the data processing, so that in the final step the predicted response could be added and the file submitted.


## 3. EDA & VISUALIZATION

Some of the questions we are trying to explore through EDA are:
What are the distributions of loan data?
What are the correlation between the variables?
What are the characteristics of good and bad borrowers?
Which factors influence the loan repayment and default rate?


## 4. DATA CLEANING & PREPROCESSING

Some of the processes involved were:
- Drop columns with more than 10% missing values
- Remove columns with only one value
- Remove columns with insignificant frequency for others
- Manually select unimportant features to be dropped
- Removing columns with more than 1% of the rows containing nulls.
- Removing rows with less than 1% of nulls in columns.


## 5. FEATURE ENGINEERING

CREATE THREE NEW FEATURES : Monthly Income, EMI AND Total Balance.

Monthly Income can be computed from annual income, while EMI from loan 
amount and term. Total balance is Monthly Income minus EMI/debt per 
month. These features seems to be more relevant to the borrowers ability to 
payback. 

DROP CORRELATED COLUMNS 'installment' AND 'annual_inc'

REMOVING THE OUTLIER VALUES IN 'DTI' COLUMN

CONVERT CATEGORICAL VARIABLE 'ISSUE_D' TO INTEGER.
It is the variable on which the data is split into IN-sample and OUT-sample.

MAP ORDINAL VALUES TO INTEGERS
Grade with levels from A to G can be converted to 1 to 7.

ENCODE NOMINAL VALUES AS DUMMY VARIABLES
The features:: home_ownership, verification_status, purpose, 
initial_list_status can be one-hot-encoded for ML modeling.

FEATURE SELECTION
So some of the questions we want to know here are:_
- Are all the 43 features equally important?
- Which features are more important?
- Will the features fetch a better model eval score?

FS USING PKG RFE (Recursive Feature Elimination)
The best 26 features was asked for, and RFE produced the following result…

FS USING UNIVARIATE FS
Standardize IV Features With MINMAXSCALER
This was done in the preprocessing step while spliting cleaned dataset into X 
and Y. The k was specified as 16, and best 16 features with p-values were got.

FS USING VARIANCETHRESHOLD
This gives a list with decreasing variance. By default it will remove all features
with zero variance. We choose to keep the best 40 variables as they exhibited
good variance

FS USING DECISIONTREE MODEL
Here the attribute feature_importances_ of DecisionTree is used to pull up the
important feature list, from the model.

FS USING ADABOOSTCLASSIFIER
Same as Decision Tree, the same attribute is used to pull up the important features list.

After running all the 5 algos we get 5 array list of selected features:
1. RFE Set with 21 features
2. UFS Set with 16 features
3. VT Set with 40 features
4. DT Set with 13 features
5. ADA Set with 14 features


## 6. MODEL BUILDING, TESTING

The chosen datasets passes through the list of algorithms and finally the results are collected in a table at the end. This can be compared and contrasted to get the best models.

The Algorithms used are:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Extra Trees Classifier
- Adaboost Classifier With Dtc
- Gradient Boosting Classifier

PREDICTION ON OUT-SAMPLE DATA WITH THE BEST MODEL
From the previous list, the best models with highest ROC-AUC, Recall, 
Precision and F-score is chosen. Those models are then tested on OUT SAMPLE dataset and best model is adjudged the winner. 
In our case we can take the final Logistic Regression on DS_Ada dataset with 14 features, as the winning model with ROC = 0.7591 

ESTIMATED MONETARY IMPACT
Of the total defaulters of 252, the loan amount from [TRUE NEGATIVES = 218]
predicted correctly by the final Logistic model is the monetary savings by the 
company. The savings amount equals to : Rs. 3149675.00


## 7. FINAL RESULTS, RECOMMENDATIONS

Based on ROC-AUC scores, Logistic Regression on DS_Ada dataset with 14 features, is the best model with ROC = 0.7591. 
With additional parametric tuning using algorithms like GridSearchCV, more 
robust results can be obtained.







