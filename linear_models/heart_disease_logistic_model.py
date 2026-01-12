import os
import pandas as pd
import random
random.seed(1)
import kagglehub
import scipy 
import seaborn as sns
import sklearn 
import statsmodels
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import math

# Download latest version
path = kagglehub.dataset_download("alexteboul/heart-disease-health-indicators-dataset")

print("Path to dataset files:", path)


dataset_path = '/kaggle/input/heart-disease-health-indicators-dataset/heart_disease_health_indicators_BRFSS2015.csv'
df = pd.read_csv(dataset_path)

#Ok so what is actually in this data set

#First get piece together what you are gonna do in collab
  #Explore the data set
    #visualise the data set
    #understand what the data set represents, what you are trying to predict, and from what information
  #clean the data set
    #deal with nans by fill, delete columns, split timeseries
      #fill forward with timeseries, else other ways work
    #deal with outliers
      #winsorise to prevent the variance inflation (if relevant to a linear model)
      #use some kind of threshold for identifying outliers (maybe like IQR)


  #Identify what are your features and what are your labels

  #make the linear model
    #Make sure the data is linearly separable (ie can be divided by some kind of line or hyper plane) #this is a very hard test and you are best served by just making a model)
    #Ensure the gauss markov assumptions are satisfied
      #test for correlations
      #test for the varaince inflation
    #remove the poor predictors from the features
      #conduct F tests
      #report significance after applying bonferroni's correction


    #?????? what does making the linear model actually involve
    #?????? what do we actually present when we are done
      #?????? Visualisations/ plots
      #?????? Results of tests 

    #?????? How do we include regularisation  
  
  

#Then move over to vs code, put it all together in the repo then push it
 

######~~~~~~~EXPLORE THE DATASET~~~~~~#######

###### pandas.PYMETHOD: DataFrame.info(verbose=None, buf=None, max_cols=None, memory_usage=None, show_counts=None)

df.info()

# Ok so this doesnt seem to actually contain any nans in it. In every column the number of non nulls is equal to the number of entries
# 
# 

###### pandas.PYMETHOD: DataFrame.describe(percentiles=None, include=None, exclude=None)

df.describe()

#print(df)

# Ok so some of these columns just have boolean data represented as floats. How do we deal with this?
# I am going to assume that HeartDiseaseorAttack are the labels which we are trying to predict
# Every other column is an input feature. That seems like its almost certainly right


#Now I need to know how to identify outliers in a high dimensional data set.
#This will probably be something like healthy people who get heart attacks or unhealthy people who don't get heart attacks
#At this point I do not know what factors actually are the most important predictors of whether or not someone gets heart disease, so I don't really know what outliers look like
#It could be that there is one feature that is essentially by far the best predictor so anything that deviates from that is an outlier
#I do however have common sense and general knowledge.
#I can imagine health specific factors like BMI, high cholesterol will be the direct causal predictors of heart disease, and correlated significantly with the general physical and mental health indicators. I also reckon that something
#like income will be a higher predictor of the general health indicators, and subsequently, the specific health indicators that serve as good predictors of heart disease.
#To identify Outliers I will make box plots. My initial instince is that from the describe table, not many actually have a maximum that far away from the mean interms of std separation
#and the exceptions like BMI which seem to have maxima a big distance from the mean likely do still contain a lot signal

###### sns.PYMETHOD: seaborn.catplot(data=None, *, x=None, y=None, hue=None, row=None, col=None, kind='strip', estimator='mean', errorbar=('ci', 95), n_boot=1000, seed=None, units=None, weights=None, order=None, hue_order=None, row_order=None, col_order=None, col_wrap=None, height=5, aspect=1, log_scale=None, native_scale=False, formatter=None, orient=None, color=None, palette=None, hue_norm=None, legend='auto', legend_out=True, sharex=True, sharey=True, margin_titles=False, facet_kws=None, ci=<deprecated>, **kwargs)
######sns.PYMETHOD: sns.boxplot(df)
###### NOTE: whiskers are drawn 1.5 IQR from the quartiles

plt.figure(figsize=(50,50))

sns.boxplot(df)

######Now I need to find some way of formatting this better lmao


#this data set could be in children
#The BMI IQR is in the range of moderately overweight individuals (25 to 30)
#This might suggests that the individuals used in this study arent of overwhelmingly poor health
#It would be good to maybe determine what percentile is represented by the 'healthy BMI threshold'of around 18.
#I am interperting Mentl and Phys as 0 is no issues since most are around 0 and there I suspect that they are not of overwhelmingly poor health

#We have a classification task
#We wanna use logistic regression which is gradient based and
#We then do want to winsorise to avoid a high variance messing up your standardisation
#90% windsorisation might be too much so lets start with 99% and go down

###### We are gonna winsorise  BMI, MentHlth, and PhysHlth
###### There are not many outliers below the bottom whisker for any of the categories we are going to windsorise

# get the BMI column

print(df['BMI'])

winbmi=scipy.stats.mstats.winsorize(df['BMI'],limits=[0,0.01])
winphyshlth=scipy.stats.mstats.winsorize(df['PhysHlth'],limits=[0,0.01])
winmenthlth=scipy.stats.mstats.winsorize(df['MentHlth'],limits=[0,0.01])


dfwin=df.copy(deep=True)

dfwin['BMI']=winbmi
dfwin['PhysHlth']=winphyshlth
dfwin['MentHlth']=winmenthlth

#Now re plot the box plot

plt.figure(figsize=(50,50))

sns.boxplot(dfwin)

###Now I wanna describe the new winsorized data frame to compare the variance with the ogs
###There is not much point trying to micromanage what you should do is apply discretion, make a model and see how good it is.
###The range of the three winsorized features is about the same now. Ment and Phys stil kept most of their long tail data, probably because it is more categorical and there were may have been a few 30s. In fact there probably were


### I am gonna assume homoskedasticity and points being generated from the same distribution.
### In a real interview plot the violin plot but for now forgo it



### first we can calculate the correlation matrix for the features

corrmat=dfwin.corr()

plt.figure(figsize=(30,30))

sns.heatmap(corrmat)

###Now calculate the variance inflation factor.


dfwin_vif=statsmodels.tools.tools.add_constant(dfwin)

vifs=pd.Series([variance_inflation_factor(dfwin_vif.values,i) for i in range (0,dfwin_vif.shape[1])], index=dfwin_vif.columns)


print(vifs)

###### The variance inflation factors are all below 5 so we can keep  them all

# Confirm we did not edit the og winsorized dataframe
dfwin.info()
dfwin_vif.info()

df_f=dfwin.copy(deep=True)

targ=df_f['HeartDiseaseorAttack']

del df_f['HeartDiseaseorAttack']

f_p=sklearn.feature_selection.f_classif(df_f, targ)

print(f_p)

##### All the f-scores are pretty good, and all the p values are as well

##### So we won't get rid of any features

##### Now we can try to just implement the logistic model

##### All we need to do is implement a logistic model 

from sklearn.model_selection import train_test_split
train, test = train_test_split(dfwin, train_size = 0.8)
trainx = train.drop(columns=["HeartDiseaseorAttack"])
trainy = train["HeartDiseaseorAttack"]
testx = test.drop(columns=["HeartDiseaseorAttack"])
testy = test["HeartDiseaseorAttack"]


from sklearn.linear_model import LogisticRegression

mean, std = trainx.mean(), trainx.std()

trainx = (trainx - mean) / std
testx = (testx - mean) / std

model = LogisticRegression()
model.fit(trainx, trainy)
pred = model.predict(testx)

from sklearn.metrics import classification_report

print(classification_report(testy, pred))

#####P-R curve

from sklearn.metrics import precision_recall_curve

probs = model.predict_proba(testx)[:, 1]

precision, recall, thresholds = precision_recall_curve(testy, probs)


plt.figure()
plt.plot(recall, precision)

