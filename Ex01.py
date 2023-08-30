#import relevant packages
import numpy as np # linear algebra
import pandas as pd # data processing
from pandas import Series,DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#<-------------------------------------------------------------------------------------------------->

# read the dataset 
df = pd.read_csv("C:\\Moditha\\Python\\Exercises\\Salary_dataset.csv")

#<-------------------------------------------------------------------------------------------------->

#returns the first 5 rows if a number is not specified
df.head()

#<-------------------------------------------------------------------------------------------------->

#check number of rows and number of columns
df.shape

#<-------------------------------------------------------------------------------------------------->

#check dataset information 
df.info()

#<-------------------------------------------------------------------------------------------------->

#returns description of the data in the DataFrame
df.describe()

#<-------------------------------------------------------------------------------------------------->

#check any missing values
df.isnull().sum()

#<-------------------------------------------------------------------------------------------------->

#any outliers
df.skew()
df.kurt()

#<-------------------------------------------------------------------------------------------------->

#visualization
sns.heatmap(df.corr(),annot=True)
plt.show()

sns.pairplot(df)
plt.show()

plt.plot(df, linestyle = '--', linewidth='5.7')
plt.show()

plt.plot(df, linestyle = '--', linewidth='5.7', color='#FF1493')
plt.show()

df.plot.line(linestyle = ':', linewidth='3')
plt.title('YearExperiene-VS-Salary')
plt.show()


df = pd.DataFrame(data=df)
df.plot.line(x='Salary',linestyle = ':', linewidth='3')

plt.title('YearExperiene-VS-Salary')
plt.show()

#<-------------------------------------------------------------------------------------------------->

#Independent and Dependent Variables
X=df.drop('Salary',axis=1)
y=df.Salary
X.head()
y.head()

#<-------------------------------------------------------------------------------------------------->

#Splitting the data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.30)
print(X_train.shape)

X_test.shape

#<-------------------------------------------------------------------------------------------------->

#Model Fitting
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X_train,y_train)

LR.coef_        # beta 1

#<-------------------------------------------------------------------------------------------------->

#Prediction
y_pred=LR.predict(X_test)
y_pred

y_test

#<-------------------------------------------------------------------------------------------------->

#Evaluation
from sklearn import metrics
R2=metrics.r2_score(y_test,y_pred)
R2
print(metrics.mean_absolute_error(y_test,y_pred))

print(metrics.mean_squared_error(y_test,y_pred))

print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

error= y_test-y_pred
error
