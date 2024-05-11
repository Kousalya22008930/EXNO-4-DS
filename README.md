# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
from scipy import stats
df1=pd.read_csv("/content/bmi.csv")
df2=pd.read_csv("/content/bmi.csv")
df3=pd.read_csv("/content/bmi.csv")
df4=pd.read_csv("/content/bmi.csv")
df5=pd.read_csv("/content/bmi.csv")
df1.head()
```
![image](https://github.com/Kousalya22008930/EXNO-4-DS/assets/119389108/ffe5ea61-7e10-4a21-9edb-ad942f9dc1a5)
```
df.dropna()
```
![image](https://github.com/Kousalya22008930/EXNO-4-DS/assets/119389108/c21a2d76-be2d-432c-84a5-b133074d46ec)
```
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![image](https://github.com/Kousalya22008930/EXNO-4-DS/assets/119389108/eb76d0d8-b049-4c58-a777-87c63cf3b0d3)
```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/Kousalya22008930/EXNO-4-DS/assets/119389108/3ea32523-896a-405d-a1c3-2fc4b59f1322)
```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df2[['Height','Weight']]=sc.fit_transform(df2[['Height','Weight']])
df2.head(10)

```
![image](https://github.com/Kousalya22008930/EXNO-4-DS/assets/119389108/9d9850bc-b908-4b5a-81ac-dd3dbd1bba10)
```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3
```
![image](https://github.com/Kousalya22008930/EXNO-4-DS/assets/119389108/b0f27bb8-16d3-46e7-9cea-a01da8475016)
```
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])
df4
```
![image](https://github.com/Kousalya22008930/EXNO-4-DS/assets/119389108/35957724-97f8-4d37-9798-cbbdc501095b)
```
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df5[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
df5.head()
```
![image](https://github.com/Kousalya22008930/EXNO-4-DS/assets/119389108/f2e2b872-7feb-4c73-8b4d-d75cab87ffbe)
### FEATURE SELECTION :

```
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

data=pd.read_csv('/content/income.csv',na_values=[" ?"])
data
```
![image](https://github.com/Kousalya22008930/EXNO-4-DS/assets/119389108/a6d26c59-ad07-4292-9edb-110d64b9b7b0)
```
data.isnull().sum()
```
![image](https://github.com/Kousalya22008930/EXNO-4-DS/assets/119389108/abb7605f-bb8f-46f7-963a-a4cc376a57e4)
```
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/Kousalya22008930/EXNO-4-DS/assets/119389108/676aeab0-a1ed-4fd2-b265-34a91cf7ec1e)
```
data2=data.dropna(axis=0)
data2
```
![image](https://github.com/Kousalya22008930/EXNO-4-DS/assets/119389108/e3ad3eba-76ce-4e9e-915c-0804fb595a0d)
```
sal=data['SalStat']
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,'greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/Kousalya22008930/EXNO-4-DS/assets/119389108/dd8b4b3a-d379-48b0-a2ea-ccd111c194f1)
```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/Kousalya22008930/EXNO-4-DS/assets/119389108/8734914c-9234-4edc-91dc-7c2eb7421f47)
```
new_data=pd.get_dummies(data2,drop_first=True)
new_data
```
![image](https://github.com/Kousalya22008930/EXNO-4-DS/assets/119389108/fec138c8-211d-4984-97de-db8a93e940c0)
### FEATURE SELECTION METHOD IMPLEMENTATION :
```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/Kousalya22008930/EXNO-4-DS/assets/119389108/1c9db717-75db-42ac-a8dc-9c99982b4872)
```
#seperating the input names from data
features=list(set(columns_list)-set(['SalStat']))
print(features)
```

![image](https://github.com/Kousalya22008930/EXNO-4-DS/assets/119389108/5a1abd80-cab4-4b3b-a882-4f786682405f)
```
#storing the output values in y
y=new_data['SalStat'].values
print(y)
```

![image](https://github.com/Kousalya22008930/EXNO-4-DS/assets/119389108/ed4daf6a-8b0b-4ce6-8ebe-589b8aa3fe33)
```
x = new_data[features].values
print(x)
```


# RESULT:
Thus, Feature selection and Feature scaling has been used on the given dataset.

