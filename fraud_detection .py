#Import required libraries
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler

df = pd.read_csv('C:/Users/USER/Google Drive/04. Lain-lain\hackathon/Training/Fraud Detection/fraud_detection_train.csv')

df.head()
df.columns
df.shape
df.label.value_counts()
var_list = list(df.columns)
data = df.copy()
unique_value_all = {} # checking variance of features
for var in var_list:
    b = pd.DataFrame(df[var].value_counts()) 
    b.columns=[u'count']
    b['percentage']=b['count']/len(data)
    b.sort_values(by=['percentage'],ascending=False)
    unique_value_all[var] =b.iat[0,1]#take unique value with the most apperence
unique_value_data = pd.DataFrame({'Max_Unique_Ratio' :unique_value_all})

#unique_value_data.to_csv('model_rama/fraud_detection_unique_value.csv')
unique_value_data
unique_value_data.sort_values('Max_Unique_Ratio')

data_na  = pd.isna(data).sum()/len(data)
missing_data = pd.DataFrame({'Missing_Ratio' :data_na})#checking missing data
#missing_data.to_csv('model_rama/fraud_detection_missing_value.csv')
missing_data

# select potential feature based on unique ratio of feature
data = data[['los','severitylevel','jnspelsep','jkpst','proc80_99','label','cmg','typeppk','umur','diagprimer','kdkc','dati2']]

# Split the data into numerical and characteristic
data.head()
data_numeric = data.select_dtypes(include=['number']).copy()

scaler = StandardScaler()
smodel = scaler.fit(data_numeric)
df2_num_scale = pd.DataFrame(smodel.transform(data_numeric))
#df2_num_scale = pd.DataFrame(data_numeric)
df2_num_scale.columns = data_numeric.columns
df2_num_scale = pd.DataFrame(data_numeric)
df2_num_scale.columns = data_numeric.columns

data_numeric.columns

df2_num_scale.shape

data_numeric.head()

df2_num_scale.head()

data_char = data.select_dtypes(include=['object']).copy()
ohe = OneHotEncoder(handle_unknown='ignore')
ohe_model = ohe.fit(data_char)

data_char.columns

ohe.get_feature_names(data_char.columns)

data_char.head()

res_OHE = ohe.transform(data_char).toarray()
res_OHE

df2_cat_encode = pd.DataFrame(res_OHE)
df2_cat_encode.columns = ohe.get_feature_names(data_char.columns)
df2_cat_encode.head()

X_train = df2_num_scale.merge(df2_cat_encode, how = 'left', left_index=True, right_index=True)
X_train = X_train.drop('label', axis=1)
X_train.shape
list(X_train.columns.values)

y_train  = data[['label']]
y_train.shape


# validation
X_test = pd.read_csv('C:/Users/USER/Google Drive/04. Lain-lain\hackathon/Training/Fraud Detection/Fraud Detection tahap 2/fraud_detection_val.csv')
X_test.shape

X_test.head()
X_test.columns
X_test.shape
var_list = list(X_test.columns)
data = X_test.copy()
unique_value_all = {} # checking variance of features
for var in var_list:
    b = pd.DataFrame(X_test[var].value_counts()) 
    b.columns=[u'count']
    b['percentage']=b['count']/len(data)
    b.sort_values(by=['percentage'],ascending=False)
    unique_value_all[var] =b.iat[0,1]#take unique value with the most apperence
unique_value_data = pd.DataFrame({'Max_Unique_Ratio' :unique_value_all})

#unique_value_data.to_csv('model_rama/fraud_detection_unique_value.csv')
unique_value_data.sort_values('Max_Unique_Ratio')

data_na  = pd.isna(data).sum()/len(data)
missing_data = pd.DataFrame({'Missing_Ratio' :data_na})#checking missing data
missing_data

# select potential feature based on unique ratio of feature
data = data[['los','severitylevel','jnspelsep','jkpst','proc80_99','cmg','typeppk','umur','diagprimer','kdkc','dati2']]

# Split the data into numerical and characteristic
data.head()
data_numeric = data.select_dtypes(include=['number']).copy()

scaler = StandardScaler()
smodel = scaler.fit(data_numeric)
df2_num_scale = pd.DataFrame(smodel.transform(data_numeric))
#df2_num_scale = pd.DataFrame(data_numeric)
df2_num_scale.columns = data_numeric.columns

data_numeric.columns

df2_num_scale.shape

data_numeric.head()

df2_num_scale.head()

data_char = data.select_dtypes(include=['object']).copy()
ohe = OneHotEncoder(handle_unknown='ignore')
ohe_model = ohe.fit(data_char)

data_char.columns

ohe.get_feature_names(data_char.columns)

data_char.head()

res_OHE = ohe.transform(data_char).toarray()
res_OHE

df2_cat_encode = pd.DataFrame(res_OHE)
df2_cat_encode.columns = ohe.get_feature_names(data_char.columns)
df2_cat_encode.head()

X_test = df2_num_scale.merge(df2_cat_encode, how = 'left', left_index=True, right_index=True)
X_test.shape
X_train.shape
list(X_train.columns.values)
list(X_test.columns.values)


# because the  u'diagprimer_u00_u85', does not exist in the X testing set / validation set, so we create a new dummy variable containing with zero
X_test['diagprimer_u00_u85'] = 0

#-------------------------------Neural Network---------------------------    
model_nn = MLPClassifier()
model_nn.fit(X_train,y_train)

predictions_nn = model_nn.predict_proba(X_test)
pp4 = pd.DataFrame(predictions_nn[:,1])
for i in range(49762):
 if pp4[0][i] > 0.5:
    pp4[0][i] = 1
 else:
    pp4[0][i] = 0     
#lb4 = pd.crosstab(pp4[0], y_test['label'])
#ac4 = accuracy_score([ 1 if p > 0.5 else 0 for p in  predictions_nn[:,1] ], y_test['label'])   

# AUC
#predictions_nn = model_nn.predict_proba(X_test)
#false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predictions_nn[:,1])
#auc(false_positive_rate, true_positive_rate)

result = pd.concat([X_test['visit_id'], pp4],axis=1, join="inner")

result.to_csv("hasil tahap 2.csv", index=False)
#y_test.to_csv("y test.csv", index=True)












from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score

#--------------------------Decision Tree----------------------------
model_dt = DecisionTreeClassifier()
model_dt.fit(X_train, y_train)
X_train.shape

#Make predictions
predictions_dt = model_dt.predict_proba(X_train)
pp2 = pd.DataFrame(predictions_dt[:,1])
for i in range(200217):
 if pp2[0][i] > 0.5:
    pp2[0][i] = 1
 else:
    pp2[0][i] = 0
lb2 = pd.crosstab(pp2[0], y_train['label'])
ac2 = accuracy_score([ 1 if p > 0.5 else 0 for p in  predictions_dt[:,1] ], y_train['label'])

# AUC
predictions_dt = model_dt.predict_proba(X_train)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, predictions_dt[:,1])
auc(false_positive_rate, true_positive_rate)


#----------------------Random Forest------------------------------
model_rf = RandomForestClassifier(n_estimators=1000)
model_rf.fit(X_train,y_train)

#Make predictions
predictions_rf = model_rf.predict_proba(X_train)
pp3 = pd.DataFrame(predictions_rf[:,1])
for i in range(200217):
 if pp3[0][i] > 0.5:
    pp3[0][i] = 1
 else:
    pp3[0][i] = 0
lb3 = pd.crosstab(pp3[0], y_train['label'])
ac3 = accuracy_score([ 1 if p > 0.5 else 0 for p in  predictions_rf[:,1] ], y_train['label'])

# AUC
predictions_rf = model_rf.predict_proba(X_train)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, predictions_rf[:,1])
auc(false_positive_rate, true_positive_rate)

#-------------------------------Neural Network---------------------------    
model_nn = MLPClassifier()
model_nn.fit(X_train,y_train)

#X_test = X_test.reset_index()
#Make predictions
predictions_nn = model_nn.predict_proba(X_train)
pp4 = pd.DataFrame(predictions_nn[:,1])
for i in range(200217):
 if pp4[0][i] > 0.5:
    pp4[0][i] = 1
 else:
    pp4[0][i] = 0     
lb4 = pd.crosstab(pp4[0], y_train['label'])
ac4 = accuracy_score([ 1 if p > 0.5 else 0 for p in  predictions_nn[:,1] ], y_train['label'])   

# AUC
predictions_nn = model_nn.predict_proba(X_train)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, predictions_nn[:,1])
auc(false_positive_rate, true_positive_rate)

ac2
ac3
ac4

pp2.value_counts()