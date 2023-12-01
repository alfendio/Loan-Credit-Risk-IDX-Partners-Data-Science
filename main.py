# %% [markdown]
# # ID/X Partners new_df Scientist Project Based Internship Program

# %% [markdown]
# ## Alfendio Alif Faudisyah

# %% [markdown]
# ![image.png](attachment:image.png)

# %% [markdown]
# Membangun model yang dapat memprediksi credit risk menggunakan new_dfset yang disediakan oleh company yang terdiri dari new_df pinjaman yang diterima dan yang ditolak.

# %% [markdown]
# # Library

# %%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

# %% [markdown]
# # new_df Collection

# %%
df = pd.read_csv('loan_new_df_2007_2014.csv', index_col=0)

# %% [markdown]
# # new_df Understanding

# %%
df.shape

# %%
df.info()

# %%
df.sample()

# %%
df.id.nunique()

# %%
df.member_id.nunique()

# %% [markdown]
# # new_df Preparation

# %%
cols_to_drop = [
    # unique id
    'id'
    , 'member_id'
    
    # free text
    , 'url'
    , 'desc'
    
    # all null / constant / others
    , 'zip_code' 
    , 'annual_inc_joint'
    , 'dti_joint'
    , 'verification_status_joint'
    , 'open_acc_6m'
    , 'open_il_6m'
    , 'open_il_12m'
    , 'open_il_24m'
    , 'mths_since_rcnt_il'
    , 'total_bal_il'
    , 'il_util'
    , 'open_rv_12m'
    , 'open_rv_24m'
    , 'max_bal_bc'
    , 'all_util'
    , 'inq_fi'
    , 'total_cu_tl'
    , 'inq_last_12m'
    
    # expert judgment
    , 'sub_grade'
]

# %%
new_df = df.drop(cols_to_drop, axis=1)

# %%
new_df.sample(10)

# %% [markdown]
# ## Labelling

# %%
new_df.loan_status.value_counts(normalize=True)*100

# %%
bad_status = [
    'Charged Off' 
    , 'Default' 
    , 'Does not meet the credit policy. Status:Charged Off'
    , 'Late (31-120 days)'
]

new_df['bad_flag'] = np.where(new_df['loan_status'].isin(bad_status), 1, 0)

# %%
new_df['bad_flag'].value_counts(normalize=True)*100

# %% [markdown]
# ## Feature Engineering

# %%
new_df['emp_length'].unique()

# %%
new_df['emp_length_int'] = new_df['emp_length'].str.replace('\+ years', '')
new_df['emp_length_int'] = new_df['emp_length_int'].str.replace('< 1 year', str(0))
new_df['emp_length_int'] = new_df['emp_length_int'].str.replace(' years', '')
new_df['emp_length_int'] = new_df['emp_length_int'].str.replace(' year', '')

# %%
new_df['emp_length_int'] = new_df['emp_length_int'].astype(float)

# %%
new_df.drop('emp_length', axis=1, inplace=True)

# %%
new_df['term'].unique()

# %%
new_df['term_int'] = new_df['term'].str.replace(' months', '')
new_df['term_int'] = new_df['term_int'].astype(float)

# %%
new_df.drop('term', axis=1, inplace=True)

# %%
new_df['earliest_cr_line'].head(3)

# %%
new_df['earliest_cr_line_date'] = pd.to_datetime(new_df['earliest_cr_line'], format='%b-%y')
new_df['earliest_cr_line_date'].head(3)

# %%
new_df['mths_since_earliest_cr_line'] = round(pd.to_numeric((pd.to_datetime('2017-12-01') - new_df['earliest_cr_line_date']) / np.timedelta64(1, 'M')))
new_df['mths_since_earliest_cr_line'].head(3)

# %%
new_df['mths_since_earliest_cr_line'].describe()

# %%
new_df[new_df['mths_since_earliest_cr_line']<0][['earliest_cr_line', 'earliest_cr_line_date', 'mths_since_earliest_cr_line']].head(5)

# %%
new_df.loc[new_df['mths_since_earliest_cr_line']<0, 'mths_since_earliest_cr_line'] = new_df['mths_since_earliest_cr_line'].max()

# %%
new_df.drop(['earliest_cr_line', 'earliest_cr_line_date'], axis=1, inplace=True)

# %%
new_df['issue_d_date'] = pd.to_datetime(new_df['issue_d'], format='%b-%y')
new_df['mths_since_issue_d'] = round(pd.to_numeric((pd.to_datetime('2017-12-01') - new_df['issue_d_date']) / np.timedelta64(1, 'M')))

# %%
new_df['mths_since_issue_d'].describe()

# %%
new_df.drop(['issue_d', 'issue_d_date'], axis=1, inplace=True)

# %%
new_df['last_pymnt_d_date'] = pd.to_datetime(new_df['last_pymnt_d'], format='%b-%y')
new_df['mths_since_last_pymnt_d'] = round(pd.to_numeric((pd.to_datetime('2017-12-01') - new_df['last_pymnt_d_date']) / np.timedelta64(1, 'M')))

# %%
new_df['mths_since_last_pymnt_d'].describe()

# %%
new_df.drop(['last_pymnt_d', 'last_pymnt_d_date'], axis=1, inplace=True)

# %%
new_df['next_pymnt_d_date'] = pd.to_datetime(new_df['next_pymnt_d'], format='%b-%y')
new_df['mths_since_next_pymnt_d'] = round(pd.to_numeric((pd.to_datetime('2017-12-01') - new_df['next_pymnt_d_date']) / np.timedelta64(1, 'M')))

# %%
new_df['mths_since_next_pymnt_d'].describe()

# %%
new_df.drop(['next_pymnt_d', 'next_pymnt_d_date'], axis=1, inplace=True)

# %%
new_df['last_credit_pull_d_date'] = pd.to_datetime(new_df['last_credit_pull_d'], format='%b-%y')
new_df['mths_since_last_credit_pull_d'] = round(pd.to_numeric((pd.to_datetime('2017-12-01') - new_df['last_credit_pull_d_date']) / np.timedelta64(1, 'M')))

# %%
new_df['mths_since_last_credit_pull_d'].describe()

# %%
new_df.drop(['last_credit_pull_d', 'last_credit_pull_d_date'], axis=1, inplace=True)

# %% [markdown]
# ## Exploratory new_df Analysis

# %%
plt.figure(figsize=(10,10))
sns.heatmap(new_df.corr())

# %%
corr_matrix = new_df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop_hicorr = [column for column in upper.columns if any(upper[column] > 0.7)]

# %%
to_drop_hicorr

# %%
new_df.drop(to_drop_hicorr, axis=1, inplace=True)

# %%
new_df.select_dtypes(include='object').nunique()

# %%
new_df.drop(['emp_title', 'title', 'application_type'], axis=1, inplace=True)

# %%
new_df.select_dtypes(exclude='object').nunique()

# %%
new_df.drop(['policy_code'], axis=1, inplace=True)

# %%
for col in new_df.select_dtypes(include='object').columns.tolist():
    print(new_df[col].value_counts(normalize=True)*100)
    print('\n')

# %%
new_df.drop('pymnt_plan', axis=1, inplace=True)

# %%
check_missing = new_df.isnull().sum() * 100 / new_df.shape[0]
check_missing[check_missing > 0].sort_values(ascending=False)

# %%
new_df.drop('mths_since_last_record', axis=1, inplace=True)

# %%
new_df['annual_inc'].fillna(new_df['annual_inc'].mean(), inplace=True)
new_df['mths_since_earliest_cr_line'].fillna(0, inplace=True)
new_df['acc_now_delinq'].fillna(0, inplace=True)
new_df['total_acc'].fillna(0, inplace=True)
new_df['pub_rec'].fillna(0, inplace=True)
new_df['open_acc'].fillna(0, inplace=True)
new_df['inq_last_6mths'].fillna(0, inplace=True)
new_df['delinq_2yrs'].fillna(0, inplace=True)
new_df['collections_12_mths_ex_med'].fillna(0, inplace=True)
new_df['revol_util'].fillna(0, inplace=True)
new_df['emp_length_int'].fillna(0, inplace=True)
new_df['tot_cur_bal'].fillna(0, inplace=True)
new_df['tot_coll_amt'].fillna(0, inplace=True)
new_df['mths_since_last_delinq'].fillna(-1, inplace=True)

# %%
categorical_cols = [col for col in new_df.select_dtypes(include='object').columns.tolist()]

# %%
onehot = pd.get_dummies(new_df[categorical_cols], drop_first=True)

# %%
onehot.head()

# %%
numerical_cols = [col for col in new_df.columns.tolist() if col not in categorical_cols + ['bad_flag']]

# %%
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
std = pd.DataFrame(ss.fit_transform(new_df[numerical_cols]), columns=numerical_cols)

# %%
std.head()

# %%
new_df_model = pd.concat([onehot, std, new_df[['bad_flag']]], axis=1)

# %% [markdown]
# # Modeling

# %% [markdown]
# ## Split Dataset

# %%
from sklearn.model_selection import train_test_split

# %%
X = new_df_model.drop('bad_flag', axis=1)
y = new_df_model['bad_flag']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
X_train.shape, X_test.shape

# %% [markdown]
# ## Training

# %%
from sklearn.ensemble import RandomForestClassifier

# %%
rfc = RandomForestClassifier(max_depth=4)
rfc.fit(X_train, y_train)

# %%
arr_feature_importances = rfc.feature_importances_
arr_feature_names = X_train.columns.values
    
df_feature_importance = pd.DataFrame(index=range(len(arr_feature_importances)), columns=['feature', 'importance'])
df_feature_importance['feature'] = arr_feature_names
df_feature_importance['importance'] = arr_feature_importances
df_all_features = df_feature_importance.sort_values(by='importance', ascending=False)
df_all_features

# %%
y_pred_proba = rfc.predict_proba(X_test)[:][:,1]

df_actual_predicted = pd.concat([pd.DataFrame(np.array(y_test), columns=['y_actual']), pd.DataFrame(y_pred_proba, columns=['y_pred_proba'])], axis=1)
df_actual_predicted.index = y_test.index

# %%
from sklearn.metrics import roc_curve, roc_auc_score

# %%
fpr, tpr, tr = roc_curve(df_actual_predicted['y_actual'], df_actual_predicted['y_pred_proba'])
auc = roc_auc_score(df_actual_predicted['y_actual'], df_actual_predicted['y_pred_proba'])

plt.plot(fpr, tpr, label='AUC = %0.4f' %auc)
plt.plot(fpr, fpr, linestyle = '--', color='k')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

# %%
df_actual_predicted = df_actual_predicted.sort_values('y_pred_proba')
df_actual_predicted = df_actual_predicted.reset_index()

df_actual_predicted['Cumulative N Population'] = df_actual_predicted.index + 1
df_actual_predicted['Cumulative N Bad'] = df_actual_predicted['y_actual'].cumsum()
df_actual_predicted['Cumulative N Good'] = df_actual_predicted['Cumulative N Population'] - df_actual_predicted['Cumulative N Bad']
df_actual_predicted['Cumulative Perc Population'] = df_actual_predicted['Cumulative N Population'] / df_actual_predicted.shape[0]
df_actual_predicted['Cumulative Perc Bad'] = df_actual_predicted['Cumulative N Bad'] / df_actual_predicted['y_actual'].sum()
df_actual_predicted['Cumulative Perc Good'] = df_actual_predicted['Cumulative N Good'] / (df_actual_predicted.shape[0] - df_actual_predicted['y_actual'].sum())

# %%
KS = max(df_actual_predicted['Cumulative Perc Good'] - df_actual_predicted['Cumulative Perc Bad'])

plt.plot(df_actual_predicted['y_pred_proba'], df_actual_predicted['Cumulative Perc Bad'], color='r')
plt.plot(df_actual_predicted['y_pred_proba'], df_actual_predicted['Cumulative Perc Good'], color='b')
plt.xlabel('Estimated Probability for Being Bad')
plt.ylabel('Cumulative %')
plt.title('Kolmogorov-Smirnov:  %0.4f' %KS)


