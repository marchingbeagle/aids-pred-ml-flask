import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import OrdinalEncoder

df = pd.read_csv('/model/asthma_disease_data.csv')

df = df [['Gender', 'Smoking', 'BMI', 'PhysicalActivity', 'ShortnessOfBreath','FamilyHistoryAsthma','HistoryOfAllergies', 'Coughing' ,'ChestTightness' ,'Diagnosis']]

scaler = MinMaxScaler()
columns_to_scale = ['BMI', 'PhysicalActivity']
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])


x = df.drop(collumns = 'Diagnosis')
y = df['Diagnosis']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)