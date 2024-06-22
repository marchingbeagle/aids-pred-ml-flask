import pickle  
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('data/AIDS_Classification.csv')

df = df[['age', 'wtkg', 'hemo', 'homo', 'drugs', 'oprior',
        'race', 'gender', 'strat', 'symptom', 'treat', 'infected'
         ]]

df.dropna(inplace=True)

X = df.drop(columns=['infected'])
y = df['infected']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2024)            

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model_rf = RandomForestClassifier(n_estimators= 200)
model_rf.fit(X_train, y_train)

y_pred = model_rf.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

with open('model/your_model.pkl', 'wb') as model_file:
    pickle.dump(model_rf, model_file)
with open('model/scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)