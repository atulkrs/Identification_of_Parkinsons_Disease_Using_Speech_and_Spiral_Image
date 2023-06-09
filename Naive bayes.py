import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load data
data = pd.read_csv('UpdatedAudioFeature.csv')

# Split data into features (X) and labels (y)
X=data.drop(['label'], axis=1)
y = data['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions on testing set
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % (accuracy*100))
report = classification_report(y_test, y_pred, digits=4, output_dict=True)
mat=confusion_matrix(y_test,y_pred)
print('Classification Report:')
sns.heatmap(mat, annot=True, cmap='Blues', fmt='g')
print(pd.DataFrame(report).transpose())
print('Sensitivity: '+str((mat[0][0]+mat[1][0])/mat[0][0]))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# temp=model.predict(X_test)
# ypred=[]
# for i in temp:
#     ypred.append(np.argmax(i))
# print(ypred)
# mat=confusion_matrix(y_test,ypred)
# print(mat)
# print(classification_report(y_test,ypred))
# print('Sensitivity: '+str((mat[0][0]+mat[1][0])/mat[0][0]))

# sns.heatmap(mat, annot=True, cmap='Blues', fmt='g')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()
