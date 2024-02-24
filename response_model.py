from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report 

import pandas as pd 

merged_df = pd.read_csv("Data/preprocessed_responses.csv")

X = merged_df.drop('Diagnosis', axis=1)
y = merged_df['Diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

total_samples = len(y_train)
class_0_samples = (y_train == 0).sum()
class_1_samples = (y_train == 1).sum()

weight_for_class_0 = total_samples / (2 * class_0_samples)
weight_for_class_1 = total_samples / (2 * class_1_samples)

class_weights = {0: weight_for_class_0, 1: weight_for_class_1}
dt_classifier = DecisionTreeClassifier(class_weight=class_weights, random_state=42, max_depth=5)

dt_classifier.fit(X_train, y_train)

y_pred = dt_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print(classification_report(y_test, y_pred))