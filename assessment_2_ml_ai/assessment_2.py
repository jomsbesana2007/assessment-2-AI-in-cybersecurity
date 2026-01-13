# DATA PREPROCESSING

# Import necessary modules
import pandas as pd

# Import dataset
heart_disease_df = pd.read_csv('assessment_2_ml_ai/heart.csv')

heart_disease_df['Sex'] = heart_disease_df['Sex'].map({'F': 0, 'M':1})
heart_disease_df['ST_Slope'] = heart_disease_df['ST_Slope'].map({'Down': 0, 'Flat': 1, 'Up': 2})
heart_disease_df['ExerciseAngina'] = heart_disease_df['ExerciseAngina'].map({'N': 0, 'Y': 1})

heart_disease_df = pd.get_dummies(heart_disease_df, drop_first=True)
heart_disease_df = heart_disease_df.fillna(0) # Replaces NaN values introduced with pd.get_dummies with 0 to avoid issues

# Checking correlation to choose strongly correlated features (X)
print(heart_disease_df.corr(method='pearson')['HeartDisease'])


# MODEL DEVELOPMENT

# Import necessary modules
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split

# Split data into training and testing sets
features = ['Age', 'Sex', 'Cholesterol', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA', 'MaxHR', 'ST_Slope', 'Oldpeak', 'ExerciseAngina'] # Represents the X values

X = heart_disease_df[features]
y = heart_disease_df['HeartDisease'] # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fitting K-Nearest Neighbor model
ml1 = tree.DecisionTreeClassifier(random_state=42, max_depth=None)
ml1.fit(X_train, y_train)

# Fitting Random Forest model
ml2 = RandomForestClassifier(random_state=42, max_depth=None)
ml2.fit(X_train, y_train)


# PREDICTING
models = {'Decision Tree': ml1, 'Random Forest': ml2}
predictions = {}

# Predicting the target variable or 'HeartDisease' in a for loop
for model_type, model in models.items():
   pred = model.predict(X_test)
   predictions[model_type] = pred


# MODEL EVALUATION
# Import necessary modules
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

for model, model_pred in predictions.items():
   model_accuracy = accuracy_score(y_test, model_pred)
   model_precision = precision_score(y_test, model_pred)
   model_recall = recall_score(y_test, model_pred)
   model_roc_auc = roc_auc_score(y_test, model_pred)
   model_f1 = f1_score(y_test, model_pred)
   
   print(" ") # Spacing
   print(f"Model : {model}")
   print(f"Accuracy : {model_accuracy}")
   print(f"Precision : {model_precision}")
   print(f"Recall : {model_recall}")
   print(f"F1 Score : {model_f1}")
   print(f"ROC-AUC : {model_roc_auc}")

   fig, cm_ax = plt.subplots(figsize=(6,6)) # Sizing

   # Confusion matrix
   cm = confusion_matrix(y_test, model_pred)
   cmdisplay = ConfusionMatrixDisplay(confusion_matrix=cm)
   cmdisplay.plot(ax=cm_ax)
   plt.title(model)


# ROC Curve
def roc_curve_calc(model_pred):
   fpr, tpr, _ = roc_curve(y_test, model_pred)
   roc_auc = auc(fpr, tpr)
   model_roc = {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr}
   return model_roc

ml1_roc = roc_curve_calc(predictions['Decision Tree'])
ml2_roc = roc_curve_calc(predictions['Random Forest'])

plt.figure(figsize=(8,6))
plt.plot(ml1_roc['fpr'], ml1_roc['tpr'], color='blue', lw=2, label=f'Decision Tree (ML1 AUC: {ml1_roc['roc_auc']:.2f})')
plt.plot(ml2_roc['fpr'], ml2_roc['tpr'], color='orange', lw=2, label=f'Random Forest (ML2 AUC: {ml2_roc['roc_auc']:.2f})')
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.legend(loc="lower right")
plt.title('Receiver Operating Characteristic (ROC)')
plt.ylabel('True Positive Rate (TPR)')
plt.xlabel('False Positive Rate (FPR)')

# Show all graphs
plt.show()