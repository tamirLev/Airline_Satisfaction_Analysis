import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.cluster import KMeans , AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score


# קריאת קובץ ה-CSV
datas = pd.read_csv(r"C:\Users\yosia\Documents\לימודים\שנה ד'\ML\פרויקט\Xy_train.csv")

# הסרת ערכים חסרים בעמודת satisfaction
datas = datas.dropna(subset=['satisfaction'])

# המרת עמודת satisfaction לערכים בינאריים
datas['satisfaction'] = datas['satisfaction'].map({"satisfied": 1, "neutral or dissatisfied": 0})

# חלוקת הנתונים לסט אימון וסט בדיקה
X = datas.drop(columns='satisfaction')
y = datas['satisfaction']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 9, random_state=42)

# הסרת ערכים חסרים מ- y_train
mask = y_train.notnull()
X_train = X_train[mask]
y_train = y_train[mask]

 #%%
# פונקציה להכנת הנתונים
def prepare_data(df):
    df = df.copy()

    def clean_column_discrete(df, column_name, valid_values):
        if column_name in df.columns:
            df.loc[:, column_name] = df[column_name].apply(lambda x: x if x in valid_values else np.nan)
        return df

    def clean_column_continuous(df, column_name, valid_range):
        if column_name in df.columns:
            min_val, max_val = valid_range
            df.loc[:, column_name] = df[column_name].apply(lambda x: x if min_val <= x <= max_val else np.nan)
        return df

    columns_to_clean_discrete = {
        'Gender': ['Female', 'Male'],
        'Customer Type': ['Loyal Customer', 'disloyal Customer'],
        'Type of Travel': ['Personal Travel', 'Business travel'],
        'Class': ['Eco', 'Eco Plus', 'Business', 'Unknown'],
    }

    columns_to_clean_continuous = {
        'Age': [0, 120],
        'Flight Distance': [0, 15843],
        'Inflight wifi service': [0, 5],
        'Departure/Arrival time convenient': [0, 5],
        'Ease of Online booking': [0, 5],
        'Gate location': [0, 5],
        'Food and drink': [0, 5],
        'Seat comfort': [0, 5],
        'On-board service': [0, 5],
        'Leg room service': [0, 5],
        'Baggage handling': [0, 5],
        'Checkin service': [0, 5],
        'Inflight service': [0, 5],
        'Cleanliness': [0, 5],
        'Departure Delay in Minutes': [0, 120],
        'Arrival Delay in Minutes': [0, 120]
    }

    for column, valid_values in columns_to_clean_discrete.items():
        df = clean_column_discrete(df, column, valid_values)

    for column, valid_range in columns_to_clean_continuous.items():
        df = clean_column_continuous(df, column, valid_range)

    df = df[df.isnull().sum(axis=1) <= 4]

    columns_to_fill = ['Age', 'Flight Distance', 'Departure Delay in Minutes']
    for column in columns_to_fill:
        df.loc[:, column] = df[column].fillna(df[column].mean())

    columns_to_fill = ['Departure/Arrival time convenient', 'Ease of Online booking', 'Gate location', 'Food and drink',
                       'Seat comfort', 'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service',
                       'Inflight service', 'Cleanliness']
    for column in columns_to_fill:
        df.loc[:, column] = df[column].fillna(df[column].median())

    df.loc[:, 'Gender'] = df['Gender'].fillna('Unknown')
    df.loc[:, 'Arrival Delay in Minutes'] = df.apply(
        lambda row: row['Departure Delay in Minutes'] if pd.isna(row['Arrival Delay in Minutes']) else row[
            'Arrival Delay in Minutes'],
        axis=1
    )
    df.loc[:, 'Inflight wifi service'] = df.apply(
        lambda row: row['Ease of Online booking'] if pd.isna(row['Inflight wifi service']) else row[
            'Inflight wifi service'],
        axis=1
    )

    def fill_missing_with_mode(df, column_name):
        if column_name in df.columns:
            mode_value = df[column_name].mode()[0]
            df.loc[:, column_name] = df[column_name].fillna(mode_value)
        return df

    df = fill_missing_with_mode(df, 'Customer Type')
    df = fill_missing_with_mode(df, 'Type of Travel')

    df.loc[:, 'Class'] = df.apply(
        lambda row: 'Eco' if pd.isna(row['Class']) and row['Type of Travel'] == 'Personal Travel' else (
            'Business' if pd.isna(row['Class']) and row['Type of Travel'] == 'Business travel' else row['Class']),
        axis=1
    )

    categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    # המרת ערכים בוליאניים לערכים בינאריים (0 או 1)
    df_encoded= df_encoded.astype(int)
    df_encoded= df_encoded.astype(int)

    return df_encoded
#%%הכנת הנתונים לאימון ובדיקה
X_train_prepared = prepare_data(X_train)
X_test_prepared = prepare_data(X_test)

# התאמה מחדש של y_train ל-X_train_prepared
y_train = y_train[X_train_prepared.index]

# וידוא שאין שורות חסרות אחרי הכנת הנתונים
print("Shape of X_train_prepared after preparation:", X_train_prepared.shape)
print("Shape of y_train after preparation:", y_train.shape)
#%%נירמול הנתונים
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_prepared)
X_test_scaled = scaler.transform(X_test_prepared)

# וידוא תאימות לאחר נרמול
print("Shape of X_train_scaled:", X_train_scaled.shape)
print("Shape of y_train:", y_train.shape)

#%%

############# neural network ##############


#%%
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler

standard_scaler = StandardScaler()
X_train_s = standard_scaler.fit_transform(X_train_prepared)
X_test_s = standard_scaler.transform(X_test_prepared)  # נירמול סט הבדיקה באותו אופן

minmax_scaler = MinMaxScaler()
X_train_n = minmax_scaler.fit_transform(X_train_prepared)
X_test_n = minmax_scaler.transform(X_test_prepared)

print('train:\n', X_train)
print('train_s:\n', X_train_s)
print('train_n:\n', X_train_n)

#%%
default_model = MLPClassifier(random_state=1,
                      hidden_layer_sizes=(100),
                      activation='relu',
                      max_iter=1000,
                      learning_rate_init=0.001)
default_model.fit(X_train_n, y_train)

#%%
print(f"Number of neurons in the input layer: {X_train_n.shape[1]}")
print(f"Number of hidden layers: {len(default_model.hidden_layer_sizes) if isinstance(default_model.hidden_layer_sizes, tuple) else 1}")
print(f"Number of neurons in each hidden layer: {default_model.hidden_layer_sizes}")
print(f"Activation function: {default_model.activation}")

#%% חיזוי ובדיקת ביצועים
y_pred_train = default_model.predict(X_train_n)
y_pred_test = default_model.predict(X_test_n)

#%% מדדים עבור סט האימון
print("Default neural network On the train set")
print("Accuracy:", "%.6f" % metrics.accuracy_score(y_train, y_pred_train))
print("Precision:", "%.6f" % metrics.precision_score(y_train, y_pred_train))
print("Recall:", "%.6f" % metrics.recall_score(y_train, y_pred_train))
print("F1 Score:", "%.6f" % metrics.f1_score(y_train, y_pred_train))

#%% מדדים עבור סט הבדיקה
print("Default neural network On the test set")
print("Accuracy:", "%.6f" % metrics.accuracy_score(y_test, y_pred_test))
print("Precision:", "%.6f" % metrics.precision_score(y_test, y_pred_test))
print("Recall:", "%.6f" % metrics.recall_score(y_test, y_pred_test))
print("F1 Score:", "%.6f" % metrics.f1_score(y_test, y_pred_test))


#%%
# הגדרת הפרמטרים לכיוונון
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
    'activation': ['tanh', 'relu', 'logistic'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.001, 0.01, 0.05],
    'learning_rate': ['constant', 'adaptive'],
}
# הגדרת GridSearchCV
grid_search = GridSearchCV(estimator=MLPClassifier(random_state=42, max_iter=500),
                           param_grid=param_grid,
                           n_jobs=-1,
                           cv=10,
                           scoring='f1',
                           verbose=2)

# חיפוש הפרמטרים הטובים ביותר
grid_search.fit(X_train_n, y_train)

# הדפסת הפרמטרים הטובים ביותר
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_}")

#%%
#
# פונקציה להצגת גרפים לפי מדד ה-F1
def plot_param_vs_f1_score(param_name, results):
    plt.figure(figsize=(10, 6))
    if results[param_name].dtype == 'O':  # אם העמודה מכילה מחרוזות
        grouped_results = results.groupby(param_name)['mean_test_score'].mean()
        grouped_results.plot(kind='bar')
        plt.xticks(rotation=45)
    else:
        grouped_results = results.groupby(param_name)['mean_test_score'].mean()
        plt.plot(grouped_results.index, grouped_results, marker='o')
    plt.title(f'{param_name} vs. Mean Test F1 Score')
    plt.xlabel(param_name)
    plt.ylabel('Mean Test F1 Score')
    plt.grid(True)
    plt.show()

# הצגת השפעת היפר-פרמטרים שונים על F1 Score
results = pd.DataFrame(grid_search.cv_results_)
plot_param_vs_f1_score('param_hidden_layer_sizes', results)
plot_param_vs_f1_score('param_activation', results)
plot_param_vs_f1_score('param_solver', results)
plot_param_vs_f1_score('param_alpha', results)
plot_param_vs_f1_score('param_learning_rate', results)
#%%
# שימוש במודל עם הפרמטרים הטובים ביותר
best_params = grid_search.best_params_
best_mlp = MLPClassifier(**best_params, random_state=42, max_iter=500)
best_mlp.fit(X_train_n, y_train)

# חיזוי עבור סט האימון
y_pred_train_best = best_mlp.predict(X_train_n)

# חיזוי עבור סט הבדיקה
y_pred_test_best = best_mlp.predict(X_test_scaled)

#%% מדדים עבור סט האימון למודל המשופר
print("Tuned neural network On the train set")
print("Precision:", "%.6f" % metrics.precision_score(y_train, y_pred_train_best))
print("Recall:", "%.6f" % metrics.recall_score(y_train, y_pred_train_best))
print("F1 Score:", "%.6f" % metrics.f1_score(y_train, y_pred_train_best))

#%% מדדים עבור סט הבדיקה למודל המשופר
print("Tuned neural network On the test set")
print("Precision:", "%.6f" % metrics.precision_score(y_test, y_pred_test_best))
print("Recall:", "%.6f" % metrics.recall_score(y_test, y_pred_test_best))
print("F1 Score:", "%.6f" % metrics.f1_score(y_test, y_pred_test_best))

# מדדים כלליים למודל המשופר על סט הבדיקה
print("Best model classification report:\n", metrics.classification_report(y_test, y_pred_test_best))

#%%
import seaborn as sns
from sklearn.metrics import confusion_matrix

# יצירת מטריצת מבוכה
conf_matrix = confusion_matrix(y_test, y_pred_test_best)

# הצגת מטריצת המבוכה באמצעות heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Neutral or Dissatisfied', 'Satisfied'], yticklabels=['Neutral or Dissatisfied', 'Satisfied'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


#%%
import seaborn as sns
from sklearn.metrics import confusion_matrix

# יצירת מטריצת מבוכה
conf_matrix = confusion_matrix(y_train, y_pred_train_best)

# הצגת מטריצת המבוכה באמצעות heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Neutral or Dissatisfied', 'Satisfied'], yticklabels=['Neutral or Dissatisfied', 'Satisfied'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#%% ######################### SVM #######################


#%% SVM
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# יצירת מודל SVM בסיסי ללא כיוונון
svm_model = SVC()

# אימון המודל על הנתונים המנורמלים
svm_model.fit(X_train_scaled, y_train)

# חיזוי על בסיס סט האימון וסט הבחינה
svm_pred_train = svm_model.predict(X_train_scaled)
svm_pred_test = svm_model.predict(X_test_scaled)

# הדפסת תוצאות המודל עבור סט האימון
print("SVM On the train set")
print("Accuracy:", "%.6f" % metrics.accuracy_score(y_train, svm_pred_train))
print("Precision:", "%.6f" % metrics.precision_score(y_train, svm_pred_train))
print("Recall:", "%.6f" % metrics.recall_score(y_train, svm_pred_train))
print("F1 Score:", "%.6f" % metrics.f1_score(y_train, svm_pred_train))

# הדפסת תוצאות המודל עבור סט הבחינה
print("SVM On the test set")
print("Accuracy:", "%.6f" % metrics.accuracy_score(y_test, svm_pred_test))
print("Precision:", "%.6f" % metrics.precision_score(y_test, svm_pred_test))
print("Recall:", "%.6f" % metrics.recall_score(y_test, svm_pred_test))
print("F1 Score:", "%.6f" % metrics.f1_score(y_test, svm_pred_test))

#%%
# הגדרת טווחי הפרמטרים לכיוונון
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear']
}

grid_search = GridSearchCV(svm_model, param_grid, scoring='f1', refit=True, verbose=2, cv=5)
grid_search.fit(X_train_scaled, y_train)


# הצגת הפרמטרים הטובים ביותר
print("Best parameters found: ", grid_search.best_params_)


#%%
# אימון המודל עם הפרמטרים הטובים ביותר
best_svm = grid_search.best_estimator_

# חיזוי על בסיס המודל המכוונן
svm_tuned_pred_train = best_svm.predict(X_train_scaled)
svm_tuned_pred_test = best_svm.predict(X_test_scaled)


# הדפסת תוצאות המודל המכוונן עבור סט האימון
print("Tuned SVM On the train set")
print("Accuracy:", "%.6f" % metrics.accuracy_score(y_train, svm_tuned_pred_train))
print("Precision:", "%.6f" % metrics.precision_score(y_train, svm_tuned_pred_train))
print("Recall:", "%.6f" % metrics.recall_score(y_train, svm_tuned_pred_train))
print("F1 Score:", "%.6f" % metrics.f1_score(y_train, svm_tuned_pred_train))

# הדפסת תוצאות המודל המכוונן עבור סט הבחינה
print("Tuned SVM On the test set")
print("Accuracy:", "%.6f" % metrics.accuracy_score(y_test, svm_tuned_pred_test))
print("Precision:", "%.6f" % metrics.precision_score(y_test, svm_tuned_pred_test))
print("Recall:", "%.6f" % metrics.recall_score(y_test, svm_tuned_pred_test))
print("F1 Score:", "%.6f" % metrics.f1_score(y_test, svm_tuned_pred_test))
#%%
##################### Decision tree #########################

#%%
#%%
# יצירת מודל עץ החלטות ואימון המודל
decision_tree = DecisionTreeClassifier(random_state=0, criterion='entropy')
decision_tree.fit(X_train_scaled, y_train)

# חיזוי על בסיס סט האימון וסט הבחינה
dt_pred_train = decision_tree.predict(X_train_scaled)
dt_pred_test = decision_tree.predict(X_test_scaled)

# הדפסת תוצאות המודל עבור סט האימון
print("Decision Tree On the train set")
print("Accuracy:", "%.6f" % metrics.accuracy_score(y_train, dt_pred_train))
print("Precision:", "%.6f" % metrics.precision_score(y_train, dt_pred_train))
print("Recall:", "%.6f" % metrics.recall_score(y_train, dt_pred_train))
print("F1 Score:", "%.6f" % metrics.f1_score(y_train, dt_pred_train))

# הדפסת תוצאות המודל עבור סט הבחינה
print("Decision Tree On the test")
print("Accuracy:", "%.6f" % metrics.accuracy_score(y_test, dt_pred_test))
print("Precision:", "%.6f" % metrics.precision_score(y_test, dt_pred_test))
print("Recall:", "%.6f" % metrics.recall_score(y_test, dt_pred_test))
print("F1 Score:", "%.6f" % metrics.f1_score(y_test, dt_pred_test))

# שימוש ב-GridSearchCV לכוונון פרמטרים
param_grid = {
    'max_depth': list(range(1, 21)),
    'min_samples_split': [2, 5, 10, 15, 20, 25, 30],
    'min_samples_leaf': [1, 2, 4, 6, 8, 10, 12],
    'criterion': ['gini', 'entropy']
}
grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42), param_grid=param_grid, scoring='f1', cv=10, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# המודל הטוב ביותר
best_dt = grid_search.best_estimator_
# הדפסת הפרמטרים הטובים ביותר
best_params = grid_search.best_params_

print("Tuned Decision Tree")
print(f"Best Parameters: {best_params}")

# תחזיות עבור סט האימון והסט הבדיקה עם המודל המכוונן
tuned_dt_pred_train = best_dt.predict(X_train_scaled)
tuned_dt_pred_test = best_dt.predict(X_test_scaled)

# הדפסת תוצאות המודל המותאם עבור סט האימון
print("Tuned Decision Tree On the train set")
print("Accuracy:", "%.6f" % metrics.accuracy_score(y_train, tuned_dt_pred_train))
print("Precision:", "%.6f" % metrics.precision_score(y_train, tuned_dt_pred_train))
print("Recall:", "%.6f" % metrics.recall_score(y_train, tuned_dt_pred_train))
print("F1 Score:", "%.6f" % metrics.f1_score(y_train, tuned_dt_pred_train))

# הדפסת תוצאות המודל המותאם עבור סט הבחינה
print("Tuned Decision Tree On the test")
print("Accuracy:", "%.6f" % metrics.accuracy_score(y_test, tuned_dt_pred_test))
print("Precision:", "%.6f" % metrics.precision_score(y_test, tuned_dt_pred_test))
print("Recall:", "%.6f" % metrics.recall_score(y_test, tuned_dt_pred_test))
print("F1 Score:", "%.6f" % metrics.f1_score(y_test, tuned_dt_pred_test))

# הצגת חשיבות המשתנים בטבלה
importances = best_dt.feature_importances_
forest_importances = pd.Series(importances, index=X_train_prepared.columns).sort_values(ascending=False)

# הדפסת הטבלה
print("Feature Importances:\n", forest_importances)

# הצגת חשיבות המשתנים בגרף
plt.figure(figsize=(12, 8))
forest_importances.plot(kind='bar')
plt.title('Feature Importances')
plt.ylabel('Importance')
plt.xlabel('Features')
plt.xticks(rotation=60)
plt.tight_layout()
plt.show()

# הצגת גרף של העץ שהתקבל
plt.figure(figsize=(20, 12))
plot_tree(best_dt, feature_names=X_train_prepared.columns, class_names=["neutral or dissatisfied", "satisfied"], filled=True, max_depth=2)
plt.show()
#%%
# הגדרת מספר האשכולות המתאים לבעיה (אצלנו 2-מרוצה לא מרוצה)
n_clusters = 2

# יצירת מודל K-means
kmeans = KMeans(n_clusters=n_clusters, random_state=42)

# התאמת המודל לנתוני האימון
kmeans.fit(X_train_scaled)

# תחזיות אשכולות על סט האימון והבדיקה
train_clusters = kmeans.predict(X_train_scaled)
test_clusters = kmeans.predict(X_test_scaled)

# הצגת תוצאות ה-K-means
print("K-means clustering on the training set")
print("Cluster centers:\n", kmeans.cluster_centers_)
print("Inertia:", kmeans.inertia_)

# השוואת התוויות האמיתיות לאשכולות
ari_train = adjusted_rand_score(y_train, train_clusters)
nmi_train = normalized_mutual_info_score(y_train, train_clusters)
homogeneity_train = homogeneity_score(y_train, train_clusters)
completeness_train = completeness_score(y_train, train_clusters)
v_measure_train = v_measure_score(y_train, train_clusters)

ari_test = adjusted_rand_score(y_test, test_clusters)
nmi_test = normalized_mutual_info_score(y_test, test_clusters)
homogeneity_test = homogeneity_score(y_test, test_clusters)
completeness_test = completeness_score(y_test, test_clusters)
v_measure_test = v_measure_score(y_test, test_clusters)

print(f"Train Adjusted Rand Index (ARI): {ari_train}")
print(f"Train Normalized Mutual Information (NMI): {nmi_train}")
print(f"Train Homogeneity: {homogeneity_train}")
print(f"Train Completeness: {completeness_train}")
print(f"Train V-Measure: {v_measure_train}")

print(f"Test Adjusted Rand Index (ARI): {ari_test}")
print(f"Test Normalized Mutual Information (NMI): {nmi_test}")
print(f"Test Homogeneity: {homogeneity_test}")
print(f"Test Completeness: {completeness_test}")
print(f"Test V-Measure: {v_measure_test}")

# שימוש ב-PCA להורדת ממד לצורך הצגת האשכולות בגרף דו-ממדי
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# הצגת גרף האשכולות בסט האימון
plt.figure(figsize=(12, 8))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=train_clusters, cmap='viridis', marker='o', edgecolor='k', s=50)
plt.title('K-means Clustering of Training Data')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.show()

# הצגת גרף האשכולות בסט הבדיקה
plt.figure(figsize=(12, 8))
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=test_clusters, cmap='viridis', marker='o', edgecolor='k', s=50)
plt.title('K-means Clustering of Test Data')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.show()

#%%
# מציאת מספר האשכולות האופטימלי עם Elbow Method
inertia = []
K = range(1, 9)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_train_scaled)
    inertia.append(kmeans.inertia_)

# הצגת גרף Elbow Method
plt.figure(figsize=(10, 6))
plt.plot(K, inertia, 'bo-')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# הגדרת מספר האשכולות שמצאנו לפי שיטת ELBOW- 3
n_clusters = 3

# יצירת מודל אשכולות היררכיים
hierarchical = AgglomerativeClustering(n_clusters=n_clusters)

# התאמת המודל לנתוני האימון
train_clusters_hierarchical = hierarchical.fit_predict(X_train_scaled)
test_clusters_hierarchical = hierarchical.fit_predict(X_test_scaled)

# הערכת ביצועים
ari_train_hierarchical = adjusted_rand_score(y_train, train_clusters_hierarchical)
nmi_train_hierarchical = normalized_mutual_info_score(y_train, train_clusters_hierarchical)
homogeneity_train_hierarchical = homogeneity_score(y_train, train_clusters_hierarchical)
completeness_train_hierarchical = completeness_score(y_train, train_clusters_hierarchical)
v_measure_train_hierarchical = v_measure_score(y_train, train_clusters_hierarchical)

ari_test_hierarchical = adjusted_rand_score(y_test, test_clusters_hierarchical)
nmi_test_hierarchical = normalized_mutual_info_score(y_test, test_clusters_hierarchical)
homogeneity_test_hierarchical = homogeneity_score(y_test, test_clusters_hierarchical)
completeness_test_hierarchical = completeness_score(y_test, test_clusters_hierarchical)
v_measure_test_hierarchical = v_measure_score(y_test, test_clusters_hierarchical)

# הצגת תוצאות האשכולות היררכיים
print("Hierarchical Clustering on the training set")
print("Train Adjusted Rand Index (ARI):", ari_train_hierarchical)
print("Train Normalized Mutual Information (NMI):", nmi_train_hierarchical)
print("Train Homogeneity:", homogeneity_train_hierarchical)
print("Train Completeness:", completeness_train_hierarchical)
print("Train V-Measure:", v_measure_train_hierarchical)
print("Test Adjusted Rand Index (ARI):", ari_test_hierarchical)
print("Test Normalized Mutual Information (NMI):", nmi_test_hierarchical)
print("Test Homogeneity:", homogeneity_test_hierarchical)
print("Test Completeness:", completeness_test_hierarchical)
print("Test V-Measure:", v_measure_test_hierarchical)

# שימוש ב-PCA להורדת ממד לצורך הצגת האשכולות בגרף דו-ממדי
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# הצגת גרף האשכולות היררכיים בסט האימון
plt.figure(figsize=(12, 8))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=train_clusters_hierarchical, cmap='viridis', marker='o', edgecolor='k', s=50)
plt.title('Hierarchical Clustering of Training Data')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.show()

# הצגת גרף האשכולות היררכיים בסט הבדיקה
plt.figure(figsize=(12, 8))
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=test_clusters_hierarchical, cmap='viridis', marker='o', edgecolor='k', s=50)
plt.title('Hierarchical Clustering of Test Data')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.show()

#%%

########################## Model Test ########################
#%%
X_test_final = pd.read_csv(r"C:\Users\yosia\Documents\לימודים\שנה ד'\ML\פרויקט\X_test.csv")

#%%
#פונקציה חדשה להשלמת הערכים החסרים בסט הבחינה
def prepare_data_for_test(df):
    df = df.copy()

    def clean_column_discrete(df, column_name, valid_values):
        if column_name in df.columns:
            df.loc[:, column_name] = df[column_name].apply(lambda x: x if x in valid_values else np.nan)
        return df

    def clean_column_continuous(df, column_name, valid_range):
        if column_name in df.columns:
            min_val, max_val = valid_range
            df.loc[:, column_name] = df[column_name].apply(lambda x: x if min_val <= x <= max_val else np.nan)
        return df

    columns_to_clean_discrete = {
        'Gender': ['Female', 'Male'],
        'Customer Type': ['Loyal Customer', 'disloyal Customer'],
        'Type of Travel': ['Personal Travel', 'Business travel'],
        'Class': ['Eco', 'Eco Plus', 'Business', 'Unknown'],
    }

    columns_to_clean_continuous = {
        'Age': [0, 120],
        'Flight Distance': [0, 15843],
        'Inflight wifi service': [0, 5],
        'Departure/Arrival time convenient': [0, 5],
        'Ease of Online booking': [0, 5],
        'Gate location': [0, 5],
        'Food and drink': [0, 5],
        'Seat comfort': [0, 5],
        'On-board service': [0, 5],
        'Leg room service': [0, 5],
        'Baggage handling': [0, 5],
        'Checkin service': [0, 5],
        'Inflight service': [0, 5],
        'Cleanliness': [0, 5],
        'Departure Delay in Minutes': [0, 120],
        'Arrival Delay in Minutes': [0, 120]
    }

    for column, valid_values in columns_to_clean_discrete.items():
        df = clean_column_discrete(df, column, valid_values)

    for column, valid_range in columns_to_clean_continuous.items():
        df = clean_column_continuous(df, column, valid_range)


    columns_to_fill = ['Age', 'Flight Distance', 'Departure Delay in Minutes']
    for column in columns_to_fill:
        df.loc[:, column] = df[column].fillna(df[column].mean())

    columns_to_fill = ['Departure/Arrival time convenient', 'Ease of Online booking', 'Gate location', 'Food and drink',
                       'Seat comfort', 'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service',
                       'Inflight service', 'Cleanliness']
    for column in columns_to_fill:
        df.loc[:, column] = df[column].fillna(df[column].median())

    df.loc[:, 'Gender'] = df['Gender'].fillna('Unknown')
    df.loc[:, 'Arrival Delay in Minutes'] = df.apply(
        lambda row: row['Departure Delay in Minutes'] if pd.isna(row['Arrival Delay in Minutes']) else row[
            'Arrival Delay in Minutes'],
        axis=1
    )
    df.loc[:, 'Inflight wifi service'] = df.apply(
        lambda row: row['Ease of Online booking'] if pd.isna(row['Inflight wifi service']) else row[
            'Inflight wifi service'],
        axis=1
    )

    def fill_missing_with_mode(df, column_name):
        if column_name in df.columns:
            mode_value = df[column_name].mode()[0]
            df.loc[:, column_name] = df[column_name].fillna(mode_value)
        return df

    df = fill_missing_with_mode(df, 'Customer Type')
    df = fill_missing_with_mode(df, 'Type of Travel')

    df.loc[:, 'Class'] = df.apply(
        lambda row: 'Eco' if pd.isna(row['Class']) and row['Type of Travel'] == 'Personal Travel' else (
            'Business' if pd.isna(row['Class']) and row['Type of Travel'] == 'Business travel' else row['Class']),
        axis=1
    )

    categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    # המרת ערכים בוליאניים לערכים בינאריים (0 או 1)
    df_encoded= df_encoded.astype(int)
    df_encoded= df_encoded.astype(int)

    return df_encoded

#%%
#קריאה לפונקציה שמשלימה את הערכים

X_test_final_prepared = prepare_data_for_test(X_test_final)

#%%נירמול הנתונים

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler

standard_scaler = StandardScaler()
X_test_final_s = standard_scaler.transform(X_test_final_prepared)  # נירמול סט הבדיקה באותו אופן

#%%
# חיזוי עם המודל המאומן
y_pred_test_final = best_mlp.predict(X_test_final_s)

#%%
#pip install openpyxl
import numpy as np
df = pd.DataFrame(y_pred_test_final, columns=['target'])
# שמירת ה-DataFrame לקובץ אקסל
df.to_excel('airline_G6_ytest.xlsx', index=False)
