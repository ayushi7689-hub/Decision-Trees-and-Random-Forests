from sklearn.datasets import load_breast_cancer

#Loading Dataset
data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset loaded successfully!")
print("Number of rows:", X.shape[0])
print("Number of features:", X.shape[1])
print("Classes:", data.target_names)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    train_size=0.2,
    random_state=42,
    stratify=y
)
print("Training sample:", X_train.shape[0])
print("Testing sample:", X_test.shape[0])

#Training Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

tree_model = DecisionTreeClassifier(random_state=42)

tree_model.fit(X_train, y_train)

tree_prediction = tree_model.predict(X_test)

tree_accuracy = accuracy_score(y_test, tree_prediction)

print("Decision tree Accuracy:", tree_accuracy)

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))

plot_tree(
    tree_model,
    feature_names=data.feature_names,
    class_names=data.target_names,
    filled=True,
    max_depth=3,
    fontsize=8
)

plt.title("Decision Tree Visualization")
plt.show()


# Deep tree
deep_tree = DecisionTreeClassifier(random_state=42)
deep_tree.fit(X_train, y_train)

# Smaller tree
small_tree = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)
small_tree.fit(X_train, y_train)

print("Deep Tree Training Accuracy:",
      deep_tree.score(X_train, y_train))

print("Deep Tree Testing Accuracy:",
      deep_tree.score(X_test, y_test))

print("Small Tree Training Accuracy:",
      small_tree.score(X_train, y_train))

print("Small Tree Testing Accuracy:",
      small_tree.score(X_test, y_test))

#Training Random Forest
from sklearn.ensemble import RandomForestClassifier

#creating 100 decision trees
forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

forest_model.fit(X_train, y_train)
forest_prediction = forest_model.predict(X_test)
forest_accuracy = accuracy_score(
    y_test,
    forest_prediction
)
print("Random forest accuracy:", forest_accuracy)

print("\n---Model Comparision---")
print(f"Decision Tree: {tree_accuracy:.4f}")
print(f"Random Forest: {forest_accuracy:.4f}")

if forest_accuracy > tree_accuracy:
    print("Random forest performed better.")
elif tree_accuracy > forest_accuracy:
    print("Decision Tree performed better.")
else:
    print("Both models achived same accuracy.")

#Finding Most Important Feature
import pandas as pd

importance = pd.Series(
    forest_model.feature_importances_,
    index=data.feature_names
)

importance = importance.sort_values(ascending=False)

print("\nTop 10 Important Features:")
print(importance.head(10))

# Draw a graph
importance.head(10).sort_values().plot(
    kind="barh",
    figsize=(10, 6)
)

plt.title("Top 10 Feature Importances")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

#Evalute Using Cross Validation
from sklearn.model_selection import cross_val_score

#Evalute Decision Tree
tree_score = cross_val_score(
    DecisionTreeClassifier(random_state=42),
    X, y,
    cv=5,
    scoring="accuracy"
)
#Evalute Random Forest
forest_score = cross_val_score(
    RandomForestClassifier(
        n_estimators=100,
        random_state=42 
    ),
    X, y,
    cv=5,
    scoring="accuracy"
)
print("Decision Tree CV score:", tree_score)
print("Decision Tree Average:", tree_score.mean())
print("\nRandom Forest CV score:", forest_score)
print("Random Forest Average:", forest_score.mean())