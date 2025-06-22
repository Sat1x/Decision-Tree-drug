import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import tree

def preprocess_data(df):
    """
    Preprocesses the drug dataset by encoding categorical variables.
    
    Returns:
        DataFrame: The preprocessed DataFrame.
        dict: A dictionary to decode the 'Drug' target variable.
    """
    # Create a copy to avoid modifying the original DataFrame
    df_processed = df.copy()

    # Define mappings for ordinal features
    bp_mapping = {'LOW': 0, 'NORMAL': 1, 'HIGH': 2}
    cholesterol_mapping = {'NORMAL': 0, 'HIGH': 1}
    sex_mapping = {'F': 0, 'M': 1}
    
    # Apply mappings
    df_processed['BP'] = df_processed['BP'].map(bp_mapping)
    df_processed['Cholesterol'] = df_processed['Cholesterol'].map(cholesterol_mapping)
    df_processed['Sex'] = df_processed['Sex'].map(sex_mapping)
    
    # Encode the target variable 'Drug' and create a decoder
    drug_codes, drug_uniques = pd.factorize(df_processed['Drug'])
    df_processed['Drug'] = drug_codes
    drug_decoder = {i: name for i, name in enumerate(drug_uniques)}
    
    return df_processed, drug_decoder

def plot_decision_tree(model, feature_names, class_names):
    """Plots and saves the decision tree."""
    plt.figure(figsize=(20, 10))
    tree.plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.title("Decision Tree for Drug Classification", fontsize=16)
    plt.savefig('decision_tree.png', dpi=300)
    print("Decision tree plot saved to decision_tree.png")
    plt.show()

def plot_confusion_matrix(y_test, y_pred, class_names):
    """Plots and saves the confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('confusion_matrix.png', dpi=300)
    print("Confusion matrix plot saved to confusion_matrix.png")
    plt.show()

def plot_feature_importance(model, feature_names):
    """Plots and saves the feature importances."""
    importances = model.feature_importances_
    indices = importances.argsort()[::-1]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], palette='viridis')
    plt.title('Feature Importance', fontsize=16)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.savefig('feature_importance.png', dpi=300)
    print("Feature importance plot saved to feature_importance.png")
    plt.show()

def main():
    """Main function to run the drug classification analysis."""
    # --- 1. Load and Preprocess Data ---
    print("--- Loading and Preprocessing Data ---")
    df = pd.read_csv('/home/ivan-koptiev/Codes/Codes/portfolio website/github projects/decision_tree_drug/drug200.csv')
    df_processed, drug_decoder = preprocess_data(df)
    
    # Define features (X) and target (y)
    X = df_processed.drop('Drug', axis=1)
    y = df_processed['Drug']
    feature_names = X.columns.tolist()
    class_names = [drug_decoder[i] for i in sorted(drug_decoder)]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Data split into {len(X_train)} training and {len(X_test)} testing samples.")

    # --- 2. Train the Decision Tree Model ---
    print("\n--- Training Decision Tree Model ---")
    # Using entropy and max_depth=4 for a balanced and interpretable tree
    model = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    print("Model training complete.")

    # --- 3. Evaluate the Model ---
    print("\n--- Evaluating Model ---")
    y_pred = model.predict(X_test)
    
    # Accuracy Score
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Classification Report
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=class_names)
    print(report)

    # --- 4. Visualize the Results ---
    print("\n--- Visualizing Results ---")
    # Plot Decision Tree
    plot_decision_tree(model, feature_names, class_names)
    
    # Plot Confusion Matrix
    plot_confusion_matrix(y_test, y_pred, class_names)
    
    # Plot Feature Importance
    plot_feature_importance(model, feature_names)

if __name__ == "__main__":
    main()