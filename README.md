# DATA-PIPELINE-DEVELOPMENT
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_data(file_path):
    """
    Load data from a CSV file into a Pandas DataFrame.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded data.
    """
    return pd.read_csv(file_path)

def create_preprocessing_pipeline(numeric_features, categorical_features):
    """
    Create a preprocessing pipeline for numeric and categorical features.
    Args:
        numeric_features (list): List of numeric feature column names.
        categorical_features (list): List of categorical feature column names.
    Returns:
        ColumnTransformer: Preprocessing pipeline.
    """
    # Define numeric transformations
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Define categorical transformations
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine both transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    return preprocessor

def preprocess_and_split_data(data, target_column, numeric_features, categorical_features):
    """
    Preprocess the data and split it into training and testing sets.
    Args:
        data (pd.DataFrame): Input data.
        target_column (str): Target column name.
        numeric_features (list): Numeric feature column names.
        categorical_features (list): Categorical feature column names.
    Returns:
        tuple: Processed training and testing sets (X_train, X_test, y_train, y_test).
    """
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features)

    # Fit and transform the data
    X_transformed = preprocessor.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Define file path and column information
    file_path = "data.csv"  # Replace with your file path
    target_column = "target"  # Replace with your target column name
    numeric_features = ["num_feature1", "num_feature2"]  # Replace with your numeric columns
    categorical_features = ["cat_feature1", "cat_feature2"]  # Replace with your categorical columns

    # Load data
    data = load_data(file_path)

    # Preprocess and split data
    X_train, X_test, y_train, y_test = preprocess_and_split_data(
        data, target_column, numeric_features, categorical_features
    )

    # Output shapes
    print("Data preprocessing complete!")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
