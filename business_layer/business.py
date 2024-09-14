import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.stats import randint, uniform
import joblib
import gc
import traceback

model = None


def load_data():
    data = sns.load_dataset('titanic')
    data = data.drop(columns=['alive', 'deck', 'embark_town', 'class', 'who', 'adult_male', 'alone'])
    data['survived'] = data['survived'].astype('category')

    mean_age = data['age'].mean()
    data['age'] = data['age'].fillna(mean_age)

    for col in data.columns:
        if data[col].dtype.name == 'category':
            data[col] = data[col].cat.add_categories('Unknown')
            data[col] = data[col].fillna('Unknown')
        elif data[col].dtype == 'object':
            data[col] = data[col].fillna('Unknown')

    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    df_categorical = pd.get_dummies(data[categorical_cols]).astype(int)
    df_processed = pd.concat([data[numerical_cols], df_categorical], axis=1)
    df_processed = df_processed.drop(columns=[col for col in df_processed.columns if 'survived' in col])

    X = df_processed
    y = data['survived'].cat.codes

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, scaler, categorical_cols


def train_model(algorithm):
    global model
    gc.collect()  # Clear garbage before training to avoid memory issues
    X_train, X_test, y_train, y_test, scaler, categorical_cols = load_data()

    param_grids = {
        'logistic_regression': {
            'max_iter': randint(100, 1000),
            'C': uniform(0.01, 10)
        },
        'decision_tree': {
            'max_depth': randint(1, 20),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 20)
        },
        'mlp': {
            'hidden_layer_sizes': [(randint(50, 150).rvs(),), (randint(50, 150).rvs(), randint(50, 150).rvs())],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha': uniform(0.0001, 0.1),
            'learning_rate': ['constant', 'adaptive']
        }
    }

    if algorithm == 'logistic_regression':
        base_model = LogisticRegression()
        param_grid = param_grids['logistic_regression']
    elif algorithm == 'decision_tree':
        base_model = DecisionTreeClassifier()
        param_grid = param_grids['decision_tree']
    elif algorithm == 'mlp':
        base_model = MLPClassifier(max_iter=1000)
        param_grid = param_grids['mlp']
    else:
        return {'error': 'Invalid algorithm specified.'}, 400

    random_search = RandomizedSearchCV(estimator=base_model, param_distributions=param_grid, n_iter=20, cv=5, verbose=1,
                                       random_state=42, n_jobs=-1)
    random_search.fit(X_train, y_train)

    model = random_search.best_estimator_

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    joblib.dump((model, scaler, categorical_cols), 'titanic_model.pkl')

    return {'message': f'Model trained using {algorithm} with best parameters found by RandomizedSearchCV.',
            'accuracy': accuracy, 'best_params': random_search.best_params_}


def load_model():
    global model, scaler, categorical_cols
    model, scaler, categorical_cols = joblib.load('titanic_model.pkl')


def predict_survival(input_data):
    global model, scaler, categorical_cols
    if not model:
        return {'error': 'Model not trained. Use /train endpoint first.'}, 400

    try:
        # Ensure all features expected by the model are present
        input_df = pd.DataFrame([input_data], columns=[
            'pclass', 'age', 'sibsp', 'parch', 'fare',
            'sex', 'embarked'
        ])

        # Add missing numerical features with default values
        numerical_features = ['pclass', 'age', 'sibsp', 'parch', 'fare']
        for feature in numerical_features:
            if feature not in input_df.columns:
                input_df[feature] = 0.0

        # Fill missing numerical features with mean or default values
        mean_age = input_df['age'].mean()
        input_df['age'] = input_df['age'].fillna(mean_age)

        # Fill missing categorical features with 'Unknown'
        for col in input_df.columns:
            if col in categorical_cols and col != 'survived':  # Exclude 'survived'
                input_df[col] = input_df[col].fillna('Unknown')

        # Transform numerical columns using the scaler
        input_df[numerical_features] = scaler.transform(input_df[numerical_features])

        # One-Hot Encode categorical columns and align with training data
        categorical_features_to_encode = [col for col in categorical_cols if col != 'survived']
        input_transformed = pd.get_dummies(input_df[categorical_features_to_encode]).astype(int)
        input_transformed = pd.concat([input_df[numerical_features], input_transformed], axis=1)

        # Add any missing columns from training time with zeros
        missing_cols = set(model.feature_names_in_) - set(input_transformed.columns)
        for col in missing_cols:
            input_transformed[col] = 0

        # Ensure the columns are in the correct order as expected by the model
        input_transformed = input_transformed.reindex(columns=model.feature_names_in_, fill_value=0)

        # Predict using the model
        proba = model.predict_proba(input_transformed)[0]
        survived_probability = float(proba[1])
        survived = bool(survived_probability >= 0.5)

        return {'survived': survived, 'survival_probability': survived_probability}

    except Exception as e:
        # Print the traceback for debugging
        print("Error in predict_survival:", e)
        print(traceback.format_exc())
        return {'error': str(e)}, 500


