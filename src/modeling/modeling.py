from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,precision_score,recall_score,f1_score,roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from scipy.stats import randint,loguniform

class Model:
    def __init__(self, data):
        self.data = data
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.train_data = None
        self.X = None
        self.y = None
        self.preprocessor = None

    def preprocess_data(self):
        """
        Preprocess the data: handle missing values, scale numeric features, and encode categorical features
        """
        self.X = self.data.drop(columns=['FraudResult'])
        self.y = self.data['FraudResult']
        numeric_features = self.X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = self.X.select_dtypes(include=['object']).columns
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)])
       
        
        self.X = self.preprocessor.fit_transform(self.X)
        print(f"Data preprocessing completed: Shape of processed data = {self.X.shape}")

    def split_data(self):
        """ Split data into train and test sets """  # Target column

        # Split the data (80% training, 20% testing)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        print(f"Training set: {self.X_train.shape}, {self.y_train.shape}")
        print(f"Testing set: {self.X_test.shape}, {self.y_test.shape}")

    def train_model(self):
        """ Train a model """
        # Initialize the model
        log_reg = LogisticRegression(random_state=42)

        # Train the model
        log_reg.fit(self.X_train, self.y_train)

        # Predict on the test set
        y_pred_log_reg = log_reg.predict(self.X_test)

        # Evaluate the model
        print("Logistic Regression Results:")
        print(f"Accuracy: {accuracy_score(self.y_test, y_pred_log_reg)}")
        print(f"Precision: {precision_score(self.y_test, y_pred_log_reg)}")
        print(f"Recall: {recall_score(self.y_test, y_pred_log_reg)}")
        print(f"F1 Score: {f1_score(self.y_test, y_pred_log_reg)}")
        print(f"ROC AUC Score: {roc_auc_score(self.y_test, y_pred_log_reg)}")
        print(f"Classification report:\n{classification_report(self.y_test, y_pred_log_reg)}")
        print(f"Confusion matrix{confusion_matrix(self.y_test, y_pred_log_reg)}")

        # Initialize the model
        rf = RandomForestClassifier(random_state=42)

        # Train the model
        rf.fit(self.X_train, self.y_train)

        # Predict on the test set
        y_pred_rf = rf.predict(self.X_test)

        # Evaluate the model
        print("Random Forest Results:")
        print(f"Accuracy: {accuracy_score(self.y_test, y_pred_log_reg)}")
        print(f"Precision: {precision_score(self.y_test, y_pred_log_reg)}")
        print(f"Recall: {recall_score(self.y_test, y_pred_log_reg)}")
        print(f"F1 Score: {f1_score(self.y_test, y_pred_log_reg)}")
        print(f"ROC AUC Score: {roc_auc_score(self.y_test, y_pred_log_reg)}")
        print(f"Classification report:\n{classification_report(self.y_test, y_pred_rf)}")
        print(f"Confusion matrix{confusion_matrix(self.y_test, y_pred_rf)}")

    
    def grid_search_lg(self):
        """ Perform a grid search for the best hyperparameters """

        # Initialize the model
        log_reg = LogisticRegression(random_state=42)


        # Define the parameter grid
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],  
            'penalty': ['l1', 'l2'],  
            'solver': ['liblinear', 'saga'],
        }

        # Perform grid search
        grid_search = GridSearchCV(estimator=log_reg, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)

        # Print the best parameters
        print("Best parameters:",grid_search.best_params_)

        # Print the best score
        print("Best score:", grid_search.best_score_)

        best_lg = grid_search.best_estimator_
        y_pred_best_lg = best_lg.predict(self.X_test)

        # Evaluate the tuned model
        print("Tuned Logistic Regresssion Results:")
        print(f"Accuracy: {accuracy_score(self.y_test, y_pred_best_lg)}")
        print(f"Classification report:\n{classification_report(self.y_test, y_pred_best_lg)}")
        print(f"Confusion matrix{confusion_matrix(self.y_test, y_pred_best_lg)}")


    def random_search_lg(self):
        """ Perform a random search for the best hyperparameters """

        # Initialize the model
        log_reg = LogisticRegression(random_state=42)

        # Define hyperparameter distributions
        param_dist = {
            'C': loguniform(1e-4, 100), 
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [100, 200, 300]
        }


        # Perform random search
        random_search = RandomizedSearchCV(estimator=log_reg, param_distributions=param_dist, cv=5, scoring='accuracy', n_jobs=-1)
        random_search.fit(self.X_train, self.y_train)

        # Print the best parameters
        print("Best parameters:", random_search.best_params_)
        # Print the best score
        print("Best score:", random_search.best_score_)

        best_lg = random_search.best_estimator_
        y_pred_best_lg = best_lg.predict(self.X_test)

        # Evaluate the tuned model
        print("Tuned model accuracy:", accuracy_score(self.y_test, y_pred_best_lg))
        # Print the tuned model's parameters
        print("Tuned model's parameters:", best_lg.get_params())    
        print(f"Classification report:\n{classification_report(self.y_test, y_pred_best_lg)}")
        print(f"Confusion matrix{confusion_matrix(self.y_test, y_pred_best_lg)}")

        return y_pred_best_lg

    def grid_search_rf(self):
        """ Perform a grid search for the best hyperparameters """

        # Initialize model
        rf = RandomForestClassifier()

         # Define the parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        # Perform grid search
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)

        # Print the best parameters
        print("Best parameters:",grid_search.best_params_)

        # Print the best score
        print("Best score:", grid_search.best_score_)

        best_rf = grid_search.best_estimator_
        y_pred_best_rf = best_rf.predict(self.X_test)

        # Evaluate the tuned model
        print("Tuned Random Forest Results:")
        print(f"Accuracy: {accuracy_score(self.y_test, y_pred_best_rf)}")
        print(f"Classification report:\n{classification_report(self.y_test, y_pred_best_rf)}")
        print(f"Confusion matrix{confusion_matrix(self.y_test, y_pred_best_rf)}")

        return y_pred_best_rf
        
    
    def random_search_rf(self):
        """ Perform a random search for the best hyperparameters """

        # Initialize model
        rf = RandomForestClassifier()

        # Define the parameter grid
        param_dist = {
            'n_estimators': randint(100, 500),
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': randint(2, 10),
            'min_samples_leaf': randint(1, 4),
            'learning_rate': [0.01, 0.1, 0.2]
        }

        # Perform random search
        random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, cv=5, scoring='accuracy', n_jobs=-1)
        random_search.fit(self.X_train, self.y_train)

        # Print the best parameters
        print("Best parameters:", random_search.best_params_)
        # Print the best score
        print("Best score:", random_search.best_score_)

        best_rf = random_search.best_estimator_
        y_pred_best_rf = best_rf.predict(self.X_test)

        # Evaluate the tuned model
        print("Tuned model accuracy:", accuracy_score(self.y_test, y_pred_best_rf))
        # Print the tuned model's parameters
        print("Tuned model's parameters:", best_rf.get_params())    
        print(f"Classification report:\n{classification_report(self.y_test, y_pred_best_rf)}")
        print(f"Confusion matrix{confusion_matrix(self.y_test, y_pred_best_rf)}")

        return y_pred_best_rf           


        
