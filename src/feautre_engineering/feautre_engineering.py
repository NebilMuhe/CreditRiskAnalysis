import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np


class FeautreEngineering:
    def __init__(self, data):
        self.data = data

    def aggregate_data(self):
        """"
        This function aggregates the data by grouping it by the specified columns
        """
        print("***Aggregating the data***")

        # Ensure the 'Amount' column is numeric (convert if needed)
        self.data["Amount"] = pd.to_numeric(self.data["Amount"], errors="coerce")
        # Group by CustomerId and calculate aggregate features 
        aggregate_features = self.data.groupby(["CustomerId","AccountId"]).agg(
            total_transaction_amount=("Amount", "sum"),
            avg_transaction_amount=("Amount", "mean"),
            transaction_count=("Amount", "count"),
            std_transaction_amount=("Amount", "std"),
            max_transaction_amount=("Amount", "max"),
            min_transaction_amount=("Amount", "min"),
            median_transaction_amount=("Amount", "median")
        ).reset_index()

        # Fill NaN values in std_transaction_amount (in case of single transactions)
        aggregate_features.fillna({"std_transaction_amount": 0}, inplace=True)
        # Display the first few rows of the aggregated features
        return aggregate_features.head()

    def extract_transactions_data(self):
        """"
        This function extracts the transactions data from the data
        """
        print("***Extracting transactions data***")
        self.data["TransactionStartTime"] = pd.to_datetime(self.data["TransactionStartTime"], errors="coerce")
        # Extract features from the 'TransactionStartTime'
        self.data["transaction_hour"] = self.data["TransactionStartTime"].dt.hour
        self.data["transaction_day"] = self.data["TransactionStartTime"].dt.day
        self.data["transaction_month"] = self.data["TransactionStartTime"].dt.month
        self.data["transaction_year"] = self.data["TransactionStartTime"].dt.year

        # Display the first few rows of the dataset with new features
        return (self.data[
            ["TransactionId","CustomerId","TransactionStartTime", 
             "transaction_hour", "transaction_day", "transaction_month", "transaction_year"]].head())
    
    def encode_categorical_data(self):
        """"
        This function encodes the categorical data in the data
        """
        print("***Encoding categorical data***")

        label_encoder = LabelEncoder()
        
        # Drop unnecessary columns
        data = self.data
        columns_to_drop = ["TransactionId", "BatchId", "AccountId", "SubscriptionId", "CustomerId"]
        data = data.drop(columns=columns_to_drop)

        # One-Hot Encoding for categorical features
        one_hot_columns = ["CurrencyCode", "ProductCategory", "ChannelId", "PricingStrategy"]
        data = pd.get_dummies(data, columns=one_hot_columns, drop_first=True)

        print("One-Hot Encoded Columns:")
        print(data.head())

        # Label Encoding for categorical features
        label_columns = ["CountryCode", "ProviderId"]
        for col in label_columns:
            data[col] = label_encoder.fit_transform(data[col].astype(str))
        
        print("Label Encoded Columns:")
        print(data.head())

    def missing_values(self):
        """"
        This function handles missing values in the data
        """
        print("***Handling missing values***")
        # Check for missing values
        missing_values = self.data.isnull().sum()
        missing_percentage = (self.data.isnull().sum() / len(self.data)) * 100

        print("Missing Values Count:")
        print(missing_values)
        print("\nMissing Values Percentage:")
        print(missing_percentage)

        # Separate numerical and categorical columns
        numerical_cols = self.data.select_dtypes(include=np.number).columns
        categorical_cols = self.data.select_dtypes(exclude=np.number).columns

        # Impute numerical features with median
        median_imputer = SimpleImputer(strategy="median")
        self.data[numerical_cols] = median_imputer.fit_transform(self.data[numerical_cols])

        # Impute categorical features with mode
        mode_imputer = SimpleImputer(strategy="most_frequent")
        self.data[categorical_cols] = mode_imputer.fit_transform(self.data[categorical_cols].astype(str))

        print("After Imputation:")
        print(self.data.isnull().sum())

        # Use KNN Imputer
        numerical_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        knn_imputer = KNNImputer(n_neighbors=5)
        numerical_data = self.data[numerical_cols]
        df_imputed = knn_imputer.fit_transform(numerical_data)

        # Convert back to DataFrame
        df = pd.DataFrame(df_imputed, columns=numerical_data.columns)

        print("After KNN Imputation:")
        print(df.isnull().sum())

    def normalize_data(self):
        """"
        This function normalizes the data
        """
        print("***Normalizing the data***")

        #  Select numerical columns
        numerical_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        numerical_data = self.data[numerical_cols]

        # Apply normalization
        minmax_scaler = MinMaxScaler()
        normalized_data = pd.DataFrame(minmax_scaler.fit_transform(numerical_data), columns=numerical_cols)

        print("Normalized Data:")
        print(normalized_data.head())   

        # Apply standardization
        standard_scaler = StandardScaler()
        standardized_data = pd.DataFrame(standard_scaler.fit_transform(numerical_data), columns=numerical_cols)

        print("Standardized Data:")
        print(standardized_data.head())


