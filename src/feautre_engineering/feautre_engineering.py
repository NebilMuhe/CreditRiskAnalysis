import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


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

        # categorical features
        threshold = self.data['Value'].quantile(0.9)
        self.data['Is_High_Value'] = self.data['Value'].apply(lambda x: 1 if x > threshold else 0)

        self.data['Is_Negative_Amount'] = self.data['Amount'].apply(lambda x: 1 if x < 0 else 0)

        # Display the first few rows of the dataset with new features
        return (self.data[
            ["TransactionId","CustomerId","TransactionStartTime", 
             "transaction_hour", "transaction_day", "transaction_month", "transaction_year","Is_High_Value","Is_Negative_Amount"]].head())
    
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
        label_columns = ["CountryCode", "ProviderId","ProductId"]
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
        # numerical_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        # knn_imputer = KNNImputer(n_neighbors=5)
        # numerical_data = self.data[numerical_cols]
        # df_imputed = knn_imputer.fit_transform(numerical_data)

        # # Convert back to DataFrame
        # df = pd.DataFrame(df_imputed, columns=numerical_data.columns)

        # print("After KNN Imputation:")
        # print(df.isnull().sum())

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

    def default_estimator_proxy(self):
        """"
        This function calculates the default estimator proxy for each feature
        """
        print("***Calculating the default estimator proxy***")

        # Convert 'TransactionStartTime' to datetime
        self.data['TransactionStartTime'] = pd.to_datetime(self.data['TransactionStartTime'], errors='coerce')
        
        # recency calculation
        self.data['Recency'] = (self.data['TransactionStartTime'].max() - self.data['TransactionStartTime']).dt.days

        # frequency calculation
        self.data['Frequency'] = self.data.groupby('CustomerId')['TransactionId'].transform('count')

        # monetary calculation
        self.data['Monetary'] = self.data.groupby('CustomerId')['Amount'].transform('sum')

        # stability calculation
        self.data['Stability'] = self.data.groupby('CustomerId')['Amount'].transform('std')

        # visualize RFMS distributions
        sns.pairplot(self.data,vars=['Recency', 'Frequency', 'Monetary', 'Stability'])

        # classify good or bad customers
        self.data['Good_Bad'] = (
            (self.data['Recency'] <= 30) &
            (self.data['Frequency'] >= 5) &
            (self.data['Monetary'] >= 100) &
            (self.data['Stability'] <= 50)
        ).astype(int)

        return self.data.head()

        # Aggregate RFMS metrics
        # rfms = self.data.groupby('CustomerId').agg(
        #     Recency=('TransactionStartTime', lambda x: (today - x.max()).days),
        #     Frequency=('TransactionId', 'count'),
        #     Monetary=('Amount', 'sum'),
        # ).reset_index()

        # scaler = MinMaxScaler()
        # rfms[['Recency', 'Frequency', 'Monetary']] = scaler.fit_transform(rfms[['Recency', 'Frequency', 'Monetary']])

        # # Pair plot to explore RFMS relationships
        # sns.pairplot(rfms, vars=['Recency', 'Frequency', 'Monetary'], diag_kind='kde')
        # plt.show()

        # # 3D scatter plot (optional, requires mpl_toolkits.mplot3d)
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(rfms['Recency'], rfms['Frequency'], rfms['Monetary'], c='blue', alpha=0.6)
        # ax.set_xlabel('Recency')
        # ax.set_ylabel('Frequency')
        # ax.set_zlabel('Monetary')   
        # plt.show()

        # rfms['UserLabel'] = rfms.apply(self.assign_label, axis=1)

        # # K-Means clustering
        # kmeans = KMeans(n_clusters=2, random_state=42)
        # rfms['Cluster'] = kmeans.fit_predict(rfms[['Recency', 'Frequency', 'Monetary']])

        # # Map clusters to 'Good' and 'Bad'
        # cluster_map = {0: 'Good', 1: 'Bad'}  # Adjust based on RFMS space
        # rfms['UserLabel'] = rfms['Cluster'].map(cluster_map)

        # print(rfms['UserLabel'].value_counts())
        # sns.countplot(x='UserLabel', data=rfms)
        # plt.show()

        # sns.scatterplot(
        #     x='Frequency', y='Monetary', hue='UserLabel', data=rfms, palette='Set1'
        # )
        # plt.show()

    def assign_label(self,row):
        if row['Recency'] < 0.5 and row['Frequency'] > 0.5 and row['Monetary'] > 0.5:
            return 'Good'
        else:
            return 'Bad'
        
    def woe_estimator(self):
        """"
        This function calculates the Weight of Evidence (WOE) estimator for each feature
        """
        print("***Calculating the Weight of Evidence (WOE) estimator***")

        # Bin the 'Recency' values
        self.data['Recency_Bin'] = pd.cut(
            self.data['Recency'],
            bins=[0, 10, 20, 30, 60, 90, 120, np.inf],
            labels=['0-10', '10-20', '20-30', '30-60', '60-90', '90-120', '120+']
        )

        # Bin the 'Frequency' values
        self.data['Frequency_Bin'] = pd.cut(
            self.data['Frequency'],
            bins=[0, 5, 10, 20, 50, 100, np.inf],
            labels=['0-5', '5-10', '10-20', '20-50', '50-100', '100+']
        )

        # Bin Monetary
        self.data['Monetary_Bin'] = pd.cut(
            self.data['Monetary'],
            bins=[0, 100, 500, 1000, 5000, 10000, np.inf],
            labels=['0-100', '100-500', '500-1000', '1000-5000', '5000-10000', '10000+']
        )

        # Bin Stability
        self.data['Stability_Bin'] = pd.cut(
            self.data['Stability'],
            bins=[0, 10, 20, 50, 100, np.inf],
            labels=['0-10', '10-20', '20-50', '50-100', '100+']
        )

        # Group by Recency_Bin and calculate Good and Bad counts
        grouped = self.data.groupby('Recency_Bin',observed=False)['Good_Bad'].agg(['count', 'sum'])
        grouped.columns = ['Total', 'Good']
        grouped['Bad'] = grouped['Total'] - grouped['Good']

        # Calculate percentages of Good and Bad in each bin
        grouped['Pct_Good'] = grouped['Good'] / grouped['Good'].sum()
        grouped['Pct_Bad'] = grouped['Bad'] / grouped['Bad'].sum()

        # Calculate WoE
        grouped['WoE'] = np.log(grouped['Pct_Good'] / grouped['Pct_Bad'])
      

        # Calculate IV (Information Value)
        grouped['IV'] = (grouped['Pct_Good'] - grouped['Pct_Bad']) * grouped['WoE']
        iv = grouped['IV'].sum()

        # Display the results
        print("WoE and IV for Recency_Bin:")
        print(grouped)
        print(f"Total IV for Recency_Bin: {iv}")



class AggregateFeatures:
    def __init__(self, data):
        self.data = data

    def customer_level_aggregate_features(self):
        """"
        This function calculates the aggregate features for each customer
        """
        print("***Calculating the aggregate features for each customer***")







