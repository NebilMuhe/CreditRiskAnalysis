import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


class EDA:
    def __init__(self, data):
        self.data = data
    
    def overview_data(self):
        """"
        This function gives an overview of the data
        """
        print("***Overview of the data***")
        print(f"First 5 rows of the data:\n {self.data.head(5)}\n")
        print(f"Last 5 rows of the data: \n{self.data.tail(5)}\n")
        print(f"Shape of the dataset(rows, columns): {self.data.shape}\n")
        print(f"Column names:\n {self.data.columns}\n")
        print(f"Data types of the columns:\n {self.data.dtypes}\n")
        print(f"Missing values in the dataset:\n {self.data.isnull().sum()}\n")

    def summary_data(self):
        """"
        This function gives a summary of the data
        """
        print("***Summary of the data***")
        print(self.data.describe())

    def numerical_distribution(self):
        """"
        This function gives the numerical distribution of the data
        """
        print("***Numerical Distribution of the data***")
        # Select numerical columns
        numerical_columns = self.data.select_dtypes(include=["float64", "int64"]).columns

        # Visualize the distribution of numerical features
        for column in numerical_columns:
            plt.figure(figsize=(8, 5))
            
            # Histogram with KDE
            sns.histplot(self.data[column], kde=True, bins=30, color="blue", alpha=0.7)
            plt.title(f"Distribution of {column}", fontsize=14)
            plt.xlabel(column, fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            plt.grid(axis="y", linestyle="--", alpha=0.6)
            
            plt.show()

    def categorical_distribution(self):
        """"
        This function gives the categorical distribution of the data
        """
        print("***Categorical Distribution of the data***")

        # Select categorical columns
        categorical_columns = self.data.select_dtypes(include=["object", "category"]).columns

        # Visualize the distribution of categorical features
        for column in categorical_columns:
            plt.figure(figsize=(10, 6))
            
            # Countplot for categorical data
            sns.countplot(y=self.data[column], order=self.data[column].value_counts().index)
            plt.title(f"Distribution of {column}", fontsize=14)
            plt.ylabel("Count", fontsize=12)
            plt.xlabel(column, fontsize=12)
            # plt.grid(axis="x", linestyle="--", alpha=0.6)
            plt.xticks(rotation=45)
            
            plt.show()
        
    def correlation_matrix(self):
        """"
        This function gives the correlation matrix of the data
        """
        print("***Correlation Matrix of the data***")
        
        # Select numerical columns
        numerical_columns = self.data.select_dtypes(include=["float64", "int64"]).columns

        # Compute correlation matrix
        correlation_matrix = self.data[numerical_columns].corr()

        # Visualize the correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            correlation_matrix,
            annot=True,       # Display correlation values
            fmt=".2f",        # Format values to 2 decimal points
            cmap="coolwarm",  # Color map
            cbar=True,        # Display color bar
            square=True,      # Ensure square cells
            linewidths=0.5    # Add lines between cells
        )
        plt.title("Correlation Matrix of Numerical Features", fontsize=16)
        plt.show()
    
    def missing_values(self):
        """"
        This function gives the missing values in the data
        """
        print("***Missing Values in the data***")
        missing_values = self.data.isnull().sum()
        missing_percentage = (missing_values / len(self.data)) * 100
        # Create a summary table for missing data
        missing_summary = pd.DataFrame({
            "Feature": self.data.columns,
            "Missing Value": missing_values,
            "Missing Percentage (%)": missing_percentage
        })
        missing_summary = missing_summary[missing_summary['Missing Value'] > 0].sort_values(by="Missing Percentage (%)", ascending=False)

        # Display missing summary
        print("Missing Values Summary:")
        print(missing_summary)

    
    def outliers_data(self):
        """"
        This function gives the outliers in the data
        """
        print("***Outliers in the data***")
        # Select numerical columns for outlier analysis
        numerical_columns = self.data.select_dtypes(include=["float64", "int64"]).columns

        # Plot box plots for each numerical feature
        plt.figure(figsize=(16, 8))

        for i, col in enumerate(numerical_columns, 1):
            plt.subplot(2, (len(numerical_columns) + 1) // 2, i)
            sns.boxplot(data=self.data, x=col, color="lightblue")
            plt.title(f"Box Plot of {col}", fontsize=12)
            plt.tight_layout()
            plt.show()