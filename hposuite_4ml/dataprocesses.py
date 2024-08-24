import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
from data import Dataset
import numpy as np
import scipy.stats as stats
from scipy.stats import shapiro, anderson, kstest
import logging


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


FILE = Path(__file__).absolute().resolve()
DATADIR = FILE.parent.parent / "data"
print(DATADIR)
fold = 1


# Load the datasets
def load_datasets(tasks):
    datasets = {}
    for task in tasks :
        dataset=Dataset.load(DATADIR, task, fold)
        datasets[task] = {
            'X_train': pd.DataFrame(dataset.X_train),
            'Y_train': pd.Series(dataset.y_train),
            'X_test': pd.DataFrame(dataset.X_test),
            'Y_test': pd.Series(dataset.y_test) if dataset.y_test is not None else None
        }
    return datasets


# Display basic information
def basic_info(datasets):
    for task, dataset in datasets.items():
        print(f"Dataset: {task}")
        print("Training data:")
        print(dataset['X_train'].info())
        print(dataset['Y_train'].describe())
        print("Test data:")
        print(dataset['X_test'].info())
        if dataset['Y_test'] is not None:
            print(dataset['Y_test'].describe())
        print("\n")


# Summary statistics
def summary_statistics(datasets):
    for task, dataset in datasets.items():
        print(f"Summary statistics for {task} - Training data:")
        print(dataset['X_train'].describe())
        print(dataset['Y_train'].describe())
        if dataset['Y_test'] is not None:
            print(f"Summary statistics for {task} - Test data:")
            print(dataset['X_test'].describe())
            print(dataset['Y_test'].describe())
        print("\n")


# Missing values
def missing_values(datasets):
    for task, dataset in datasets.items():
        print(f"Missing values in {task} - Training data:")
        print(dataset['X_train'].isnull().sum())
        print(dataset['Y_train'].isnull().sum())
        if dataset['Y_test'] is not None:
            print(f"Missing values in {task} - Test data:")
            print(dataset['X_test'].isnull().sum())
            print(dataset['Y_test'].isnull().sum())
        print("\n")


# Univariate analysis
def univariate_analysis(datasets):
    for task, dataset in datasets.items():
        numerical_features = dataset['X_train'].select_dtypes(include=['int64', 'float64']).columns
        for feature in numerical_features:
            plt.figure(figsize=(10, 6))
            sns.histplot(dataset['X_train'][feature], kde=True)
            plt.title(f'Distribution of {feature} in {task} - Training data')
            plt.show()


# Bivariate analysis
def bivariate_analysis(datasets):
    for task, dataset in datasets.items():
        numerical_features = dataset['X_train'].select_dtypes(include=['int64', 'float64']).columns
        fig, axes = plt.subplots(7,6,figsize=(12,8))
        axes = axes.flatten()
        # fig, axes = plt.subplots()
        i=0
        for feature in numerical_features:
            sns.scatterplot(x=dataset['X_train'][feature], y=dataset['Y_train'], ax=axes[i])
            axes[i].set_title(f'{feature}')
            i += 1
            # axes.plot(dataset.X_train[feature], dataset.y_train, label =f'{feature}' )
        plt.tight_layout()
        plt.show()


def extract_high_corr_features(dataset, threshold=0.7):
    corr_matrix = dataset['X_train'].corr()
    # Get pairs of features with correlation >= threshold
    high_corr_pairs = corr_matrix[(corr_matrix.abs() >= threshold) & (corr_matrix.abs() < 1.0)].stack().index.tolist()
    
    # Extract unique features from the pairs
    high_corr_features = set()
    for i, j in high_corr_pairs:
        high_corr_features.add(i)
        high_corr_features.add(j)
    print("high_corr_features : ")
    print(high_corr_features)
    # Filter the dataset to keep only these features
    filtered_dataset = {
        'X_train': dataset['X_train'][list(high_corr_features)],
        'Y_train': dataset['Y_train'],
        'X_test': dataset['X_test'][list(high_corr_features)] if dataset['X_test'] is not None else None,
        'Y_test': dataset['Y_test']
    }
    return filtered_dataset


def correlation_analysis(datasets, threshold=0.7):
    filtered_datasets = {}
    for task, dataset in datasets.items():
        filtered_datasets[task] = extract_high_corr_features(dataset, threshold)
        corr_matrix = filtered_datasets[task]['X_train'].corr()
       
        # Create a mask for the heatmap
        # Create a mask for values below the threshold
        # mask = corr_matrix.abs() < threshold
        
        # Create an annotation matrix with only the values above the threshold
        annotation_matrix = corr_matrix.map(lambda x: f"{x:.2f}" if abs(x) >= threshold else "")

        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=annotation_matrix, fmt="",cmap='coolwarm', vmin=-1, vmax=1)
        plt.title(f'Correlation Matrix for {task} - Training data (|correlation| >= {threshold})')
        plt.show()
    return filtered_datasets


def plot_histogram_density_qq(df, task):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    max_rows_per_plot = 5
    chunks = [numeric_columns[i:i + max_rows_per_plot] for i in range(0, len(numeric_columns), max_rows_per_plot)]
    for chunk_index, chunk in enumerate(chunks):
        # n_plots = len(chunk) * 3  # 3 plots (histogram, density, QQ) per column
        n_cols = 3  # Number of columns for subplots
        n_rows = len(chunk)  # One row per numeric column
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols,  6* n_rows))
        axes = np.atleast_2d(axes)  # Ensure axes is always 2D
        
        for i, column in enumerate(chunk):
            # Histogram and Density Plot
            sns.histplot(df[column], kde=True, ax=axes[i, 0])
            axes[i, 0].set_title(f'Hist&Den for {column}')
            
            # Density Plot
            sns.kdeplot(df[column], ax=axes[i, 1])
            axes[i, 1].set_title(f'Density for {column}')
            
            # QQ Plot
            stats.probplot(df[column], dist="norm", plot=axes[i, 2])
            axes[i, 2].get_lines()[1].set_color('r')  # Adjust the QQ line color
            axes[i, 2].set_title(f'QQ for {column}')
        
        # plt.tight_layout()
        plt.suptitle(f'Distribution Analysis for {task} (Part {chunk_index + 1})', fontsize=16)
        plt.show()

#  Plot histogram, density, and QQ plot for important features
def plot_histogram_density_qq_imp_features(df, important_features, task):
    max_rows_per_plot = 5
    chunks = [important_features[i:i + max_rows_per_plot] for i in range(0, len(important_features), max_rows_per_plot)]
    for chunk_index, chunk in enumerate(chunks):
        n_cols = 3  # Number of columns for subplots
        n_rows = len(chunk)  # One row per numeric column
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 6 * n_rows))
        axes = np.atleast_2d(axes)  # Ensure axes is always 2D
        
        for i, column in enumerate(chunk):
            # Histogram and Density Plot
            sns.histplot(df[column], kde=True, ax=axes[i, 0])
            axes[i, 0].set_title(f'Hist&Den for {column}')
            
            # Density Plot
            sns.kdeplot(df[column], ax=axes[i, 1])
            axes[i, 1].set_title(f'Density for {column}')
            
            # QQ Plot
            stats.probplot(df[column], dist="norm", plot=axes[i, 2])
            axes[i, 2].get_lines()[1].set_color('r')  # Adjust the QQ line color
            axes[i, 2].set_title(f'QQ for {column}')
        
        # plt.tight_layout()
        plt.suptitle(f'Distribution Analysis for {task}', fontsize=16)
        plt.show()

def plot_qq_plots(datasets):
    for task, dataset in datasets.items():
        for column in dataset['X_train'].select_dtypes(include=[np.number]).columns:
            plt.figure(figsize=(10, 6))
            stats.probplot(dataset['X_train'][column], dist="norm", plot=plt)
            plt.title(f'QQ Plot for {column} in {task}')
            plt.show()

def shapiro_test(datasets):
    for task, dataset in datasets.items():
        for column in dataset['X_train'].select_dtypes(include=[np.number]).columns:
            stat, p = shapiro(dataset['X_train'][column])
            print(f'Shapiro-Wilk Test for {column} in {task}: Statistics={stat}, p-value={p}')
            if p > 0.05:
                print(f'{column} in {task} looks Gaussian (fail to reject H0)')
            else:
                print(f'{column} in {task} does not look Gaussian (reject H0)')
            print()

def anderson_test(datasets):
    for task, dataset in datasets.items():
        for column in dataset['X_train'].select_dtypes(include=[np.number]).columns:
            result = anderson(dataset['X_train'][column])
            print(f'Anderson-Darling Test for {column} in {task}: Statistic={result.statistic}')
            for i in range(len(result.critical_values)):
                sl, cv = result.significance_level[i], result.critical_values[i]
                if result.statistic < cv:
                    print(f'{column} in {task}: {sl}%: {cv}, data looks Gaussian (fail to reject H0)')
                else:
                    print(f'{column} in {task}: {sl}%: {cv}, data does not look Gaussian (reject H0)')
            print()

def ks_test(datasets, dist='norm'):
    for task, dataset in datasets.items():
        for column in dataset['X_train'].select_dtypes(include=[np.number]).columns:
            stat, p = kstest(dataset['X_train'][column], dist)
            print(f'Kolmogorov-Smirnov Test for {column} in {task} against {dist} distribution: Statistics={stat}, p-value={p}')
            if p > 0.05:
                print(f'{column} in {task} looks like {dist} distribution (fail to reject H0)')
            else:
                print(f'{column} in {task} does not look like {dist} distribution (reject H0)')
            print()



        

if __name__ == "__main__":
    
    # Tasks (datasets) to load
    tasks = ['y_prop', 'bike_sharing', 'brazilian_houses','exam_dataset']
    
    # Load the datasets
    datasets = load_datasets(tasks)
    
    # Perform basic information analysis
    print("Basic Information:")
    basic_info(datasets)
    
    # Perform summary statistics
    print("Summary Statistics:")
    summary_statistics(datasets)
    
    # Check for missing values
    print("Missing Values:")
    missing_values(datasets)
    
    # Perform univariate analysis
    # print("Univariate Analysis:")
    # univariate_analysis(datasets)
    
    # Assuming 'target' is the target variable for bivariate analysis
    # target_variable = 'target'  # Replace with the actual target variable name
    # print("Bivariate Analysis:")
    # bivariate_analysis(datasets)
    
    # Perform correlation analysis
    print("Correlation Analysis:")
    # correlation_analysis(datasets)
    
    # Distribution Analysis
    print("Analyze Distribution:")
    # Perform distribution analysis
    # print("Histogram and Density Plots:")
    # plot_histogram_and_density(datasets)
    for task, dataset in datasets.items():
        plot_histogram_density_qq(dataset['X_train'], task)

    # print("QQ Plots:")
    # # plot_qq_plots(datasets)

    # print("Shapiro-Wilk Test:")
    # shapiro_test(datasets)

    # print("Anderson-Darling Test:")
    # anderson_test(datasets)

    # print("Kolmogorov-Smirnov Test against normal distribution:")
    # ks_test(datasets, 'norm')

    # print("Kolmogorov-Smirnov Test against exponential distribution:")
    # ks_test(datasets, 'expon')
