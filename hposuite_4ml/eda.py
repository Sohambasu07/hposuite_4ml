import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
from data import Dataset
from dataprocesses import correlation_analysis
from dataprocesses import plot_histogram_density_qq_imp_features
import numpy as np
from scipy.stats import entropy, wasserstein_distance, skew, kurtosis
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


FILE = Path(__file__).absolute().resolve()
DATADIR = FILE.parent.parent / "data"
print(DATADIR)
fold = 1
output_file = 'meta_learning_output.txt'

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


# Principal Component Analysis
def _PCA(
    df: pd.DataFrame, 
    task: str,
)-> tuple:
    pca_num_components = 2
    # scaler = StandardScaler()
    # df_standardized = scaler.fit_transform(df)
    pca = PCA(n_components=pca_num_components)
    # reduced_data = pca.fit_transform(df_standardized)
    reduced_data = pca.fit_transform(df)
    pca_df = pd.DataFrame(reduced_data,columns=['pca1','pca2'])
    exp_var = pca.explained_variance_ratio_.sum()
    print(f"Variance explained by PCA: {exp_var}")
    plt.figure(figsize=(10,6))
    plt.title(f"Principal Components in {task} with Explained Variance: {exp_var}")
    sns.scatterplot(x="pca1", y="pca2", data = pca_df, palette = "deep", legend = 'full')
    plt.show()
    
    return pca, pca_df,reduced_data

# PCA feature importance
def _PCA_feat_importance(
    pca: PCA,
    df: pd.DataFrame,
    task : str,
)-> list:
    abs_eigenvals = np.abs(pca.components_)
    most_important_indices = np.argsort(abs_eigenvals, axis=1)[:, ::-1]
    most_important_features = []
    for i, indices in enumerate(most_important_indices):
        print(f"Principal Component {i+1}:")
        # for idx in indices[0:]: #Take the top 2 features for each pc
        idx=indices[0]
        most_important_features.append(df.columns[idx])
        print(f"Feature {idx}: {abs_eigenvals[i, idx]}")
        most_important_features = list(set(most_important_features))  # Remove duplicates
    return most_important_features

# PCA eigenvectors
def _PCA_eigenvectors(
    pca: PCA,
    reduced_pca: pd.DataFrame,
    df: pd.DataFrame,
    task: str,
)-> None:
    abs_eigenvals = np.abs(pca.components_)
    most_imp_indices = np.argsort(abs_eigenvals, axis=1)[:, ::-1]
    
    # score = reduced_pca.iloc[:, 0:2].values
    coeff = np.transpose(pca.components_[0:2, :])
    labels = df.columns
    # xs = score[:,0]
    # ys = score[:,1]
    # scalex = 1.0/(xs.max() - xs.min())
    # scaley = 1.0/(ys.max() - ys.min())
    
    plt.figure(figsize=(10, 6))
    for i in range(2):  # Only display the 2 most important eigenvectors
        idx = most_imp_indices[i, 0]
        plt.arrow(0, 0, coeff[idx, 0], coeff[idx, 1], color='b', alpha=0.5, width=0.005)
        plt.text(coeff[idx, 0] * 1.15, coeff[idx, 1] * 1.15, labels[idx], color='black', ha='center', va='center')
        # plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, labels[i], color='black', ha='center', va='center')
    
    # plt.xlim(-0.5,2)
    # plt.ylim(-0.5,1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()
    plt.title(f"PCA Eigenvectors for {task}")
    # plt.legend()
    plt.show()


# Compute Wasserstein Distance
def compute_wasserstein_distance(data1, data2):
    distances = []
    for i in range(data1.shape[1]):
        distances.append(wasserstein_distance(data1[:, i], data2[:, i]))
    return np.mean(distances)


def extract_meta_features_from_pca(pca_data):
    meta_features = {
        'pca_n_samples': pca_data.shape[0],
        'pca_n_components': pca_data.shape[1],
        'pca_mean': np.mean(pca_data, axis=0).tolist(),
        'pca_std': np.std(pca_data, axis=0).tolist(),
        'pca_var': np.var(pca_data, axis=0).tolist()
    }
    return meta_features

#  Extract simple meta-features
def extract_simple_meta_features(X,task):
    return {
        'task': task,
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
    }

# Extract statistical meta-features
def extract_statistical_meta_features(X):
    return {
        'mean': np.mean(X, axis=0).tolist(),
        'std': np.std(X, axis=0).tolist(),
        'var': np.var(X, axis=0).tolist(),
        'skew': skew(X, axis=0).tolist(),
        'kurtosis': kurtosis(X, axis=0).tolist(),
    }

# Extract information-theoretic meta-features
def extract_information_theoretic_meta_features(y):
    class_probabilities = np.bincount(y) / len(y)
    return {
        'class_entropy': entropy(class_probabilities),
    }



# Combine all meta-features
def extract_meta_features(X, y,task):
    meta_features = {}
    meta_features.update(extract_simple_meta_features(X,task))
    meta_features.update(extract_statistical_meta_features(X))
    meta_features.update(extract_information_theoretic_meta_features(y))
    return meta_features



def PC_analysis(datasets)-> None:
    all_important_features = set()
    all_meta_features ={}
    pca_results = {}
    index=0
    for task, dataset in datasets.items():
        logger.info(f"\nPrincipal Component Analysis for {task}")
        numerical_features = dataset['X_train'].select_dtypes(include=['int64', 'float64']).columns
        d_num = dataset['X_train'][numerical_features]
        meta_features=extract_meta_features(d_num, dataset['Y_train'],task)
        pca, reduced_pca, reduced_data = _PCA(d_num,task)
        pca_results[task] = reduced_data
        important_features =_PCA_feat_importance(pca, d_num,task)
        all_important_features.update(important_features)
        _PCA_eigenvectors(pca,reduced_pca, d_num, task)
        plot_histogram_density_qq_imp_features(d_num, important_features, task)
        meta_features.update(extract_meta_features_from_pca(reduced_pca))
        all_meta_features[index]= meta_features
        index+=1
    
    all_meta_feature_df=pd.DataFrame(all_meta_features)   
    with open(output_file, 'a') as f:
        f.write(all_meta_feature_df.to_string(index=False)) 
    # all_meta_feature_df.to_csv('metafeatures.csv', index=False)
    
    # Compare datasets
    tasks_list = list(pca_results.keys())
    for i in range(len(tasks_list)):
        for j in range(i + 1, len(tasks_list)):
            task1, task2 = tasks_list[i], tasks_list[j]
            data1, data2 = pca_results[task1], pca_results[task2]
            
            wasserstein_dist = compute_wasserstein_distance(data1, data2)
            
            print(f"Comparison between {task1} and {task2}:")
            print(f"Wasserstein Distance: {wasserstein_dist}")
            print()

        

if __name__ == "__main__":
    
    # Tasks (datasets) to load
    tasks = ['y_prop', 'bike_sharing', 'brazilian_houses','exam_dataset']
    
    # Load the datasets
    datasets = load_datasets(tasks)
    
    # Perform basic information analysis
    # print("Basic Information:")
    # basic_info(datasets)
    
    # Perform summary statistics
    # print("Summary Statistics:")
    # summary_statistics(datasets)
    
    # Check for missing values
    # print("Missing Values:")
    # missing_values(datasets)
    
    # Perform univariate analysis
    # print("Univariate Analysis:")
    # univariate_analysis(datasets)
    
    # Assuming 'target' is the target variable for bivariate analysis
    # target_variable = 'target'  # Replace with the actual target variable name
    # print("Bivariate Analysis:")
    #bivariate_analysis(datasets)
    
    
    
    # Perform correlation analysis
    print("Correlation Analysis:")
    filtered_datasets=correlation_analysis(datasets,0.7)
    
    # Perform correlation analysis
    print("PCA Analysis:")
    PC_analysis(filtered_datasets)
    
