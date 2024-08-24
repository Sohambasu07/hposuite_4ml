from __future__ import annotations

# from fanova import fANOVA
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt
import os
from pathlib import Path
import argparse

ROOT_DIR = Path(__file__).absolute().resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
GLOBAL_SEED = 0


fid_enabled = {
    "Hyperband": True,
    "BOHB": True,
    "DEHB": True
}

def generate_seeds(
    num_seeds: 5,
) -> np.ndarray:
    """Generate a set of seeds using a Global Seed."""
    _rng = np.random.default_rng(GLOBAL_SEED)
    seeds = _rng.integers(0, 2 ** 30 - 1, size=num_seeds)
    return seeds


def filter_df(
    df: pd.DataFrame,
    fidelity_enabled: bool,
    filename: str
) -> pd.DataFrame:


    # fidelity_key = df[['fidelity_type']].iloc[0].values[0]

    if not fidelity_enabled:
        df = df.drop(columns=['fidelity'])

    # print(df.columns)
    # print(df.head())

    # Drop common irrelevant columns
    hp_df = df.drop(columns = [
        'config_id',
        # fidelity_key,
        'r2',
        'fit_time',
        # 'fidelity_type',
        'max_budget', 
        'objectives', 
        'minimize',
        'budget_type', 
        'budget',
        'seed',
        'runtime_cost',
        'optimizer_name',
        'optimizer_hyperparameters',
        'objective_function'
        ])
    
    if 'activation' in hp_df.columns:
        hp_df = hp_df.drop(columns = ['activation'])
    if 'solver' in hp_df.columns:
        hp_df = hp_df.drop(columns = ['solver'])

    if 'MLP' in filename:
        hp_df = hp_df.drop(columns = ['learning_rate', 'max_iter'])
    if 'fidelity_type' in hp_df.columns:
        hp_df = hp_df.drop(columns = ['fidelity_type'])

    # print(hp_df.columns)

     # Drop device_type column
    if 'device' in hp_df.columns:
         hp_df = hp_df.drop(columns = ['device'])
    elif 'device_type' in hp_df.columns:
        hp_df = hp_df.drop(columns = ['device_type'])
    
    # Drop verbosity column
    if 'verbosity' in hp_df.columns:
        hp_df = hp_df.drop(columns = ['verbosity'])
    elif 'verbose' in hp_df.columns:
        hp_df = hp_df.drop(columns = ['verbose'])

    # Drop random_state column
    if 'random_state' in hp_df.columns:
        hp_df = hp_df.drop(columns = ['random_state'])
    elif 'seed' in hp_df.columns:
        hp_df = hp_df.drop(columns = ['seed'])
    elif 'random_seed' in hp_df.columns:
        hp_df = hp_df.drop(columns = ['random_seed'])

    
    print(filename)
    
    
    # Filter for models: LGBM

    if 'LightGBM' in filename:

        if 'bootstrap' in hp_df.columns:
            hp_df = hp_df.drop(columns = ['bootstrap'])
        if 'max_features' in hp_df.columns:
            hp_df = hp_df.drop(columns = ['max_features'])
        if 'min_samples_leaf' in hp_df.columns:
            hp_df = hp_df.drop(columns = ['min_samples_leaf'])
        if 'max_leaf_nodes' in hp_df.columns:
            hp_df = hp_df.drop(columns = ['max_leaf_nodes'])
        if 'min_impurity_decrease' in hp_df.columns:
            hp_df = hp_df.drop(columns = ['min_impurity_decrease'])
        if 'min_samples_leaf' in hp_df.columns:
            hp_df = hp_df.drop(columns = ['min_samples_leaf'])
        if 'min_samples_split' in hp_df.columns:
            hp_df = hp_df.drop(columns = ['min_samples_split'])
        if 'min_weight_fraction_leaf' in hp_df.columns:
            hp_df = hp_df.drop(columns = ['min_weight_fraction_leaf'])
        if 'colsample_bytree' in hp_df.columns:
            hp_df = hp_df.drop(columns = ['colsample_bytree'])
        if 'max_leaves' in hp_df.columns:
            hp_df = hp_df.drop(columns = ['max_leaves'])

    # Filter for models: XGBoost

    if 'XGBoost' in filename:

        if 'feature_fraction' in hp_df.columns:
            hp_df = hp_df.drop(columns = ['feature_fraction'])
        if 'min_child_samples' in hp_df.columns:
            hp_df = hp_df.drop(columns = ['min_child_samples'])
        if 'min_data_in_leaf' in hp_df.columns:
            hp_df = hp_df.drop(columns = ['min_data_in_leaf'])
        if 'num_leaves' in hp_df.columns:
            hp_df = hp_df.drop(columns = ['num_leaves'])
        if 'bootstrap' in hp_df.columns:
            hp_df = hp_df.drop(columns = ['bootstrap'])
        if 'max_features' in hp_df.columns:
            hp_df = hp_df.drop(columns = ['max_features'])
        if 'max_leaf_nodes' in hp_df.columns:
            hp_df = hp_df.drop(columns = ['max_leaf_nodes'])
        if 'min_impurity_decrease' in hp_df.columns:
            hp_df = hp_df.drop(columns = ['min_impurity_decrease'])
        if 'min_samples_leaf' in hp_df.columns:
            hp_df = hp_df.drop(columns = ['min_samples_leaf'])
        if 'min_samples_split' in hp_df.columns:
            hp_df = hp_df.drop(columns = ['min_samples_split'])
        if 'min_weight_fraction_leaf' in hp_df.columns:
            hp_df = hp_df.drop(columns = ['min_weight_fraction_leaf'])



    return hp_df

# def get_importance(
#     df: pd.DataFrame,
#     objective: str,
#     fidelity_enabled: bool,
#     filepath: str,
#     filename: str
# ) -> None:
#     """
#     Get the importance of each hyperparameter in the dataframe
#     using fANOVA
#     """
    
#     hp_df = filter_df(df, fidelity_enabled, filename)
#     print(hp_df)
#     hp_df = hp_df.astype(np.float64)
#     hp_df.to_csv("./hp_df.csv", index=False, header=False)
#     y_df = df[objective]
#     y_df.to_csv("./y_df.csv", index=False, header=False)

#     # X = hp_df.to_numpy()
#     # y = df[objective].to_numpy()

#     X = np.loadtxt("./hp_df.csv", delimiter=",")
#     y = np.loadtxt("./y_df.csv", delimiter=",")

#     imp = fANOVA(X, y, cutoffs=(-np.inf, np.inf))
#     importance = imp.quantify_importance()
#     print(importance)


def fit_rf(
    df: pd.DataFrame,
    objective: str,
    fidelity_enabled: bool,
    filepath: str,
    imp_df_dict: dict[str, list],
    seed: int = 0
) -> None:
    """
    Fit an sklearn RandomForestRegressor model to the dataframe
    """
    
    hp_df = filter_df(df, fidelity_enabled, filepath)

    X = hp_df.to_numpy()
    y = df[objective].to_numpy()

    rf = RandomForestRegressor(n_estimators=100, random_state=seed)
    rf.fit(X, y)
    
    importances = rf.feature_importances_
    for idx, val in enumerate(importances):
        hp = hp_df.columns[idx]
        if hp not in imp_df_dict:
            imp_df_dict[hp] = []
        imp_df_dict[hp].append(val)


def get_importance_rf(
    df: pd.DataFrame,
    objective: str,
    fidelity_enabled: bool,
    filepath: str,
    filename: str
) -> None:
    """
    Get the importance of each hyperparameter in the dataframe
    using sklearn RandomForestRegressor
    """
    imp_df_dict = {}
    for seed in generate_seeds(11):
        fit_rf(
            df, 
            objective, 
            fidelity_enabled,
            filename,
            imp_df_dict,
            seed
        )

    imp_mean_dict = {}
    for idx, val in imp_df_dict.items():
        imp_mean_dict[idx] = np.mean(val)
    importances = pd.DataFrame.from_dict(imp_mean_dict, orient='index', columns=['importance'])
    print(importances)
    plot_importance(
        importances,
        filepath,
        filename
    )

    

def plot_importance(
    importances: pd.DataFrame,
    filepath: str,
    filename: str
) -> None:
    """
    Plot the importance of each hyperparameter
    """

    filename = filename.split("report_")[1]

    plt.figure(figsize=(10, 10), dpi=300)
    plt.bar(importances.index, importances['importance'], color='orange')
    plt.subplots_adjust(bottom=0.30)
    plt.xlabel("Features")
    plt.xticks(rotation=90)
    plt.ylabel("Increasing Importance")
    plt.title(f"Feature Importance of hyperparameters {filename}")
    save_dir = f"./feature_importances/{filepath}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(f"{save_dir}/{filename}.png")

    

def load_parquet(
    exp_dir: str | Path,
    imp_rf: bool,
    fidelity_enabled: bool = False
) -> tuple[pd.DataFrame, str, bool]:
    for runs in os.listdir(exp_dir):
        if ".csv" in runs:
            continue
        for obj_func_dir in os.listdir(exp_dir/ runs):
            print(obj_func_dir)
            match obj_func_dir:
                case "y_prop":
                    pass
                case "bike_sharing":
                    pass
                case "brazilian_houses":
                    pass
                case "exam_dataset":
                    pass
                case _:
                    continue
            opts = os.listdir(exp_dir / runs / obj_func_dir)
            objective = None
            minimize = True
            for i, opt_dir in enumerate(opts):
                if "Hyperband" in opt_dir or "BOHB" in opt_dir or "DEHB" in opt_dir:
                    fidelity_enabled = True
                for seed in os.listdir(exp_dir/ runs / obj_func_dir / opt_dir):
                    files = os.listdir(exp_dir/ runs / obj_func_dir / opt_dir / seed)
                    for file in files:
                        res_df = pd.read_parquet(exp_dir/ runs / obj_func_dir / opt_dir / seed /file)
                        if res_df.empty:
                            continue
                        objective = res_df[['objectives']].iloc[0].values[0]
                        minimize = res_df[['minimize']].iloc[0].values[0]

                        exp_dir_str = str(exp_dir).split("/")[-1]
                        filepath = f"{exp_dir_str}/runs/{obj_func_dir}/{opt_dir}/{seed}"
                        print(filepath)
                        filename = file.split(".parquet")[0]
                        print(filename)


                        if imp_rf:
                            get_importance_rf(
                                res_df, 
                                objective,
                                fidelity_enabled,
                                filepath,
                                filename
                        )
                        # else:
                        #     get_importance(
                        #         res_df, 
                        #         objective,
                        #         fidelity_enabled,
                        #         filepath,
                        #         filename
                        #     )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        default="./",
        help="Root directory",
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        help="Experiment directory",
        required=True
    )
    parser.add_argument(
        "--imp_rf",
        action="store_true",
        help="Use Random Forest for importance"
    )
    parser.add_argument(
        "--fidelity_enabled",
        action="store_true",
        help="Use fidelity"
    )
    args = parser.parse_args()

    exp_dir = RESULTS_DIR / args.exp_dir
    # save_dir = RESULTS_DIR / args.exp_dir / args.seed_dir

    load_parquet(
        exp_dir=exp_dir,
        imp_rf=args.imp_rf,
        fidelity_enabled=args.fidelity_enabled
    )

