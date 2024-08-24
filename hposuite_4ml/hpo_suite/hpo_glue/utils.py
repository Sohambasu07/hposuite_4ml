from typing import Dict, Any
from pathlib import Path
import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
import argparse
import os
import pandas as pd
import numpy as np
import pprint
from itertools import cycle
# import random
from collections import defaultdict, OrderedDict
import operator
import yaml

yaml_file_path = Path(__file__).parent /'style.yaml'

style_dict={}

def get_color_dict(style_dict):
    with open(yaml_file_path, 'r') as file:
        style_dict.update(yaml.safe_load(file))

def get_marker_color_style(
        instance_name: str
) -> tuple[str, str, str | tuple]:
    """Get the marker and color for the instance"""

    matched_key = next((key for key in style_dict if key in instance_name), None)
    print("Matched Key:", matched_key)
    default_marker = 'o'  # Default marker style
    default_color = '#000000'  # Default color (black)
    default_linestyle = '-'  # Default line style
    if matched_key in style_dict:
        linestyle, color, marker = style_dict.get(
        instance_name,
        (default_marker, default_color, default_linestyle)
    )
        print("Instance Name:", instance_name)
        print("Marker:", marker)
        print("Color:", color)
        print("Linestyle:", linestyle)
    return (linestyle, color, marker)


def plot_results(
        report: Dict[str, Any],
        budget_type: str,
        budget: int,
        objective: str,
        minimize: bool,
        save_dir: Path,
        obj_func_name: str,
        cut_off: float,
        dataset_name: str
):
    """Plot the results for the optimizers on the given Objective Function"""
    get_color_dict(style_dict)
    marker_list = ["o", "X", "^", "H", ">", "^", "p", "P", "*", "h", "<", "s", "x", "+", "D", "d", "|", "_"]
    colors_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    color_sel = cycle(colors_list)
    opt_rank_dict = OrderedDict()
    # random.shuffle(marker_list)
    marker_sel = cycle(marker_list)
    # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # colors_mean = cycle(colors)
    optimizers = list(report.keys())
    print(f"Plotting performance of Optimizers on {obj_func_name}")
    plt.figure(figsize=(20, 20))
    optim_res_dict = dict()
    for instance in optimizers:
        optim_res_dict[instance] = dict()
        seed_cost_dict = dict()
        for seed in report[instance].keys():
            results = report[instance][seed]["results"]
            cost_list = results[results["objectives"][0]].values.astype(np.float64)
            if budget_type == "fidelity_budget":
                budget_list = report[instance][seed]["results"]["fidelity"].values.astype(np.float64)
                if np.isnan(budget_list[0]):
                    budget_list = np.cumsum(np.repeat(float(results["max_budget"][0]), len(budget_list)))
                else:
                    budget_list = np.cumsum(budget_list)
            elif budget_type =="n_trials":
                budget_list = np.arange(1, budget + 1)
            elif budget_type == "time_budget":
                budget_list = report[instance][seed]["results"]["fit_time"].values.astype(np.float64)
                budget_list = np.cumsum(budget_list)
            seed_cost_dict[seed] = pd.Series(cost_list, index = budget_list)
            seed_cost_dict[seed].loc[seed_cost_dict[seed] <= cut_off] = cut_off
        seed_cost_df = pd.DataFrame(seed_cost_dict)
        seed_cost_df.ffill(axis = 0, inplace = True)
        seed_cost_df.dropna(axis = 0, inplace = True)
        means = pd.Series(seed_cost_df.mean(axis=1), name = f"means_{instance}")
        std = pd.Series(seed_cost_df.std(axis=1), name = f"std_{instance}")
        optim_res_dict[instance]["means"] = means
        optim_res_dict[instance]["std"] = std
        if minimize:
            means = means.cummin()
        else:
            means = means.cummax()
        means = means.drop_duplicates()
        std = std.loc[means.index]
        means[budget] = means.iloc[-1]
        std[budget] = std.iloc[-1]

        opt_rank_dict[instance] = means[budget]
        print("instance:",instance)
        if instance in style_dict:
            style, color, marker = get_marker_color_style(instance)
        else:
            style = "solid"
            color = next(color_sel)
            marker = next(marker_sel)
        print(instance)


        plt.step(
            means.index, 
            means, 
            where = 'post', 
            label = instance,
            marker = marker,
            # marker = next(marker),
            markersize = 10,
            markerfacecolor = '#ffffff',
            # markeredgecolor = color,
            # markeredgewidth = 2,
            color = color,
            linewidth = 2,
            linestyle = style
        )
        plt.fill_between(
            means.index, 
            means - std, 
            means + std, 
            alpha = 0.2,
            step = 'post',
            color = color,
            edgecolor = color,
            linewidth = 1            
        )
    
    # Printing the Optimizers by Ranking
    opt_rank_dict = OrderedDict(sorted(opt_rank_dict.items(), key=operator.itemgetter(1), reverse= (not minimize)))
    print("Optimizers by Ranking")
    pprint.pprint(opt_rank_dict)


    plt.xlabel(f"{budget_type}")
    plt.ylabel(f"{objective}")
    # plt.yscale("asinh")
    if budget_type == "fidelity_budget" or budget_type == "time_budget":
        plt.xscale("log")
        plt.grid(True, axis='x', alpha = 0.5, which='both')
    plt.title(f"Performance of Optimizers on {obj_func_name}")
    if len(optimizers) == 1:
        plt.title(f"Performance of {optimizers[0]} on {obj_func_name}")
    box = plt.gca().get_position()
    plt.gca().set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    plt.savefig(save_dir / f"{obj_func_name}_{budget_type}_performance.png", dpi = 300)
    return pd.Series(opt_rank_dict)


def agg_data(
        exp_dir: str,
        budget_type: str | None = None,
        budget: int | None = None,
        budget_max: bool = False
):
    """Aggregate the data from the run directory for plotting"""

    avg_rank_list = []
    dataset_names = []
    for runs in os.listdir(exp_dir):
        for obj_func_dir in os.listdir(exp_dir/ runs):
            print(obj_func_dir)
            match obj_func_dir:
                case "y_prop":
                    cut_off = 0.015
                case "bike_sharing":
                    cut_off = 0.90
                case "brazilian_houses":
                    cut_off = 0.95
                case "exam_dataset":
                    cut_off = 0.80
                case _:
                    continue

            df_agg = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
            opts = os.listdir(exp_dir / runs / obj_func_dir)
            objective = None
            minimize = True
            max_cumulative_budget = 0

            for i, opt_dir in enumerate(opts):
                for seed in os.listdir(exp_dir/ runs / obj_func_dir / opt_dir):
                    files = os.listdir(exp_dir/ runs / obj_func_dir / opt_dir / seed)
                    for file in files:
                        res_df = pd.read_parquet(exp_dir/ runs / obj_func_dir / opt_dir / seed /file)
                        if res_df.empty:
                            continue
                        objective = res_df[['objectives']].iloc[0].values[0]
                        budget_type_df = res_df[['budget_type']].iloc[0].values[0]
                        budget_df = res_df[['budget']].iloc[0].values[0]

                        if budget_type is None or budget_type_df == budget_type:
                            budget_type = budget_type_df

                        if budget is None or budget_df == budget:
                            budget = budget_df

                        if budget_max:
                            if budget_type == "fidelity_budget":
                                budget = res_df['fidelity'].cumsum().max().astype(np.float64)
                            elif budget_type == "time_budget":
                                budget = res_df['runtime_cost'].iloc[0].astype(np.float64)
                            if budget > max_cumulative_budget:
                                max_cumulative_budget = budget

                        minimize = res_df[['minimize']].iloc[0].values[0]
                        res_df = res_df[[objective, 'fidelity', 'max_budget', 'objectives', 'fit_time']]
                        instance = (file.split(".parquet")[0]).split(obj_func_dir)[-1][1:]
                        instance = instance[:-1] if instance[-1] == "_" else instance
                        df_agg[instance][int(seed)] = {
                            "results": res_df
                        }

            budget = max_cumulative_budget if budget_max else budget
            avg_rank_list.append(
                plot_results
                (
                    report = df_agg,
                    budget_type = budget_type,
                    budget = budget,
                    objective = objective,
                    minimize = minimize,
                    save_dir = exp_dir / runs / "plots",
                    obj_func_name = obj_func_dir,
                    cut_off = cut_off,
                    dataset_name = obj_func_dir
                )
            )
            dataset_names.append(obj_func_dir)
    
    avg_rank_df = pd.DataFrame(avg_rank_list, index = dataset_names)
    avg_rank_df.loc['mean'] = avg_rank_df.mean()

    if 'y_prop' in dataset_names:
        avg_rank_df.loc['y_prop_rank'] = avg_rank_df.loc['y_prop'].rank(ascending=False)
    if 'bike_sharing' in dataset_names:
        avg_rank_df.loc['bike_sharing_rank'] = avg_rank_df.loc['bike_sharing'].rank(ascending=False)
    if 'brazilian_houses' in dataset_names:
        avg_rank_df.loc['brazilian_houses_rank'] = avg_rank_df.loc['brazilian_houses'].rank(ascending=False)
    if 'y_prop' in dataset_names and 'bike_sharing' in dataset_names and 'brazilian_houses' in dataset_names:
        avg_rank_df.loc['mean_rank'] = avg_rank_df.loc[['y_prop_rank', 'bike_sharing_rank', 'brazilian_houses_rank']].mean()
        avg_rank_df.loc['avg_rank'] = avg_rank_df.loc['mean'].rank(ascending=False)
        avg_rank_df = avg_rank_df.sort_values(by='avg_rank', axis=1)
    print(avg_rank_df)
    
    avg_rank_df.to_csv(exp_dir / "avg_rank.csv")
                               


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plotting Incumbents after GLUE Experiments")

    parser.add_argument("--root_dir",
                        type=Path,
                        help="Location of the root directory",
                        default=Path("./"))

    parser.add_argument("--results_dir",
                        type=str,
                        help="Location of the results directory",
                        default="./results")
    
    parser.add_argument("--exp_dir",
                        type=str,
                        help="Location of the Experiment directory",
                        default=None)
    
    parser.add_argument("--save_dir",
                        type=str,
                        help="Directory to save the plots",
                        default="plots")
    
    parser.add_argument("--budget_type",
                        type=str,
                        help="Type of Budget for posthoc analysis",
                        default=None)
    
    parser.add_argument("--budget",
                        type=int,
                        help="Budget for posthoc analysis",
                        default=None)
    
    parser.add_argument("--budget_max",
                        action="store_true",
                        help="Whether to set the budget to the maximum budget"
                        "ever seen in the results."
                        "NOTE: Only active if budget is not specified"
                        "and budget_type is specified."
                        "Doesn't work with n_trials budget type"
                        )

    args = parser.parse_args()

    if args.exp_dir is None:
        raise ValueError("Experiment directory not specified")
    
    if args.budget_max and args.budget is not None:
        raise ValueError("Cannot set both budget_max and budget")
    
    if args.budget_max and args.budget_type is None:
        raise ValueError("Cannot set budget_max without specifying budget_type")
    
    if args.budget_max and args.budget_type == "n_trials":
        raise ValueError("Cannot set budget_max with n_trials budget type")
    
    
    exp_dir = args.root_dir / args.results_dir / args.exp_dir
    
    agg_data(
        exp_dir = exp_dir,
        budget_type = args.budget_type,
        budget = args.budget,
        budget_max = args.budget_max
    )


