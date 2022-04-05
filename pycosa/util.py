import pandas as pd
import networkx as nx
import logging
import math
import numpy as np

from statsmodels.stats.outliers_influence import variance_inflation_factor


def get_vif(df: pd.DataFrame, threshold: float = 5.0):
    """
    Calculates the variance inflation factor (VIF) for each feature column. A VIF
    value greater than a specific threshold (default: 5.0) can be considered problematic since
    this column is likely correlated with other ones, i.e., we cannot properly
    infer the effect of this column.
    """

    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns

    # calculate the variance variance inflation factor
    vif_data["vif"] = [
        variance_inflation_factor(df.values, i) for i in range(len(df.columns))
    ]

    # drop a warning, if VIF exceeds threshold or is infinite
    critical_features = vif_data[
        (vif_data["vif"] == float("inf")) | (vif_data["vif"] > threshold)
    ]

    for row in critical_features.iterrows():
        feature = row[1]["feature"]
        vif = round(row[1]["vif"], 2)
        logging.warning(
            "Feature «{}» exceeds threshold (VIF = {})!".format(feature, vif)
        )

    return vif_data


def remove_multicollinearity(df: pd.DataFrame):

    # courtesy by johannes

    # remove columns with identical values (dead features or mandatory features)
    nunique = df.nunique()
    mandatory_or_dead = nunique[nunique == 1].index.values

    df = df.drop(columns=mandatory_or_dead)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df_configs = df
    alternative_ft = []
    alternative_ft_names = []
    group_candidates = {}

    for i, col in enumerate(df_configs.columns):
        filter_on = df_configs[col] == 1
        filter_off = df_configs[col] == 0
        group_candidates[col] = []
        for other_col in df_configs.columns:
            if other_col != col:
                values_if_col_on = df_configs[filter_on][other_col].unique()
                if len(values_if_col_on) == 1 and values_if_col_on[0] == 0:
                    # other feature is always off if col feature is on
                    group_candidates[col].append(other_col)

    G = nx.Graph()
    for ft, alternative_candidates in group_candidates.items():
        for candidate in alternative_candidates:
            if ft in group_candidates[candidate]:
                G.add_edge(ft, candidate)

    cliques_remaining = True
    while cliques_remaining:
        cliques_remaining = False
        cliques = nx.find_cliques(G)
        for clique in cliques:
            # check if exactly one col is 1 in each row
            sums_per_row = df_configs[clique].sum(axis=1).unique()
            if len(sums_per_row) == 1 and sums_per_row[0] == 1.0:
                delete_ft = sorted(clique)[0]
                alternative_ft_names.append(delete_ft)
                df_configs.drop(delete_ft, inplace=True, axis=1)
                for c in clique:
                    G.remove_node(c)
                cliques_remaining = True
                break

    return df


def reconstruct_categorical_variable(
    df: pd.DataFrame, replace_map: dict, new_option: str, drop_old: bool = True
):
    """
    Reconstructs a numerical option that has been discretized.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame containing discretized numerical features.
    replace_map : dict
        Mapping of column names to option value of column $new_option.
    new_option : str
        Name of the newly created column.
    drop_old : bool, optional
        Should the to-be-replaced columns be dropped? The default is True.

    Returns
    -------
    df : pd.DataFrame
        Transformed data frame with new  column.
    """

    maximums = df[replace_map.keys()].idxmax(axis=1)
    maximums = maximums.replace(replace_map)
    df[new_option] = maximums

    if drop_old:
        df = df.drop(columns=replace_map.keys())

    return df


def construct_categorical_variable(
    df: pd.DataFrame, categorical_column: str, drop_old: bool = True
):
    """
    Introduces a dummy variable (one-hot encoding) for each value of a
    categorical variable.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame containing the independent variables
    column : str
        Column to split.
    drop_old : bool, optional
        Should the to-be-replaced columns be dropped? The default is True.

    Returns
    -------
    df : pd.DataFrame
        Transformed data frame with new columns.

    """

    # get set of unique values
    unique_values = df[categorical_column].unique()

    # create a new column in df for each unique value
    for uvalue in unique_values:

        index = df.index[df[categorical_column] == uvalue]
        new_column = np.zeros(shape=(df.shape[0],))
        new_column[index] = 1

        column_name = "{}_{}".format(categorical_column, uvalue)
        df[column_name] = new_column

    if drop_old:
        df = df.drop(columns=[categorical_column])

    return df


def construct_interaction_terms(
    df: pd.DataFrame, categorical_column: str, sep: str = "__", drop_old: bool = True
):

    # get set of unique values
    unique_values = df[categorical_column].unique()

    interacting_columns = df.drop(columns=[categorical_column]).columns

    for uvalue in unique_values:
        for icolumn in interacting_columns:
            index = df.index[df[categorical_column] == uvalue]
            new_column = np.zeros(shape=(df.shape[0],))
            new_column[index] = df.iloc[index][icolumn]

            # mask NaN as 0
            new_column = np.nan_to_num(new_column)
            column_name = "{}{}{}".format(uvalue, sep, icolumn)
            df[column_name] = new_column

    if drop_old:
        df = df.drop(columns=[i for i in interacting_columns] + [categorical_column])
    return df
