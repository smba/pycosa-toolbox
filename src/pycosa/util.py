import pandas as pd
import networkx as nx
import logging
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


def remove_multicollinearity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes sources of perfect multicollinearity from a data frame of binary
    configuration options (courtesy of Johannes):

    1. columns with a single value (mandatory or dead options), and
    2. for each alternative group (set of mutually exclusive options of which
       exactly one is selected in every row), one (the alphabetically first)
       option of the group.

    Non-binary columns are kept unchanged and not considered for (2).
    """
    df = df.copy()

    # remove columns with identical values (dead features or mandatory features)
    nunique = df.nunique()
    df = df.drop(columns=nunique[nunique <= 1].index)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    binary = [c for c in df.columns if set(pd.unique(df[c])) <= {0, 1, True, False}]
    if len(binary) < 2:
        return df

    X = df[binary].to_numpy(dtype=float)
    # co-occurrence counts: cooc[i, j] == 0 <=> options i and j are never
    # selected together, i.e., they are mutually exclusive in the data
    cooc = X.T @ X

    G = nx.Graph()
    rows, cols = np.nonzero(np.triu(cooc == 0, k=1))
    G.add_edges_from((binary[i], binary[j]) for i, j in zip(rows, cols))

    cliques_remaining = True
    while cliques_remaining:
        cliques_remaining = False
        for clique in nx.find_cliques(G):
            # check if exactly one col is 1 in each row
            if (df[clique].astype(float).sum(axis=1) == 1.0).all():
                df = df.drop(columns=sorted(clique)[0])
                G.remove_nodes_from(clique)
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

    maximums = df[list(replace_map.keys())].idxmax(axis=1)
    maximums = maximums.replace(replace_map)
    df[new_option] = maximums

    if drop_old:
        df = df.drop(columns=list(replace_map.keys()))

    return df


def construct_categorical_variable(
    df: pd.DataFrame, categorical_column: str, drop_old: bool = True
) -> pd.DataFrame:
    """
    Introduces a dummy variable (one-hot encoding) for each value of a
    categorical variable.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame containing the independent variables
    categorical_column : str
        Column to split.
    drop_old : bool, optional
        Should the to-be-replaced columns be dropped? The default is True.

    Returns
    -------
    df : pd.DataFrame
        Transformed data frame with new columns.

    """
    values = df[categorical_column]
    categories = sorted(pd.unique(values), key=str)

    expanded = pd.DataFrame(
        {
            "{}_{}".format(categorical_column, c): (values == c).to_numpy()
            for c in categories
        },
        index=df.index,
    )

    df = pd.concat([df, expanded], axis="columns")

    if drop_old:
        df = df.drop(columns=[categorical_column])

    return df


def construct_interaction_terms(
    df: pd.DataFrame, categorical_column: str, sep: str = "__", drop_old: bool = True
) -> pd.DataFrame:
    """
    For each value ``v`` of a categorical column and each other column ``c``,
    adds an interaction column ``v<sep>c`` that equals ``c`` in rows where the
    categorical column is ``v`` and 0 otherwise (NaN is treated as 0).
    """
    unique_values = pd.unique(df[categorical_column])
    interacting_columns = [c for c in df.columns if c != categorical_column]

    new_columns = {}
    for uvalue in unique_values:
        mask = (df[categorical_column] == uvalue).to_numpy()
        for icolumn in interacting_columns:
            values = np.nan_to_num(df[icolumn].to_numpy(dtype=float))
            new_columns["{}{}{}".format(uvalue, sep, icolumn)] = np.where(
                mask, values, 0.0
            )

    df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis="columns")

    if drop_old:
        df = df.drop(columns=interacting_columns + [categorical_column])
    return df
