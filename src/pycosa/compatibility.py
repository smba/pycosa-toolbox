"""
Backwards-compatible readers for third-party data formats.

Currently only SPL Conqueror measurement files (XML) are supported.
"""

import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


def read_measurements(path, METRIC):
    """
    Reads an SPL Conqueror measurement file into a data frame.

    The file must be XML with one ``<row>`` element per measurement.
    Each row's entries must contain a column named ``Configuration``
    (comma-separated list of the selected options) and one column
    named after the performance metric of interest. If a measurement
    consists of several comma-separated values, their median is used
    as the performance value.

    Parameters
    ----------
    path : str
        Path to the measurement file.
    METRIC : str
        Name of the column that holds the performance values.

    Returns
    -------
    pd.DataFrame
        One row per measurement, with one boolean column per option
        that occurs in any configuration, plus a column ``METRIC`` with
        the (median) performance values.
    """
    root = ET.parse(path).getroot()

    features = set([])
    performance = []
    configs = []
    for row in root.findall(".//row"):
        for data in row:
            col = data.attrib["columname"]
            val = data.text.strip()
            if col == METRIC:
                if "," in val:
                    values = [float(v) for v in val.split(",")]
                    perf = np.median(values)
                else:
                    perf = float(val)
                performance.append(perf)

            elif col == "Configuration":
                options = val.split(",")
                options = list(filter(lambda x: len(x) > 0, options))
                for option in options:
                    features.add(option)
                configs.append(options)

    res = []
    performance = np.array(performance)
    # make data frame
    features = sorted(list(features))
    for i, config in enumerate(configs):
        encoded = [1 if f in config else 0 for f in features]
        encoded += [performance[i]]

        res.append(encoded)

    res = np.vstack(res)
    df = pd.DataFrame(res, columns=features + [METRIC], index=np.arange(len(configs)))

    for f in features:
        df[f] = df[f].astype("bool")

    return df
