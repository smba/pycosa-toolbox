import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np


def read_measurements(path, METRIC):

    root = ET.parse(path).getroot()

    features = set([])
    performance = []
    configs = []
    for row in root.findall(".//row"):
        for data in row:
            col = data.attrib["columname"]
            val = data.text.strip()
            if col == METRIC:
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
