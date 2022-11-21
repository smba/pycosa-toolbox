import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

def read_measurements(path: str):
    
    root = ET.parse(path).getroot()
    
    features = set([])
    performance = []
    configs = []
    for row in root.findall(".//row"):
        for data in row:
            col = data.attrib["columname"]
            val = data.text.strip()
            if col == "Performance":
                perf = float(val)
                performance.append(perf)
                
            elif col == "Configuration":
                options = val.split(",")
                for option in options:
                    features.add(option)
                configs.append(options)
            
    # make data frame
    features = sorted(list(features))
    data = pd.DataFrame(
        np.zeros(shape=(len(performance), len(features) + 1)),
        columns = features + ["performance"],
    )
    
    for i, config in enumerate(configs):
        data.loc[i, config] = 1
    
    for feature in features:
        data[feature] = data[feature].astype("bool")
    
    data["performance"] = performance
    
    return data