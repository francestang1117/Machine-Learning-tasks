import numpy as np

def cal_entroy(data):

    features, counts = np.unique(data, return_counts=True)

    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(features))])  

    # for i in range(len(features)):
    #     pb = counts[i]/np.sum(counts)
    #     entroy += -1 * np.sum(pb*np.log2(pb))
    
    return entropy


def information_gain(dataset, split_name, target="LEVEL"):


    entroy_before = cal_entroy(dataset[target])

    values, counts = np.unique(dataset[split_name], return_counts=True)

        # prob = counts[i]/np.sum(counts)
        # weight_entropy += prob*entroy(dataset.where(dataset[split_name]==values[i]).dropna()[target])
    weighted_entropy = np.sum([(counts[i]/np.sum(counts))*cal_entroy(dataset.where(dataset[split_name]==values[i]).dropna()[target]) for i in range(len(values))])
    
    gain = entroy_before - weighted_entropy
    return gain




