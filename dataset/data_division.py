import pandas as pd
import numpy as np


def get_corr(dataset: pd.DataFrame):
    """

    :param dataset: pandas Dataframe
    :return: a new Dataframe containing datas from CORR dataset
    """

    #corr_vectors = []
    corr_vectors = dataset.loc[dataset['Dataset'] != 'UCLA']
    corr_vectors = corr_vectors.loc[dataset['Dataset'] != 'COBRE']
    #for i in range(len(dataset)):
    #    if dataset.Dataset[1] == "UCLA":
    #        break
    #    elif dataset.Dataset[1] == "COBRE":
    #        break
    #    else:
    #        corr_vectors.append(dataset.iloc[i])
    return corr_vectors


def get_ucla(dataset: pd.DataFrame):
    """

    :param dataset: dataset: pandas Dataframe
    :return: a new Dataframe containing datas from UCLA dataset
    """
    ucla_vectors = dataset.loc[dataset['Dataset'] == 'UCLA']
    return ucla_vectors


def get_cobre(dataset: pd.DataFrame):
    """

    :param dataset: dataset: pandas Dataframe
    :return: a new Dataframe containing datas from COBRE dataset
    """
    cobre_vectors = dataset.loc[dataset['Dataset'] == 'COBRE']
    return cobre_vectors


def get_healthy(dataset: pd.DataFrame):
    """

    :param dataset: dataset: pandas Dataframe
    :return: a list of vectors from healthy patients
    """
    healthy_vectors = dataset.loc[dataset['Disease'] == 0]
    return healthy_vectors


def get_training_data(dataset: pd.DataFrame):
    training_data = get_corr(dataset)
    ucla_data = get_ucla(dataset)
    ucla_healthy = get_healthy(ucla_data)
    training_data = pd.concat([training_data, ucla_healthy[0:58]])
    # only keep the functional connectivity correlation values
    training_data = training_data.iloc[:, 7:98]
    return training_data.to_numpy()


def get_testing_data(dataset: pd.DataFrame):
    """

    :param dataset: a pandas Dataframe
    :return: a numpy vector of patients with various illnesses
    """
    testing_data = get_ucla(dataset)
    testing_data = testing_data.iloc[59:, 7:98]
    return testing_data.to_numpy()


def get_testing_healthy_data(dataset: pd.DataFrame):
    """

    :param dataset: pandas Dataframe
    :return: a numpy vector of healthy patients that the network never saw
    """
    testing_data = get_ucla(dataset)
    testing_data = testing_data.iloc[:58, 7:98]
    return testing_data.to_numpy()


def get_testing_scz_data(dataset: pd.DataFrame):
    scz_data = dataset.loc[dataset['Disease'] == 1]
    scz_data = scz_data.iloc[:, 7:98]
    return scz_data.to_numpy()


def get_bd_data(dataset: pd.DataFrame):
    bd_data = dataset.loc[dataset['Disease'] == 2]
    bd_data = bd_data.iloc[:, 7:98]
    return bd_data.to_numpy()


def get_adhd_data(dataset: pd.DataFrame):
    adhd_data = dataset.loc[dataset['Disease'] == 3]
    adhd_data = adhd_data.iloc[:, 7:98]
    return adhd_data.to_numpy()

