import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting

get_images = False


def get_timeseries(dataset):
    """
    This function accepts a float with the number of subjects, and a string with the name of the folder
    where the subjects are

    Returns a dictionary containing in each key the timeseries of each subject.
    For each subject it contains an array with shape (number of timepoints, number of regions)

    Goal: To get the timeseries of each subject for each region
    """
    mydict = {0: []}
    list = next(os.walk("{dataset}".format(dataset=dataset)))[2]
    number_of_subjects = len(list)
    for n in range(number_of_subjects):
        mydict[n] = pd.read_csv("{dataset}/{subject}".format(dataset=dataset, subject=list[n]), delimiter="\s+",
                                header=None)
    return mydict


def get_matrix(mydict, z_score=True):
    """
    This function accepts a dictionary (a subject per key), containing in each key the timeseries extracted for each
    functional brain network (FBN).
    Each key is a numpy array with shape (number of timepoints, number of regions).
    You can decide to apply Z-score or not.
    Returns a all subjects matrix with vectorized_correlations in each row, and a all subjects matrix with a correlation
    matrix per subject within that matrix
    """

    all_correlation_matrix = []
    all_vector = []
    z_all_correlation_matrix = []
    z_all_vector = []

    for s in range(len(mydict)):  # run a cycle for each subject
        timeseries = mydict[s]
        # Create a MATRIX of (Number_of_subjects x Number_of_FBN x Number_of_FBN)
        correlation_matrix = timeseries.corr()  # create a correlation matrix using pearson's correlation coefficient
        correlation_matrix = correlation_matrix.to_numpy()

        all_correlation_matrix.append(
            correlation_matrix)  # append each correlation matrix to a list, which will contain all subject's
        # correlation matrices

        # Create a VECTOR ((no diagonal, no lower matrix))
        subject_vector = []  # after the following cycle, this list will contain the non-redundant correlations between
        # all FBN.
        for i in range(len(correlation_matrix)):
            for j in range(len(correlation_matrix[0])):
                if i > j:
                    subject_vector.append(correlation_matrix[i, j])  # appends a correlation between two different FBN

        all_vector.append(subject_vector)  # list with information of correlations of all subjects

        # Apply Z-SCORE
        if z_score:
            rows_cm = len(correlation_matrix)
            columns_cm = len(correlation_matrix[0])
            columns_vc = len(subject_vector)

            z_correlation_matrix = np.zeros((rows_cm, columns_cm))
            z_subject_vector = np.zeros((columns_vc))

            for i in range(columns_vc):
                z_subject_vector[i] = 0.5 * (np.log(1 + subject_vector[i]) - np.log(1 - subject_vector[i]))
            z_all_vector.append(z_subject_vector)  # list with correlation information of all subjects
            np.fill_diagonal(correlation_matrix, 0)
            for i in range(rows_cm):
                for j in range(columns_cm):
                    z_correlation_matrix[i, j] = 0.5 * (
                                np.log(1 + correlation_matrix[i, j]) - np.log(1 - correlation_matrix[i, j]))
            z_all_correlation_matrix.append(z_correlation_matrix)

    if not z_score:
        return np.array(all_correlation_matrix), np.array(all_vector)
    else:
        return np.array(z_all_correlation_matrix), np.array(z_all_vector)


def plot_subject(subjectTS, title='Timerseries', xlabel='Time Points', ylabel='Amplitude'):
    """
    This function accepts a numpy array with shape (timepoints, functional brain networks) of a specific subject

    Goal: save a image of the plot of the subject timeseries
    """

    plt.plot(subjectTS, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()
    plt.savefig("subject_timeseries.png", dpi=100)


def get_figures(folder, matrix_correlations, vector_correlations):
    """This function accepts a string 'folder' where it saves the images, and two numpy arrays of correlation matrices and vectors of all subjects
    It saves an image for each subject (number of FBN x number of FBN), and an image of all subjects (number of FBN x Number of Subjects)
    """

    for s in range(len(matrix_correlations)):
        fig = plotting.plot_matrix(matrix_correlations[s], tri='lower', figure=(10, 8), vmax=1, vmin=-1)
        plt.savefig("{folder}/subject{s}.png".format(s=s, folder=folder), dpi=100)

        plt.close('all')
        print('finished %s of %s' % (s + 1, len(matrix_correlations)))

    im = plt.imshow(vector_correlations, aspect='auto', cmap='RdBu')
    plt.colorbar()
    im.set_clim(-1, 1)  # can also use min and max
    plt.title('Subjects Matrix')
    plt.xlabel('features')
    plt.ylabel('subjects')
    plt.savefig("{folder}/all_subjects.png".format(folder=folder), dpi=500)
    plt.close('all')


def reshape_to_lower_matrix(row_correlation, lower=True):
    """
    This function accepts a row array with vectorized correlations
    It reshapes that row into a lower matrix
    Returns a lower matrix of that subject

    """
    if len(row_correlation) == 1:
        row_correlation = row_correlation.reshape(len(row_correlation[0]), )
    n = int(np.sqrt(len(row_correlation) * 2)) + 1
    mask = np.tri(n, dtype=bool, k=-1)
    lower_matrix = np.zeros((n, n), dtype=float)
    lower_matrix[mask] = row_correlation

    if not lower:
        lower_matrix = lower_matrix + lower_matrix.T

    return lower_matrix


def reshape_all(all_vector, lower=True):
    "This function reshapes several subjects' vector into lower matrices"
    n_subjects = len(all_vector)
    all_matrix = []
    for i in range(n_subjects):
        matrix = reshape_to_lower_matrix(all_vector[i], lower)
        all_matrix.append(matrix)
    all_matrix = np.array(all_matrix)

    return all_matrix


data_information = pd.read_csv("C:\\Users\\morga\\Documents\\ETS\\Projet\\data\\data_id_14FBN.csv", sep=";")
timeseries = get_timeseries("C:\\Users\\morga\\Documents\\ETS\\Projet\\data\\dr_fbn_timeseries")
all_correlation_matrices, all_vectors = get_matrix(timeseries, z_score=True)
all_data_timeseries = pd.concat([data_information, pd.DataFrame(all_vectors)], axis=1)
vectors_to_lower_matrix = reshape_all(all_vectors)

if get_images:
    plot_subject(all_vectors[0])
    get_figures("C:\\Users\\morga\\Documents\\ETS\\Projet\\data\\train_maps", all_correlation_matrices, all_vectors)

