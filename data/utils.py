import os
import csv
import numpy as np
import scipy.io as sio
from nilearn import connectome
from scipy.spatial import distance
import pandas as pd

root_folder = '.'
data_folder = os.path.join(root_folder, 'ABIDE_pcp/cpac/filt_noglobal')


ta = ["M1","M2","M3","M4","M5","M6","L","I","C","IC"]
index = {"M1":7,"M2":8,"M3":9,"M4":10,"M5":11,"M6":12,"L":4,"I":5,"C":3,"IC":6}



def fetch_filenames(subject_IDs, file_type):

    """
        subject_list : list of short subject IDs in string format
        file_type    : must be one of the available file types

    returns:

        filenames    : list of filetypes (same length as subject_list)
    """

    import glob

    # Specify file mappings for the possible file types
    filemapping = {'func_preproc': '_func_preproc.nii.gz',
                   'rois_ho': '_rois_ho.1D'}

    # The list to be filled
    filenames = []

    # Fill list with requested file paths
    for i in range(len(subject_IDs)):
        os.chdir(r"")  # os.path.join(data_folder, subject_IDs[i]))
        try:
            filenames.append(glob.glob('*' + subject_IDs[i] + filemapping[file_type])[0])
        except IndexError:
            # Return N/A if subject ID is not found
            filenames.append('N/A')

    return filenames


# Get timeseries arrays for list of subjects
def get_timeseries(subject_list, atlas_name):
    """
        subject_list : list of short subject IDs in string format
        atlas_name   : the atlas based on which the timeseries are generated e.g. aal, cc200

    returns:
        time_series  : list of timeseries arrays, each of shape (timepoints x regions)
    """

    timeseries = []
    for i in range(len(subject_list)): #subject_list
        subject_folder = os.path.join(r"", subject_list[i])
        ro_file = [f for f in os.listdir(subject_folder) if f.endswith('_rois_' + atlas_name + '.1D')]
        fl = os.path.join(subject_folder, ro_file[0])
        print("Reading timeseries file %s" %fl)
        timeseries.append(np.loadtxt(fl, skiprows=0))

    return timeseries


# Compute connectivity matrices
def subject_connectivity(timeseries, subject, atlas_name, kind, save=True, save_path=r""):
    """
        timeseries   : timeseries table for subject (timepoints x regions)
        subject      : the subject ID
        atlas_name   : name of the parcellation atlas used
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        save         : save the connectivity matrix to a file
        save_path    : specify path to save the matrix if different from subject folder

    returns:
        connectivity : connectivity matrix (regions x regions)
    """

    print("Estimating %s matrix for subject %s" % (kind, subject))

    if kind in ['tangent', 'partial correlation', 'correlation']:
        conn_measure = connectome.ConnectivityMeasure(kind=kind)
        connectivity = conn_measure.fit_transform([timeseries])[0]

    if save:
        subject_file = os.path.join(save_path, subject,
                                    subject + '_' + atlas_name + '_' + kind.replace(' ', '_') + '.mat')
        sio.savemat(subject_file, {'connectivity': connectivity})

    return connectivity

# read excel
def readexcel1(path):
    raw_data = pd.read_excel(path, header=0)
    data = raw_data.values
    info = data[:, 0]
    with open("subject_ids1.txt", "a") as f:
        for i in range(1, len(info)):
            f.write(str(info[i]) + "\n")
    return data


import random
# Get the list of subject IDs
def get_ids(num_subjects=None, path = ""):
    """
    return:
        subject_IDs    : list of all subject IDs
    """
    subject_IDs = np.genfromtxt(os.path.join(path), dtype=str)
    if num_subjects is not None:
        subject_IDs = subject_IDs[:num_subjects]
    index = [i for i in range(len(subject_IDs))]
    random.shuffle(index)
    subject_IDs = subject_IDs[index]
    read_text(path=path, subject_IDs=subject_IDs)
    return subject_IDs
def read_text(path, subject_IDs):
    filename = os.path.join(os.path.dirname(path), "subject_ids123.txt")
    with open(filename, 'w', encoding='utf-8') as file:
        for item in subject_IDs:
            # 将列表中的每个元素写入文件，并在每个元素后添加换行符
            file.write(item + '\n')


def get_ids_pro(num_subjects=None):
    """

    return:
        subject_IDs    : list of all subject IDs

    """
    path = ""
    subject_IDs = np.genfromtxt(os.path.join(path), dtype=str)
    if num_subjects is not None:
        subject_IDs = subject_IDs[:num_subjects]
    index = [i for i in range(len(subject_IDs))]
    return subject_IDs



# Get phenotype values for a list of subjects
def get_subject_score(subject, score):
    #print(score)
    #print(subject)
    subject_list = []
    for i in ta:
        subject_list.append(subject + '-' + i)
    #print(subject_list)


    if score == "ID":
        return subject_list

    scores_dict = {}

    phenotype1 = os.path.join(phenotype, subject + ".csv")
    subject_list = []
    for i in ta:
        subject_list.append(subject + '-' + i)
    with open(phenotype1) as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            if row['ID'] in subject_list:

                scores_dict[row['ID']] = row[score]

    return scores_dict

def get_subject_lable(subject_list,score):
    path = ""
    scores_dict = {}
    with open(path) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row["id"] in subject_list:
                scores_dict[row["id"]] = row[score]
    num_nodes = len(scores_dict)
    y = np.zeros([num_nodes])
    for i in range(num_nodes):
        y[i] = abs(int(scores_dict[subject_list[i]]))  # 数组，标签
    return y

def get_subject_lable_pro(subject_list,score):
    path = f""
    if score == "Left":
        score = "Right"
    scores_dict = {}
    with open(path) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row["subjects_ids"] in subject_list:
                scores_dict[row["subjects_ids"]] = row[score]
    num_nodes = len(scores_dict)
    y = np.zeros([num_nodes])
    for i in range(num_nodes):
        y[i] = abs(int(scores_dict[subject_list[i]]))  # 数组，标签
    return y




def site_percentage(train_ind, perc, subject_list):
    """
        train_ind    : indices of the training samples
        perc         : percentage of training set used
        subject_list : list of subject IDs

    return:
        labeled_indices      : indices of the subset of training samples
    """

    train_list = subject_list[train_ind]
    sites = get_subject_score(train_list, score='SITE_ID')
    unique = np.unique(list(sites.values())).tolist()
    site = np.array([unique.index(sites[train_list[x]]) for x in range(len(train_list))])

    labeled_indices = []

    for i in np.unique(site):
        id_in_site = np.argwhere(site == i).flatten()

        num_nodes = len(id_in_site)
        labeled_num = int(round(perc * num_nodes))
        labeled_indices.extend(train_ind[id_in_site[:labeled_num]])
    return labeled_indices

def get_networks(subjests_list,SORCE):
    """
        subject_list : list of subject IDs
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        atlas_name   : name of the parcellation atlas used
        variable     : variable name in the .mat file that has been used to save the precomputed networks


    return:
        matrix      : feature matrix of connectivity networks (num_subjects x network_size)
    """
    dicts = {"M1":0,"M2":1,"M3":2,"M4":3,"M5":4,"M6":5,"L":6,"I":7,"C":8,"IC":9}
    path = ""
    sub = []
    dis = []
    hea = []
    n = len(subjests_list)
    for i in range(n):
        subject = subjests_list[i] + ".mat"
        fl = os.path.join(path, subject)
        matrix = sio.loadmat(fl)["feature"]
        dis.append(matrix[dicts[SORCE]])
        hea.append(matrix[dicts[SORCE] + 10] + matrix[dicts[SORCE]])
    dis = np.array(dis)
    hea = np.array(hea)
    return dis, hea

def create_affinity_graph_from_scores(train_ids, labels,subjects):
    num_nodes = len(labels)
    graph = np.ones((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(i + 1,num_nodes):
            if not judege(subjects[i],subjects[j]):
                graph[i][j] = 0.5
                graph[j][i] = 0.5
            if i in train_ids and j in train_ids:
                if labels[i] != labels[j]:
                    graph[i][j] = 0
                    graph[j][i] = 0
    return graph
def judege(subject1,subject2):
    if "Kmd" in subject1 and "Kmd" in subject2:
        return 1
    elif "Kmd" not in subject1 and "Kmd" not in subject2:
        return 1
    else:
        return 0

def get_static_affinity_adj(features):
    distv = distance.pdist(features, metric='correlation') 
    dist = distance.squareform(distv)  
    sigma = np.mean(dist)
    feature_sim = np.exp(- dist ** 2 / (2 * sigma ** 2))
    adj = feature_sim
    return adj
    

if __name__ == '__main__':
    A = get_ids()
    get_networks(A)