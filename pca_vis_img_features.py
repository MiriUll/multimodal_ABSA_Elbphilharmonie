import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D
import pandas as pd

##############################################################
# Change these lines to apply on your custom datasets
##############################################################
features = 'scraping/test_data_evaluation/r-mac_features.npy'
dataset = 'scraping/test_dataset.csv'
landmark_colors = {'holstentor': 'b', 'rathaus': 'tab:purple', 'sternschanze': 'g', 'michaelis': 'tab:olive',
                   'elbphilharmonie': 'tab:orange', 'random': 'tab:brown'}
class_list = ['holstentor', 'rathaus', 'sternschanze', 'michaelis', 'elbphilharmonie', 'random']
save = False
##############################################################
# End of hardcoded parameters
##############################################################


def vis_pca_features(pca_result: np.array, query_id: int = None, answer_id: np.array = None, save_name: str = None,
                     title: str = ''):
    """
    Plot the downprojected features vectors in 2D
    :param pca_result: The 2D projected version of the vectors
    :param query_id: Index of image that was used as query
    :param answer_id: Array of indices of retrieved images
    :param save_name: Filename to save the plot
    :param title: Title of the plot
    :return:
    """
    fig, ax = plt.subplots(1)
    for l in class_list:
        idxs = test_data[test_data.landmark == l].index.values  # find all images of that class
        ax.scatter(pca_result[idxs, 0], pca_result[idxs, 1], label=l,
                   color=landmark_colors[l])
        ax.scatter(np.average(pca_result[idxs, 0]), np.average(pca_result[idxs, 1]), label=l, marker='*',
                   color=landmark_colors[l])  # plot the class average as star marker
    if query_id is not None:
        ax.scatter(pca_result[query_id, 0], pca_result[query_id, 1], color='r', marker='x', label='query')
    if answer_id is not None:
        ax.scatter(pca_result[answer_id[1:], 0], pca_result[answer_id[1:], 1], color='r', marker=2, label='answer')

    # summarize legend entries of same landmark
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]

    # insert legend dummy for average star
    avg_patch = Line2D([0], [0], marker='*', color='grey', label='Class average', markersize=9, linestyle='None')
    handles.insert(-1, avg_patch)
    plt.legend(handles, np.insert(labels, -1, 'Class average'), loc='best')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title(title)
    if save_name:
        plt.savefig(save_name + '.pdf')
    plt.show()


print("Loading features..")
features = np.load(features)
test_data = pd.read_csv(dataset)
class_order = test_data.landmark.values
print("Projecting features..")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features)
savepath = 'vis_pca_features.pdf' if save else None

vis_pca_features(pca_result, save_name=savepath)
