from sklearn.neighbors import NearestNeighbors
from matplotlib import image as mpimg
from matplotlib import pyplot as plt
import random
import pandas as pd
import numpy as np

##############################################################
# Change these lines to apply on your custom datasets
##############################################################
features = 'scraping/test_data_evaluation/r-mac_features.npy'
dataset = 'scraping/test_dataset.csv'
img_path = 'scraping/test_data/'
##############################################################
# End of hardcoded parameters
##############################################################


def retrieve_images(query_id: int, query_dataset: np.array, search_dataset: np.array, num_neigbors: int = 4,
                    class_order: list = None):
    """
    Fit a nearest neighbor to retrieve the num_neigbors closest images
    :param query_id: Index of the query image, -1 for a random image
    :param query_dataset: Feature vectors of query data
    :param search_dataset: Feature vectors of search data, can be similar or different to query data
    :param num_neigbors: The number of images that should be retrieved
    :param class_order: labels of the images
    :return: the index of the query image
    :return: the indices of retrieved images
    :return: distances of retrieved images
    :return: labels of retrieved images
    """
    if query_id == -1:  # if query_id is -1, select a random image from the dataset
        query_id = random.randrange(query_dataset.shape[0])
    query = query_dataset[query_id, :].reshape(-1, 1).T

    knn = NearestNeighbors(n_neighbors=num_neigbors, p=2)
    knn.fit(search_dataset)

    answer_dist, answer_id = knn.kneighbors(query)  # retrieve distances and ids of closest images

    # !!the classes are not ordered -> indices do not correspond to them in answer_id !!!
    classes = [class_order[i] for i in answer_id[0]] if class_order is not None else None
    return query_id, answer_id[0], answer_dist[0], classes


def plot_image(img_id: int, position: int) -> None:
    """
    Plot retrieved image
    :param img_id: Index in dataset
    :param position: position on plot
    :return: None
    """
    plt.subplot(3, 3, position)
    plt.imshow(mpimg.imread(img_path + str(test_data.ID.values[img_id]) + '.jpg'))
    plt.axis('off')


print("Loading features..")
features = np.load(features)
test_data = pd.read_csv(dataset)
class_order = test_data.landmark.values
print("Retrieve closest images..")
query_id, answer_id, answer_dist, classes = retrieve_images(-1, search_dataset=features, query_dataset=features,
                                                            num_neigbors=7, class_order=class_order)
fig = plt.figure()
plot_image(query_id, 1)
plt.title('Query image')
for j, i in enumerate(answer_id):
    if answer_dist[j] == 0:
        continue
    plot_image(img_id=i, position=3 + j)
    plt.title(f"Distance: {round(answer_dist[j], 2)}")
plt.show()
