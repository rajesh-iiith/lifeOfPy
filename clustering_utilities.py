import numpy
import pandas as pd
from matplotlib import cm, pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation, AgglomerativeClustering, Birch
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min
from random import randint


def reduce_data(input_vector, number_of_components):
    pca = PCA(n_components=number_of_components)
    reduced_fv = pca.fit_transform(input_vector)
    return reduced_fv
    

def apply_clustering():
    normalized_feature_vectors = numpy.loadtxt('normalized_fv.csv', delimiter=',')
    feature_vectors = reduce_data(normalized_feature_vectors, 2)

    KMeans_clusterer = KMeans(n_clusters=10)
    DBScan_clusterer = DBSCAN(eps=0.05, min_samples=20, metric='euclidean')
    AP_clusterer = AffinityPropagation(damping=0.5, max_iter=200, convergence_iter=15, copy=True, preference=None, affinity='euclidean', verbose=False)
    Agglomerative_clusterer = AgglomerativeClustering(n_clusters=3, affinity='euclidean', connectivity=None, n_components=None, compute_full_tree='auto', linkage='ward')
    Birch_clusterer = Birch(threshold=0.05, branching_factor=50, n_clusters=3, compute_labels=True, copy=True)

    predicted_clusters = KMeans_clusterer.fit_predict(feature_vectors, y=None)
    unique_clusters, per_cluster_count = numpy.unique(predicted_clusters, return_counts=True)
    numpy.savetxt('predicted_clusters.csv', predicted_clusters.astype(int), fmt='%i')
    return feature_vectors, predicted_clusters, unique_clusters, per_cluster_count

    


def plot_clusters_3D(feature_vectors, predicted_clusters):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = [ cm.jet(x) for x in numpy.linspace(0.0, 1.0, len(unique_clusters)) ]
    for row, pp in enumerate(predicted_clusters):
        ax.scatter(feature_vectors[row][0], feature_vectors[row][1],feature_vectors[row][2], color = colors[predicted_clusters[row]])
    plt.show()

def plot_clusters_2D(feature_vectors, predicted_clusters):
    colors = [ cm.jet(x) for x in numpy.linspace(0.0, 1.0, len(unique_clusters)) ]
    for row, pp in enumerate(predicted_clusters):
        plt.scatter(feature_vectors[row][0], feature_vectors[row][1], color = colors[predicted_clusters[row]])
    plt.show()

def sort_by_cluster():
    identifier_df = pd.read_csv("identifiers.csv", header=None)
    original_index_df =  pd.DataFrame(data=identifier_df.index.values[0:])
    predicted_clusters_df = pd.read_csv("predicted_clusters.csv", header=None)
    both_df = pd.concat([identifier_df, predicted_clusters_df, original_index_df], axis=1)
    both_df.columns = ['identifier', 'cluster_id', 'original_index']
    both_df = both_df.sort_index(by=['cluster_id'])
    #both_df.to_csv('tags_and_clusters.csv', sep=',', index=False)
    return both_df

#create an array for each cluster (start_index,)
# unique_clusters, counts = numpy.unique(predicted_clusters, return_counts=True)
# for 1 to unique clusters: for start_index to start_index+count: do
def align_points_with_feaure_vectors():
    temp_list = []
    for index, row in tag_cluster_df.iterrows():
        temp_list.append(feature_vectors[row['original_index']])
    tag_cluster_df['feature_vectors'] = temp_list
    tag_cluster_df.to_csv('tags_and_clusters.csv', sep=',', index=False)
    return tag_cluster_df

def find_representative_points():
    for cluster_number in unique_clusters:
        current_df = tag_cluster_df.loc[tag_cluster_df['cluster_id'] == cluster_number]
        centroid = numpy.mean(current_df['feature_vectors'].as_matrix())
        min_distance = 1000000000.0
        representative_index = 0
        for index, row in current_df.iterrows():
            distance = numpy.linalg.norm( (row['feature_vectors']) - (centroid) )
            if  distance < min_distance:
                print "less"
                min_distance = distance
                representative_index = index
            print representative_index

        #print current_df.iloc[[representative_index]]

            


feature_vectors, predicted_clusters, unique_clusters, per_cluster_count = apply_clustering()
#numpy.savetxt('reduced_fv.csv', feature_vectors)
plot_clusters_2D(feature_vectors, predicted_clusters)
#tag_cluster_df = sort_by_cluster()
#tag_cluster_df = align_points_with_feaure_vectors()
#find_representative_points()


