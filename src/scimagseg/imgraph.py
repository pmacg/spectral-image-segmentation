"""
Provides an object to represent an image as a graph, for spectral clustering.
"""
import math
from typing import Dict, List, Optional

import numpy
import scipy as sp
import sgtl
import skimage.filters
import skimage.measure
import skimage.transform
from matplotlib import image
from skimage.color import label2rgb


class DatasetGraph:
    """
    This base class represents some data as a graph, for clustering.
    """

    def __init__(self,
                 data_file=None,
                 gt_clusters_file=None,
                 graph_file=None,
                 num_data_points=None,
                 graph_type="knn10"):
        """
        Intiialise the dataset, optionally specifying a data file.
        """
        # Raw data can be a dense or sparse numpy or scipy vector
        self.raw_data = None
        self.gt_clusters: Optional[List[List[int]]] = None
        self.gt_labels: Optional[List[int]] = None
        self.graph: Optional[sgtl.Graph] = None
        self.num_data_points = num_data_points

        # We only load the data if there is no graph file supplied
        if graph_file is None:
            self.load_data(data_file)

        self.load_gt_clusters(gt_clusters_file)
        self.load_graph(graph_file, graph_type=graph_type)

    @staticmethod
    def set_default_files(data_file: Optional[str],
                          gt_clusters_file: Optional[str],
                          graph_file: Optional[str],
                          kwargs: Dict[str, Optional[str]]):
        """Set the default values of the data file and gt_clusters file in the keyword arguments."""
        if 'data_file' not in kwargs:
            kwargs['data_file'] = data_file
        if 'gt_clusters_file' not in kwargs:
            kwargs['gt_clusters_file'] = gt_clusters_file
        if 'graph_file' not in kwargs:
            kwargs['graph_file'] = graph_file
        return kwargs

    def load_data(self, data_file):
        """
        Load the raw data from the given data file.
        :param data_file:
        :return:
        """
        # TODO:: Implement this method correctly
        if data_file is not None:
            self.raw_data = None

    def load_gt_clusters(self, gt_clusters_file):
        """
        Load the ground truth clusters from the specified file.
        :param gt_clusters_file:
        :return:
        """
        # TODO:: Implement this method correctly
        if gt_clusters_file is not None:
            self.gt_clusters = None

    def load_graph(self, graph_file=None, graph_type="knn10"):
        """
        Load the graph from the specified file, or create a graph for the dataset if the file is not specified.

        The 'graph_type' parameter can be used to specify how the graph should be constructed if it is not otherwise
        specified. Valid formats are:
            'knn10' - the k nearest neighbour graph, with 10 neighbours. Can replace 10 as needed.

        :param graph_file: (optional) the file containing the edgelist of the graph to load.
        :param graph_type: (optional) if there is no edgelist, the type of graph to be constructed.
        """
        if graph_file is not None:
            self.graph = sgtl.graph.from_edgelist(
                graph_file, num_vertices=self.num_data_points)
        elif self.raw_data is not None:
            # Construct the graph using the method specified
            if graph_type[:3] == "knn":
                # We will construct the k-nearest neighbour graph
                k = int(graph_type[3:])
                self.graph = sgtl.graph.knn_graph(self.raw_data, k)
            elif graph_type[:3] == "rbf":
                self.graph = sgtl.graph.rbf_graph(self.raw_data, variance=20)
        else:
            # Nothing to do
            pass

    def construct_and_save_graph(self,
                                 graph_filename: str,
                                 graph_type="knn10"):
        """
        Construct a graph representing this dataset (using knn10 by default), and save it to the specified file.

        :param graph_filename: the name of the file to save the graph to
        :param graph_type: which type of graph to construct from the data
        """
        # Construct the graph
        self.load_graph(graph_file=None, graph_type=graph_type)

        # Save the graph
        sgtl.graph.to_edgelist(self.graph, graph_filename)

    def ground_truth(self) -> Optional[List[List[int]]]:
        """
        Return the ground truth clusters or None if they are unknown.
        :return:
        """
        return self.gt_clusters

    def __repr__(self):
        return self.__str__()


class ImageDatasetGraph(DatasetGraph):

    def __init__(self,
                 img_filename,
                 *args,
                 downsample_factor=None,
                 blur_variance=1,
                 **kwargs):
        """Construct a dataset graph from a single image.

        We will construct a graph from the image based on the gaussian radial basis function.

        :param img_filename: the filename containing the image
        :param downsample_factor: The factor by which do downsample the image. If none is supplied, then the factor is
                                  computed to make the total number of vertices in the dataset graph roughly equal to
                                  20,000.
        :param blur_variance: The variance of the gaussian blur applied to the downsampled image
        """
        self.image_filename = img_filename
        self.original_image_dimensions = []
        self.downsampled_image_dimensions = []
        self.downsample_factor = downsample_factor
        self.blur_variance = blur_variance
        self.image = None
        super(ImageDatasetGraph, self).__init__(*args,
                                                graph_type="rbf",
                                                **kwargs)

    def load_graph(self, *args, **kwargs):
        super(ImageDatasetGraph, self).load_graph(*args, **kwargs)

        # Add a grid to the graph, with weight 0.01.
        grid_graph_adj_mat = sp.sparse.lil_matrix(
            (self.num_data_points, self.num_data_points))
        for x in range(self.downsampled_image_dimensions[0]):
            for y in range(self.downsampled_image_dimensions[1]):
                this_data_point = x * self.downsampled_image_dimensions[1] + y

                # Add the four orthogonal edges
                if x > 0:
                    that_data_point = (
                        (x - 1) * self.downsampled_image_dimensions[1] + y)
                    grid_graph_adj_mat[this_data_point, that_data_point] = 0.1
                    grid_graph_adj_mat[that_data_point, this_data_point] = 0.1
                if y > 0:
                    that_data_point = (
                        x * self.downsampled_image_dimensions[1] + y - 1)
                    grid_graph_adj_mat[this_data_point, that_data_point] = 0.1
                    grid_graph_adj_mat[that_data_point, this_data_point] = 0.1
        grid_graph = sgtl.graph.Graph(grid_graph_adj_mat)
        self.graph += grid_graph

    def load_data(self, data_file):
        """
        Load the dataset from the image. Each pixel in the image is a data point. Each data point will have 5
        dimensions, namely the normalised 'rgb' values and the (x, y) coordinates of the pixel in the image.

        To reformat the data to be a manageable size, we downsample by a factor of 3.

        :param data_file:
        :return:
        """
        img = image.imread(self.image_filename)
        self.image = img
        self.original_image_dimensions = (img.shape[0], img.shape[1])

        # Compute the downsample factor if needed
        if self.downsample_factor is None:
            current_num_vertices = (self.original_image_dimensions[0] *
                                    self.original_image_dimensions[1])

            if current_num_vertices > 20000:
                self.downsample_factor = int(
                    math.sqrt(current_num_vertices / 20000))
            else:
                self.downsample_factor = 1

        # Do the downsampling here
        img_l1 = skimage.measure.block_reduce(img[:, :, 0],
                                              self.downsample_factor,
                                              func=numpy.mean)
        img_l2 = skimage.measure.block_reduce(img[:, :, 1],
                                              self.downsample_factor,
                                              func=numpy.mean)
        img_l3 = skimage.measure.block_reduce(img[:, :, 2],
                                              self.downsample_factor,
                                              func=numpy.mean)
        img = numpy.stack([img_l1, img_l2, img_l3], axis=2)
        self.downsampled_image_dimensions = (img.shape[0], img.shape[1])

        # Blur the image slightly
        img = skimage.filters.gaussian(img, sigma=self.blur_variance)

        # Extract the data points from the image
        self.num_data_points = img.shape[0] * img.shape[1]
        self.raw_data = []
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                self.raw_data.append(
                    [img[x, y, 0], img[x, y, 1], img[x, y, 2], x, y])
        self.raw_data = numpy.array(self.raw_data)

    def save_downsampled_image(self, filename):
        """
        Save the downsampled image to the given file.

        :param filename:
        """
        # Load and downsample the image
        img = image.imread(self.image_filename)
        img_l1 = skimage.measure.block_reduce(img[:, :, 0],
                                              self.downsample_factor,
                                              func=numpy.mean)
        img_l2 = skimage.measure.block_reduce(img[:, :, 1],
                                              self.downsample_factor,
                                              func=numpy.mean)
        img_l3 = skimage.measure.block_reduce(img[:, :, 2],
                                              self.downsample_factor,
                                              func=numpy.mean)
        img = numpy.stack([img_l1, img_l2, img_l3], axis=2)

        # Blur the image slightly
        img = skimage.filters.gaussian(img, sigma=self.blur_variance)

        # Save the image to the given file
        image.imsave(filename, img / 255)

    def downsampled_image(self):
        # Load and downsample the image
        img = image.imread(self.image_filename)
        img_l1 = skimage.measure.block_reduce(img[:, :, 0],
                                              self.downsample_factor,
                                              func=numpy.mean)
        img_l2 = skimage.measure.block_reduce(img[:, :, 1],
                                              self.downsample_factor,
                                              func=numpy.mean)
        img_l3 = skimage.measure.block_reduce(img[:, :, 2],
                                              self.downsample_factor,
                                              func=numpy.mean)
        img = numpy.stack([img_l1, img_l2, img_l3], axis=2)

        # Blur the image slightly
        img = skimage.filters.gaussian(img, sigma=self.blur_variance)

        return img

    def save_clustering(self, clusters, filename):
        """
        Save the segmentation given by the given clusters.
        """
        # Construct the labels rather than the list of clusters.
        pixel_labels = [0] * self.num_data_points
        for i, segment in enumerate(clusters):
            for pixel in segment:
                pixel_labels[pixel] = i

        # Construct the labelled image with the downsampled dimensions
        labelled_image = numpy.array(pixel_labels, dtype="int32")
        labelled_image = numpy.reshape(labelled_image,
                                       self.downsampled_image_dimensions) + 1

        # Scale up the segmentation by taking the appropriate tensor product
        labelled_image_upsample = (numpy.kron(
            labelled_image,
            numpy.ones((self.downsample_factor, self.downsample_factor))))
        labelled_image_upsample = (
            labelled_image_upsample[:self.original_image_dimensions[0],
                                    :self.original_image_dimensions[1]])

        # Save the image, setting the color of each segment to the average of the segment in the original image.
        image.imsave(
            filename, label2rgb(labelled_image_upsample,
                                self.image,
                                kind='avg'))

    def __str__(self):
        return f"image({self.image_filename})"
