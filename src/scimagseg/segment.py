"""
A python script for segmenting a given image.
"""
import argparse
import warnings

import scimagseg.imgraph
import scimagseg.sc

warnings.filterwarnings("ignore", message="Images with dimensions")


def segment_image(input_filename, output_filename, num_clusters):
    """
    Segment the given image with spectral clustering, and save the output to output_filename.

    :param input_filename:
    :param output_filename:
    :param num_clusters:
    :return:
    """
    # Construct the image graph
    img_graph_ds = scimagseg.imgraph.ImageDatasetGraph(input_filename)

    # Apply spectral clustering to the image - experimental evidence suggest that using half as many eigenvectors as
    # clusters provides a better image segmentation.
    segments = scimagseg.sc.sc_num_eigenvectors(img_graph_ds, num_clusters,
                                                int(num_clusters / 2))

    # Save the result
    img_graph_ds.save_clustering(segments, output_filename)


def parse_args():
    """Configure the command line arguments for this script."""
    parser = argparse.ArgumentParser(description='Segment a single image.')
    parser.add_argument('inputFilename',
                        type=str,
                        help="the filename of the image to be segmented")
    parser.add_argument(
        'outputFilename',
        type=str,
        help="the name of the file to save the segmented image to")
    parser.add_argument('numSegments',
                        type=int,
                        help="the number of segments to find in the image")
    return parser.parse_args()


def main():
    args = parse_args()
    segment_image(args.inputFilename, args.outputFilename, args.numSegments)


if __name__ == "__main__":
    main()
