from keras.utils import Sequence
import numpy as np
import pickle
import os
import json
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import xception


class DataGenerator(Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, batch_size=32, shuffle=True, nb_img=5,
                 img_width=299, img_height=299, root_path='./data'):
        """
        This function creates a DataGenerator able to feed a network with
        examples.

        Parameters
        ----------
        list_IDs : list of directories to use for data generation
        batch_size : int representing size of a batch of examples
        shuffle : boolean indicating weither to shuffle examples
        nb_img : number of neighbour images needed to create an example
        img_width : the target image width
        img_height : the target image width
        root_path : the parent directory
        """
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.nb_img = nb_img
        self.img_width = img_width
        self.img_height = img_height
        self.root_path = root_path
        self.on_epoch_end()

    def __len__(self):
        """
        This function returns the maximal number of batches which can be
        generated without using any example twice.

        Returns
        -------
        the number of batches of examples that the generator can provide
        """
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        """
        This function returns the next batch of examples to use.
        Note that an example is entirely determined by the fist image and the
        number of images to use.

        Parameters
        ----------
        index : the number of batches already used

        Returns
        -------
        X : the next batch of examples
        y : the labels associated to the examples
        """

        # Generate indexes of the batch
        indexes = self.indexes[index *
                               self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        """
        This function creates a new ordering of the indexes associated to the
        examples.
        """

        self.file_indexes = []
        for index_dir in self.list_IDs:
            index_file = 0
            tmp_path = os.path.join(self.root_path, str(index_dir),
                                    f"crop_{(index_file+self.nb_img):02d}.png")
            while os.path.isfile(tmp_path):
                index_file += 1
                tmp_path = os.path.join(
                    self.root_path,
                    str(index_dir),
                    f"crop_{(index_file+self.nb_img):02d}.png")
            self.file_indexes.append(index_file)

        self.indexes = [[self.list_IDs[k], i]
                        for k in range(len(self.list_IDs))
                        for i in range(self.file_indexes[k])]
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        """
        This function builds a batch of examples, given the first image of each
        example.

        Parameters
        ----------
        indexes : list of images to be used as first images in each of the next
        batch_size examples.

        Returns
        -------
        X : the next batch of examples
        y : the labels associated to the examples
        """
        labels = []
        examples = []

        for index in indexes:
            label_path = os.path.join(
                self.root_path, str(
                    index[0]), "metadata.json")
            json_file = open(label_path)
            metadata = json.load(json_file)
            label = metadata["fake"]
            val = np.zeros((1, 1))
            if label == 'True':
                val[0][0] = 0
            else:
                val[0][0] = 1
            labels.append(val)

            examples.append([self.__single_data_generation(
                index[0], index[1] + i) for i in range(self.nb_img)])

        X = [np.concatenate([example[i] for example in examples])
             for i in range(self.nb_img)]
        y = np.concatenate([label for label in labels], axis=0)
        return X, y

    def __single_data_generation(self, file, index):
        """
        Builds a single example

        Parameters
        ----------
        file : directory of the first image used to build the example
        index : id of the image in the directory

        Returns
        -------
        example : a list of nb_img images
        """
        example_path = os.path.join(
            self.root_path, str(file), f"crop_{index:02d}.png")
        example = self.format_image_classif(example_path)[0]
        return example

    def __visualisation__(self, file):
        """
        This function returns visualisation ready images from a given directory

        Parameters
        ----------
        file : the directory to explore

        Returns
        -------
        X : the next batch of examples
        Xvis : the corresponding list of RGB images for visualisation
        y : the labels associated to the examples
        """
        index = 0
        for f in self.list_IDs:
            if (f == file):
                break
            index += 1

        labels = []
        examples = []

        for k in range(self.file_indexes[index]):
            label_path = os.path.join(
                self.root_path, str(file), "metadata.json")
            json_file = open(label_path)
            metadata = json.load(json_file)
            label = metadata["fake"]
            val = np.zeros((1, 1))
            if label == 'True':
                val[0][0] = 0
            else:
                val[0][0] = 1
            labels.append(val)

            examples.append([self.__single_visualisation_generation(
                file, k + i) for i in range(self.nb_img)])

        X = [np.concatenate([example[i][0] for example in examples])
             for i in range(self.nb_img)]
        Xvis = [example[0][1] for example in examples]
        y = np.concatenate([label for label in labels], axis=0)
        return X, Xvis, y

    def __single_visualisation_generation(self, file, index):
        """
        This function generates a single example and images which can be
        visualised.

        Parameters
        ----------
        file : directory of the first image used to build the example
        index : id of the image in the directory

        Returns
        -------
        example : a list of nb_img images with their visualisation
        """
        example_path = os.path.join(
            self.root_path, str(file), f"crop_{index:02d}.png")
        example = self.format_image_classif(example_path)
        return example

    def format_image_classif(self, img_file):
        """
        This function reads and formats an image so that it can be fed to the xception network.
        In this case, we wish to force the image size to a certain shape, since we want to use the image for
        classification

        Parameters
        ----------
        img_file : image file name
        img_width : the target image width
        img_height : he target image height

        Returns
        -------
        img_out_xception : the correctly formatted image for xception
        img : the image as read by the load_img function of keras.preprocessing.image
        """
        img_width = self.img_width
        img_height = self.img_height
        # read image. Force the image size to a certain shape (uses a resize of
        # the pillow package)
        img = load_img(img_file, target_size=(img_height, img_width))
        # convert image to an array
        img_out = img_to_array(img)
        # preprocess the image to put in the correct format for use with the
        # xception network trained on imagenet
        img_out_xception = xception.preprocess_input(img_out)
        # add a dimension at the beginning, coresponding to the batch dimension
        img_out_xception = np.expand_dims(img_out_xception, axis=0)
        return img_out_xception, img
