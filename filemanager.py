import pickle
import os
import json
import random as rand


class FileManager:

    def __init__(self, doc_types=None, reuse=True):
        """
        This function creates a FileManager which handles the repartition of the
        directories between the train, crossval and test datasets.

        Parameters
        ----------
        doc_types : a list of strings representing the types of the files to
            include in the datatasets.
        reuse : if True, loads the previous directory repartition.
        """

        self.doc_types = doc_types
        self.root_path = "./data"
        if reuse:
            self.load_files()
        else:
            self.new_directory_repartition(self.doc_types, self.root_path)

    def load_files(self):
        """
        This function restores a previous directory repartition, it does NOT check
        that the doc_types are correct.
        """

        with open('data_files_train', 'rb') as fp:
            self.data_files_train = pickle.load(fp)

        with open('data_files_crossval', 'rb') as fp:
            self.data_files_crossval = pickle.load(fp)

        with open('data_files_test', 'rb') as fp:
            self.data_files_test = pickle.load(fp)

    def new_directory_repartition(self, doc_types=None, root_path="./data"):
        """
        This function creates a new directory repartition between the datasets :
        train : 85%
        crossval : 5%
        test : 15%
        It checks the type of the files in the directory and that they are not
        empty.

        Parameters
        ----------
        doc_types : a list of strings representing the types of the files to
            include in the datatasets.
        root_path : the path of the parent directory
        """

        self.doc_types = doc_types
        self.root_path = root_path
        files = os.listdir(root_path)
        datas = []
        for file in files:
            file_path = os.path.join(root_path, file, "metadata.json")
            json_file = open(file_path)
            try:
                metadata = json.load(json_file)
                if metadata["template"] in doc_types:
                    datas.append(int(file))
            except BaseException:
                print(file + " empty")
        datas.sort()

        self.data_files_train = []
        self.data_files_test = []
        self.data_files_crossval = []
        for file in datas:
            tmp = rand.randint(1, 100)
            if (tmp < 80):
                self.data_files_train.append(file)
            elif (tmp < 85):
                self.data_files_crossval.append(file)
            else:
                self.data_files_test.append(file)

        with open('data_files_train', 'wb') as fp:
            pickle.dump(self.data_files_train, fp)

        with open('data_files_crossval', 'wb') as fp:
            pickle.dump(self.data_files_crossval, fp)

        with open('data_files_test', 'wb') as fp:
            pickle.dump(self.data_files_test, fp)

    def describe(self, list):
        """
        This function builds lists of directories based on the type of the files
        that they contain and returns them in a dictionnary whose keys are
        the file types.

        Parameters
        ----------
        list : list of directories.

        Returns
        -------
        datas : the dictionnary of the directories, indexed by their type.
        """
        
        datas = {}
        for doc_type in self.doc_types:
            datas[doc_type] = []

        for file in list:
            file_path = os.path.join(
                self.root_path, str(file), "metadata.json")
            json_file = open(file_path)
            try:
                metadata = json.load(json_file)
                if metadata["template"] in self.doc_types:
                    datas[metadata["template"]].append(int(file))
            except BaseException:
                print(file + " empty")

        for doc_type in self.doc_types:
            datas[doc_type].sort()
            print(f"{doc_type} : {len(datas[doc_type])} sessions")

        return datas
