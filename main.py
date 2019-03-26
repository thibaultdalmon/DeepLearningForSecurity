from filemanager import FileManager
from custommodel import CustomModel
from datagenerator import DataGenerator
import matplotlib.gridspec as gridspec
from sklearn.metrics import *
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def display_pictures(images, n=4):
    """
    This function prints a grid of images.

    Parameters
    ----------
    images : list of images
    n : width of the grid
    """
    width = n
    heigth = len(images) // n
    remains = len(images) % n
    for i in range(heigth + 1):
        # Get the image
        fig = plt.figure(figsize=(16, n))
        gs = gridspec.GridSpec(1, width, wspace=0.0, hspace=0.0)
        for index, img in enumerate(images[i * width:i * width + width]):
            j = index - i * width
            k = i * width + index
            I = img
            # I = imresize(I, (224, 224))
            ax = plt.subplot(gs[0, index])
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.imshow(I, cmap='gray')
            fig.add_subplot(ax)
        plt.show()


target_docs = [
    "FRA_RP15_Front",
    "FRA_RB6_Front",
    "FRA_P3_Front",
    "FRA_P5_Front",
    "CHE_I3_Front"]
#target_docs = ["FRA_P3_Front", "FRA_P5_Front"]

nb_img = 5
img_width = 299
img_height = 299

print('Creating model')
model = CustomModel(
    nb_img=nb_img,
    img_width=img_width,
    img_height=img_height,
    reuse=True)
print('Model created\n')

print('Creating file manager')
f_mng = FileManager(doc_types=target_docs, reuse=True)
print('File manager created\n')

params = {'batch_size': 32,
          'shuffle': True,
          'nb_img': nb_img,
          'img_width': img_width,
          'img_height': img_height}

training_generator = DataGenerator(f_mng.data_files_train, **params)
validation_generator = DataGenerator(f_mng.data_files_crossval, **params)

#model.train(training_generator, validation_generator)

data_files_test = f_mng.describe(f_mng.data_files_test)

predictions = {}
labels = {}
for doc_type in target_docs:
    predictions[doc_type] = []
    labels[doc_type] = []
    test_generator = DataGenerator(data_files_test[doc_type], **params)
    for file in data_files_test[doc_type]:
        X, Xvis, y = test_generator.__visualisation__(file)
        pred = np.mean(model.predict(X))
        predictions[doc_type].append(pred)

        label = y[0]
        labels[doc_type].append(np.mean(label))
print("\n")


threshold = 0.9
epsilon = 1e-07

for doc_type in target_docs:

    acc = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    misclassified_files = []
    for k in range(len(predictions[doc_type])):
        res = np.floor(predictions[doc_type][k] + 1 - threshold)
        if (np.abs(res - labels[doc_type][k]) < 0.5):
            acc += 1
            if (labels[doc_type][k] > 1 - epsilon):
                tp += 1
            else:
                tn += 1
        else:
            misclassified_files.append(data_files_test[doc_type][k])
            if (res - labels[doc_type][k] < 0 + epsilon):
                fn += 1
            else:
                fp += 1
    acc /= len(data_files_test[doc_type])
    tp /= len(data_files_test[doc_type])
    tn /= len(data_files_test[doc_type])
    fp /= len(data_files_test[doc_type])
    fn /= len(data_files_test[doc_type])

    FAR = fp / (fp + tn)
    FRR = fn / (fn + tp)
    MCC = (tp * tn - fp * fn) / np.sqrt((tp + fp)
                                        * (tp + fn) * (tn + fp) * (tn + fn))

    print(
        f"""{doc_type} Misclassified files : {f'{misclassified_files}'[1:-1]}""")
    print(
        f'{doc_type} Nombre exemples: {len(predictions[doc_type])}, accuracy: {acc:.2f}, FAR: {FAR:.2f}, FRR: {FRR:.2f}, MCC: {MCC:.2f}\n')

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')

fpr = {}
tpr = {}
for doc_type in target_docs:
    fpr[doc_type], tpr[doc_type], _ = roc_curve(
        np.array(
            labels[doc_type]), np.array(
            predictions[doc_type]))
    plt.plot(fpr[doc_type], tpr[doc_type], label=doc_type)
    print(
        f'{doc_type} AUC : {roc_auc_score(np.array(labels[doc_type]), np.array(predictions[doc_type])):.2f}')

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
