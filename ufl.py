from __future__ import print_function, unicode_literals
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import regex
import multiprocessing

from pprint import pprint
from PyInquirer import style_from_dict, Token, prompt
from PyInquirer import Validator, ValidationError
import click
import numpy as np

from utils import Persistance
from utils import reshape

from feature_learner import kernel
from ab_llloyds import local_search_procedures, initialization_procedures
from classifier import classification_algorithms
from model import Model

from load_images import load_images
from load_images import load_labels
from pretraining import performance_test, cost_function

from tabulate import tabulate

import csv
from imageio import imwrite
from math import log10, ceil
import random

class NumberValidator(Validator):
    def validate(self, document):
        try:
            int(document.text)
            if int(document.text) < 0:
                raise ValidationError(
                    message='Please enter a positive number',
                    cursor_position=len(document.text))
        except ValueError:
            raise ValidationError(
                message='Please enter a number',
                cursor_position=len(document.text))

class FloatValidator(Validator):
    def validate(self, document):
        try:
            float(document.text)
            if float(document.text) < 0 or float(document.text) > 1:
                raise ValidationError(
                    message='Please enter a number between 0 and 1',
                    cursor_position=len(document.text))
        except ValueError:
            raise ValidationError(
                message='Please enter a float number',
                cursor_position=len(document.text)) 

style = style_from_dict({
    Token.QuestionMark: '#E91E63 bold',
    Token.Selected: '#673AB7 bold',
    Token.Instruction: '',
    Token.Answer: '#2196f3 bold',
    Token.Question: '',
})

preprocess_questions = [
    {
        'type': 'list',
        'name': 'dataset',
        'message': 'Choose a dataset',
        'choices': [],
        'filter': lambda val: val.lower()
    },
    {
        'type': 'confirm',
        'name': 'toBeWhiten',
        'message': 'Apply whitening to patches?',
        'default': False
    },
    {
        'type': 'input',
        'name': 'rfs',
        'message': 'Set receptive field size',
        'validate': NumberValidator,
        'filter': lambda val: int(val)
    },
    {
        'type': 'input',
        'name': 's',
        'message': 'Set stride',
        'validate': NumberValidator,
        'filter': lambda val: int(val)
    },
    {
        'type': 'input',
        'name': 'pp',
        'message': 'Set patching probability',
        'validate': FloatValidator,
        'filter': lambda val: float(val)
    },
    {
        'type': 'confirm',
        'name': 'toNew',
        'message': "Reprocess if already exists?",
        'default': False
    },
]

pretrain_questions = [
    {
        'type': 'list',
        'name': 'patch',
        'message': 'Choose a patches batch',
        'choices': [],
        'filter': lambda val: val.lower()
    },
    {
        'type': 'input',
        'name': 'k',
        'message': 'Set number of centroids',
        'validate': NumberValidator,
        'filter': lambda val: int(val)
    },
    {
        'type': 'confirm',
        'name': 'dynamicConfigure',
        'message': 'Do dynamic configuration of alpha parameter?',
        'default': False
    },
]

pretrain_questions2 = [
    {
        'type': 'rawlist',
        'name': 'init',
        'message': 'Select the initialization procedure',
        'choices': initialization_procedures.keys()
    },
    {
        'type': 'rawlist',
        'name': 'localsearch',
        'message': 'Select the local search procedure',
        'choices': local_search_procedures.keys()
    },
]


extractfeatures_questions1 = [
    {
        'type': 'list',
        'name': 'dataset',
        'message': 'Choose dataset',
        'choices': [],
        'filter': lambda val: val.lower()
    },
]

extractfeatures_questions2 = [
    {
        'type': 'list',
        'name': 'centroids',
        'message': 'Choose centroid set',
        'choices': [],
        'filter': lambda val: val.lower()
    },
    {
        'type': 'rawlist',
        'name': 'kernel',
        'message': 'Choose kernel encoding',
        'choices': kernel.keys()
    },
]

trainmodel_questions = [
    {
        'type': 'checkbox',
        'message': 'Select features to use for building the classifier',
        'name': 'features',
        'choices': [

        ],
        'validate': lambda answer: 'You must choose at least one set of image reprezentations' \
            if len(answer) == 0 else True
    },
    {
        'type': 'checkbox',
        'name': 'classalg',
        'message': 'Choose linear classifier',
        'choices': [{'name':c} for c in classification_algorithms.keys()],
        'validate': lambda answer: 'You must choose at least one linear classifier' \
            if len(answer) == 0 else True
    },
]

predict_questions = [
    {
        'type': 'list',
        'name': 'model',
        'message': 'Choose model',
        'choices': [],
        'filter': lambda val: val.lower()
    },
]

loaddata_questions = [
    {
        'type': 'input',
        'name': 'dataset_name',
        'message': 'Type a name, to identify the dataset',
        'validate': lambda document: 'Name should not be empty' \
            if len(document) == 0 else True
    },
    {
        'type': 'input',
        'name': 'width',
        'message': 'Set images width',
        'validate': NumberValidator,
        'filter': lambda val: int(val)
    },
    {
        'type': 'input',
        'name': 'height',
        'message': 'Set images height',
        'validate': NumberValidator,
        'filter': lambda val: int(val)
    },
]

loaddata_questions2 = [
    {
        'type': 'confirm',
        'name': 'replace',
        'message': 'Dataset already exists. Do you want to replace it?',
        'default': False
    },
]

remove_questions = [
    {
        'type': 'list',
        'name': 'dataset',
        'message': 'Choose dataset for removal',
        'choices': [],
        'filter': lambda val: val.lower()
    },
]

getbyaccuracy_questions = [
    {
        'type': 'list',
        'name': 'dataset',
        'message': 'Choose a dataset',
        'choices': [],
        'filter': lambda val: val.lower()
    },
]

def preprocess():
    files = get_files("datasets") 
    preprocess_questions[0]['choices'] = files
    answers = prompt(preprocess_questions, style=style)

    from preprocessing import extract_random_patches, preprocessing_algorithms
    x_train_raw = Persistance('datasets').load(answers['dataset'], '')[0]['x_train_raw']
    nextf = [preprocessing_algorithms['tf_whitening'], preprocessing_algorithms['whitening']] if answers['toBeWhiten'] else [preprocessing_algorithms['nothing']]
    whitening_s = 'w' if answers['toBeWhiten'] else 'n'

    suffix = '_' + whitening_s + '_rfs' + str(answers['rfs']) + '_s' + str(answers['s'])
    if answers['toNew'] == False:
        try:
            data, args = Persistance('patches').load(answers['dataset'], suffix)
        except FileNotFoundError:
            data = extract_random_patches(x_train_raw, nextf, receptive_field_size = answers['rfs'], stride = answers['s'], patching_probability = answers['pp'])
            Persistance('patches').save(data, answers['dataset'], suffix, receptive_field_size = answers['rfs'], stride = answers['s'], whitening = answers['toBeWhiten'])
    else:
        data = extract_random_patches(x_train_raw, nextf, receptive_field_size = answers['rfs'], stride = answers['s'], patching_probability = answers['pp'])
        Persistance('patches').save(data, answers['dataset'], suffix, receptive_field_size = answers['rfs'], stride = answers['s'], whitening = answers['toBeWhiten'])


def pretrain():
    files = get_files("patches") 
    pretrain_questions[0]['choices'] = files
    answers = prompt(pretrain_questions, style=style)

    patches = reshape(Persistance('patches').load(answers['patch'],'')[0])
    from pretraining import dynamic_configure, extract_centroids
    suffix = '_k' + str(answers['k'])
    if answers['dynamicConfigure'] == True:
        beta = 2
        try:
            alpha, arguments = Persistance('alfas').load(answers['patch'], suffix)
        except FileNotFoundError:
            alpha = dynamic_configure(patches, 2, answers['k'], m = 11)
            Persistance('alfas').save(alpha, answers['patch'], suffix, k = answers['k'])
    else:
        answers2 = prompt(pretrain_questions2, style=style)
        alpha = initialization_procedures[answers2['init']]
        beta = local_search_procedures[answers2['localsearch']]
    suffix += "_alpha" + str(alpha) + "_beta" + str(beta)
    try:
        data, arguments = Persistance('centroids').load(answers['patch'], suffix)
    except FileNotFoundError:
        data = extract_centroids(patches, 2, answers['k'], alpha, beta)
        Persistance('centroids').save(data, answers['patch'], suffix, k = answers['k'], alpha = alpha, beta = beta)

def extractfeatures():
    files = get_files("datasets") 
    extractfeatures_questions1[0]['choices'] = files
    answers1 = prompt(extractfeatures_questions1, style=style)

    files = get_files("centroids") 
    extractfeatures_questions2[0]['choices'] = [f for f in files if f.startswith(answers1['dataset'])]
    answers2 = prompt(extractfeatures_questions2, style=style)

    files = get_files("patches")
    patch_batch_file = [f for f in files if answers2['centroids'].startswith(f)][0]

    from feature_extraction import FeatureExtractor
    data, _ = Persistance("datasets").load(answers1['dataset'], "")
    x_train_raw = data['x_train_raw']
    x_test_raw = data['x_test_raw']

    data, arguments = Persistance("patches").load(patch_batch_file, "")
    receptive_field_size = arguments['receptive_field_size']
    stride = arguments['stride']

    data, arguments = Persistance("centroids").load(answers2['centroids'], "")
    final_centroids = data[0]
    k = arguments['k']

    suffix = "_" + answers2['kernel']
    try:
        data, _ = Persistance("repr_train").load(answers2['centroids'], suffix)
    except FileNotFoundError:
        print ("Train images")
        data = FeatureExtractor()(x_train_raw, kernel[answers2['kernel']](final_centroids), k, receptive_field_size, stride)
        Persistance("repr_train").save(data, answers2['centroids'], suffix, kernel = answers2['kernel'])

    try:
        data, _ = Persistance("repr_test").load(answers2['centroids'], suffix)
    except FileNotFoundError:
        print ("Test images")
        data = FeatureExtractor()(x_test_raw, kernel[answers2['kernel']](final_centroids), k, receptive_field_size, stride)
        Persistance("repr_test").save(data, answers2['centroids'], suffix, kernel = answers2['kernel'])

def trainmodel():
    files = get_files("repr_train") 
    trainmodel_questions[0]['choices'] = [{"name": f} for f in files]
    answers = prompt(trainmodel_questions, style=style)
    
    for feature_set in answers['features']:
        train_features, arguments = Persistance('repr_train').load(feature_set, '')
        test_features, _ = Persistance('repr_test').load(feature_set, '')
        
        files = get_files("datasets") 
        dataset = [f for f in files if feature_set.startswith(f)][0]
        data, _ = Persistance("datasets").load(dataset, "")

        files = get_files("patches")
        patch_batch = [f for f in files if feature_set.startswith(f)][0]
        _, patch_args = Persistance("patches").load(patch_batch, "")

        files = get_files("centroids")
        centroids = [f for f in files if feature_set.startswith(f)][0]
        (final_centroids, _, _), _ = Persistance("centroids").load(centroids, "")
        
        feature_learner = kernel[arguments['kernel']](final_centroids)
        k = len(final_centroids)
        receptive_field_size = patch_args['receptive_field_size']
        stride = patch_args['stride']

        print (feature_set)
        for classifier in answers['classalg']:
            suffix = "_" + classifier
            classalg = classification_algorithms[classifier]()
            score = classalg(train_features, data['y_train'], test_features, data['y_test'], True)
            model = Model(classalg, feature_learner, train_features, test_features, data['x_train_raw'], data['y_train'], data['x_test_raw'], data['y_test'], k, receptive_field_size, stride, data['labels'])
            
            print ("Classifier: " + str(model.classifier.__class__.__name__))
            print ("\tTraining accuracy: " + str(model.classifier.train_score))
            print ("\tValidation accuracy: " + str(model.classifier.accuracy))
            print ("\tBest regularization parameter: " + str(model.classifier.best_c))
            
            Persistance("models").save(model, feature_set, suffix)

predict_questions1 = [
    {
        'type': 'list',
        'name': 'dataset',
        'message': 'Choose dataset',
        'choices': [],
        'filter': lambda val: val.lower()
    },
]


def predict(path, plot):
    files = get_files("datasets") 
    predict_questions1[0]['choices'] = files
    answers1 = prompt(predict_questions1, style=style)

    files = get_files("models") 
    predict_questions[0]['choices'] = [f for f in files if f.startswith(answers1['dataset'])]
    answers = prompt(predict_questions, style=style)
    
    model, _ = Persistance("models").load(answers['model'], '')

    print ("Training accuracy: " + str(model.classifier.train_score))
    print ("Validation accuracy: " + str(model.classifier.accuracy))

    width, height = model.x_train.shape[1:3]

    images, files = load_images(path, width, height, to_resize = True)
    
    predictions = dict()

    if len(images) > 0:
        y_pred = model.predict(images)
        for i, f in enumerate(files):
            predictions[f] = model.labels[y_pred[i]]

        
        if plot == False:
            pprint(predictions)
        else:

            import matplotlib.pyplot as plt
            from skimage.io import imread

            fig, ax = plt.subplots()
            i = 0
            for k, v in predictions.items():
                img = imread(os.path.join(path, k))
                p = plt.imshow(img)
                plt.title(v, fontsize = 40)
                fig = plt.gcf()
                plt.pause(1.3)



def loaddata(xtp, xtep, ytp, ytep, lp):
    answers = prompt(loaddata_questions, style=style)
    files = get_files("datasets") 
    
    if answers['dataset_name'] in files:
        answers2 = prompt(loaddata_questions2, style=style)
        if answers2['replace'] == False:
            return



    x_train_raw, _ = load_images(xtp, answers['width'], answers['height'], to_resize=True)
    x_test_raw, _ = load_images(xtep, answers['width'], answers['height'], to_resize=True)

    y_train = np.array(load_labels(ytp))
    y_test = np.array(load_labels(ytep))
    labels = load_labels(lp, type=str)

    data = dict()
    data['x_train_raw'] = x_train_raw
    data['x_test_raw'] = x_test_raw
    data['y_train'] = y_train
    data['y_test'] = y_test
    data['labels'] = labels

    Persistance("datasets").save(data, answers['dataset_name'], "")

def get_by_accuracy():
    files = get_files("datasets") 
    getbyaccuracy_questions[0]['choices'] = files
    answers = prompt(getbyaccuracy_questions, style=style)

    files = get_files("models")

    models_info = []
    for file in files:
        if file.startswith(answers['dataset']):
            model, _ = Persistance("models").load(file, '')
            models_info.append((file, model.classifier.__class__.__name__, model.classifier.train_score if model.classifier.train_score != None else '-', model.classifier.accuracy))

    models_info.sort(key = lambda el: el[3] + 1e-5 * el[2], reverse = True)

    print (tabulate(models_info, headers = ['Model', 'Classifier', 'Train', 'Validation']))

performance_test_questions = [
    {
        'type': 'list',
        'name': 'patch',
        'message': 'Choose a patches batch',
        'choices': [],
        'filter': lambda val: val.lower()
    },
    {
        'type': 'input',
        'name': 'k',
        'message': 'Set number of centroids',
        'validate': NumberValidator,
        'filter': lambda val: int(val)
    },
    {
        'type': 'rawlist',
        'name': 'costfunc',
        'message': 'Choose cost function',
        'choices': cost_function.keys()
    },
]

def take_perfomance_test():
    files = get_files("patches") 
    performance_test_questions[0]['choices'] = files
    answers = prompt(performance_test_questions, style=style)
    
    patches = reshape(Persistance('patches').load(answers['patch'],'')[0])
    suffix = '_k' + str(answers['k'])

    try:
        costs, arguments = Persistance('performance').load(answers['patch'], suffix)
    except FileNotFoundError:
        costs = performance_test(patches, 2, answers['k'], answers['costfunc'])
        Persistance('performance').save(costs, answers['patch'], suffix, k = answers['k'])

base_dataset_questions = [
    {
        'type': 'input',
        'name': 'train_sample',
        'message': 'Select the train sample size',
        'validate': NumberValidator,
        'filter': lambda val: int(val)
    },
    {
        'type': 'input',
        'name': 'test_sample',
        'message': 'Select the test sample size',
        'validate': NumberValidator,
        'filter': lambda val: int(val)
    },
]

def load_base_dataset(dataset):
    from base_datasets import dataset_name

    ((x_train, y_train), (x_test, y_test)), labels = dataset_name[dataset]()

    answers = prompt(base_dataset_questions, style=style)

    r1 = min(x_train.shape[0], answers['train_sample'])
    r2 = min(x_test.shape[0], answers['test_sample'])

    n1 = len(x_train)
    x1 = {i for i in range(n1)}
    xs1 = set(random.sample(x1, r1))

    n2 = len(x_test)
    x2 = {i for i in range(n2)}
    xs2 = set(random.sample(x2, r2))

    x_train_raw = x_train[np.array(list(xs1))]
    x_test_raw = x_test[np.array(list(xs2))]
    y_train = y_train[np.array(list(xs1))].flatten()
    y_test = y_test[np.array(list(xs2))].flatten()

    data = dict()
    data['x_train_raw'] = x_train_raw
    data['x_test_raw'] = x_test_raw
    data['y_train'] = y_train
    data['y_test'] = y_test
    data['labels'] = labels

    dir_path = str(r1) + dataset

    p = Persistance("datasets")
    p.save(data, dir_path, "")

    if is_dir(dir_path) == False:
        os.makedirs(dir_path)

    xt_len = ceil(log10(len(x_train_raw)))

    if is_dir(dir_path + "\\train") == False:
            os.makedirs(dir_path + "\\train")

    if is_dir(dir_path + "\\test") == False:
        os.makedirs(dir_path + "\\test")
    
    for i, image in enumerate(x_train_raw):
        istr = str(i)
        imwrite(dir_path + '\\train\\' + istr.zfill(xt_len) + '.jpg', image)

    xte_len = ceil(log10(len(x_test_raw)))    
    for i, image in enumerate(x_test_raw):
        istr = str(i)
        imwrite(dir_path + '\\test\\' + istr.zfill(xte_len) + '.jpg', image)

    train_labels = []
    for i, label in enumerate(y_train):
        train_labels.append(label)

    with open(dir_path + '\\train.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(train_labels)

    test_labels = []
    for i, label in enumerate(y_test):
        test_labels.append(label)

    with open(dir_path + '\\test.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(test_labels)

    with open(dir_path + '\\labels.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(data['labels'])

def remove():
    files = get_files("datasets")
    remove_questions[0]['choices'] = files
    answers = prompt(remove_questions, style=style)
    
    remove_files("datasets", answers['dataset'])
    remove_files("patches", answers['dataset'])
    remove_files("alfas", answers['dataset'])
    remove_files("centroids", answers['dataset'])
    remove_files("repr_train", answers['dataset'])
    remove_files("repr_test", answers['dataset'])
    remove_files("models", answers['dataset'])



def remove_files(dirpath, prefix):
    if not os.path.isdir(dirpath):
        return

    for file in os.listdir(dirpath):
        if dirpath == "datasets":
            print (file, prefix)
        if file.startswith(prefix):
            os.remove(os.path.join(dirpath, file))

def is_dir(dir): return os.path.isdir(dir)
def is_file(file): return os.path.isfile(file)
def is_csv(file): return os.path.basename(file).lower().endswith(".csv")

def get_files(dir):
    from os import listdir
    from os.path import isfile, join, basename
    files = [f.split('.npz')[0] for f in listdir(dir) if isfile(join(dir, f))]
    return files

@click.command()
@click.option('-c', '-command', type = click.Choice(['loaddata', 'performance', 'remove', 'preprocess', 'pretrain', 'extractfeatures', 'trainmodel', 'predict', 'getbyaccuracy', 'mnist', 'cifar10']))
@click.option('-p', '-path', type = str)
@click.option('-xtp', '-x_train_path', type = str)
@click.option('-ytp', '-y_train_path', type = str)
@click.option('-xtep', '-x_test_path', type = str)
@click.option('-ytep', '-y_test_path', type = str)
@click.option('-lp', '-labels_path', type = str)
@click.option('--plot/--no-plot', default = False)
def main(c, p, xtp, ytp, xtep, ytep, lp, plot):
    if c in ['mnist', 'norb', 'cifar10']:    
        load_base_dataset(c)
    elif c == 'performance':
        take_perfomance_test()
    elif c == 'loaddata':
        if xtp == None or ytp == None or xtep == None or ytep == None or lp == None:
            print ("Specify path to train and test images directories,train and test labels files and the labels semnification file...\n\t-xtp\n\t--x_train_path\n\n\t-ytp\n\t--y_train_path\n\n\t-xtep\n\t--x_test_path\n\n\t-ytep\n\t--y_test_path\n\n\t-lp\n\t--labels_path\n")
            return
        if is_dir(xtp) == False:
            print ("x train path is not a valid directory path")
            return
        if is_dir(xtep) == False:
            print ("x test path is not a valid directory path")
            return
        if is_file(ytp) == False or is_csv(ytp) == False:
            print ("y train path is not a valid csv file path")
            return
        if is_file(ytep) == False or is_csv(ytep) == False:
            print ("y test path is not a valid csv file path")
            return
        if is_file(lp) == False or is_csv(lp) == False:
            print ("labels path is not a valid csv file path")
            return
        loaddata(xtp, xtep, ytp, ytep, lp)
    elif c == 'getbyaccuracy':
        get_by_accuracy()
    elif c == 'remove':
        remove()
    elif c == 'preprocess':
        preprocess()
    elif c == 'pretrain':
        pretrain()
    elif c == 'extractfeatures':
        extractfeatures()
    elif c == 'trainmodel':
        trainmodel()
    else:
        if p == None:
            print ("Specify path to folder or image...\n\t-p\n\t--path")
            return
        try:
            predict(p, plot)
        except NotADirectoryError:
            print ("Directory doesn't exist")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()