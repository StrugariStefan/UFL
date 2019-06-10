from __future__ import print_function, unicode_literals
import regex

from pprint import pprint
from PyInquirer import style_from_dict, Token, prompt
from PyInquirer import Validator, ValidationError
import click

from utils import Persistance
from utils import reshape

from feature_learner import kernel
from classifier import classification_algorithms
from model import Model

class NumberValidator(Validator):
    def validate(self, document):
        try:
            int(document.text)
        except ValueError:
            raise ValidationError(
                message='Please enter a number',
                cursor_position=len(document.text))  # Move cursor to end

style = style_from_dict({
    Token.QuestionMark: '#E91E63 bold',
    Token.Selected: '#673AB7 bold',
    Token.Instruction: '',  # default
    Token.Answer: '#2196f3 bold',
    Token.Question: '',
})

preprocess_questions = [
    {
        'type': 'confirm',
        'name': 'toBeWhiten',
        'message': 'Apply whitening to patches?',
        'default': False
    },
    {
        'type': 'rawlist',
        'name': 'dataset',
        'message': 'Choose a dataset',
        'choices': []
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
        'type': 'confirm',
        'name': 'toNew',
        'message': "Reprocess if already exists?",
        'default': False
    },
]

pretrain_questions = [
    {
        'type': 'rawlist',
        'name': 'patch',
        'message': 'Choose a patches batch',
        'choices': []
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
        'message': 'Do dynamic configuration of alpha paramaeter?',
        'default': False
    },
]
extractfeatures_questions1 = [
    {
        'type': 'rawlist',
        'name': 'dataset',
        'message': 'Choose dataset',
        'choices': []
    },
]

extractfeatures_questions2 = [
    {
        'type': 'rawlist',
        'name': 'centroids',
        'message': 'Choose centroid set',
        'choices': []
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

def preprocess():
    files = get_files("datasets") 
    preprocess_questions[1]['choices'] = files
    answers = prompt(preprocess_questions, style=style)

    from preprocessing import extract_random_patches, preprocessing_algorithms
    x_train_raw = Persistance('datasets').load(answers['dataset'], '')[0]['x_train_raw']
    nextf = preprocessing_algorithms['whitening'] if answers['toBeWhiten'] else preprocessing_algorithms['nothing']
    whitening_s = 'w' if answers['toBeWhiten'] else 'n'

    suffix = '_' + whitening_s + '_rfs' + str(answers['rfs']) + '_s' + str(answers['s'])
    if answers['toNew'] == False:
        try:
            data, args = Persistance('patches').load(answers['dataset'], suffix)
        except FileNotFoundError:
            data = extract_random_patches(x_train_raw, nextf, receptive_field_size = answers['rfs'], stride = answers['s'])
            Persistance('patches').save(data, answers['dataset'], suffix, receptive_field_size = answers['rfs'], stride = answers['s'], whitening = answers['toBeWhiten'])
    else:
        data = extract_centroids(x_train_raw, nextf, receptive_field_size = answers['rfs'], stride = answers['s'])
        Persistance('patches').save(data, answers['dataset'], suffix, receptive_field_size = answers['rfs'], stride = answers['s'], whitening = answers['toBeWhiten'])


def pretrain():
    files = get_files("patches") 
    pretrain_questions[0]['choices'] = files
    answers = prompt(pretrain_questions, style=style)

    patches = reshape(Persistance('patches').load(answers['patch'],'')[0])
    from pretraining import dynamic_configure, extract_centroids
    suffix = '_k' + str(answers['k'])
    if answers['dynamicConfigure'] == True:
        try:
            alpha, arguments = Persistance('alfas').load(answers['patch'], suffix)
            
        except FileNotFoundError:
            alpha = dynamic_configure(patches, 2, answers['k'], m = 11)
            Persistance('alfas').save(alpha, answers['patch'], suffix, k = answers['k'])
    else:
        alpha = 2
    suffix += "_alpha" + str(alpha)
    try:
        data, arguments = Persistance('centroids').load(answers['patch'], suffix)
    except FileNotFoundError:
        data = extract_centroids(patches, 2, answers['k'], alpha, 2)
        Persistance('centroids').save(data, answers['patch'], suffix, k = answers['k'], alpha = alpha)

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
    print (final_centroids.shape)

    print (k)
    print (receptive_field_size)
    print (stride)


    suffix = "_" + answers2['kernel']
    try:
        data, _ = Persistance("repr_train").load(answers2['centroids'], suffix)
    except FileNotFoundError:
        data = FeatureExtractor()(x_train_raw, kernel[answers2['kernel']](final_centroids), k, receptive_field_size, stride)
        Persistance("repr_train").save(data, answers2['centroids'], suffix, kernel = answers2['kernel'])

    try:
        data, _ = Persistance("repr_test").load(answers2['centroids'], suffix)
    except FileNotFoundError:
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

        for classifier in answers['classalg']:
            suffix = "_" + classifier
            classalg = classification_algorithms[classifier]()
            score = classalg(train_features, data['y_train'], test_features, data['y_test'])
            model = Model(classalg, feature_learner, train_features, test_features, data['x_train_raw'], data['y_train'], data['x_test_raw'], data['y_test'], k, receptive_field_size, stride)
            Persistance("models").save(model, feature_set, suffix)




def get_files(dir):
    from os import listdir
    from os.path import isfile, join, basename
    files = [f.split('.npz')[0] for f in listdir(dir) if isfile(join(dir, f))]
    return files

@click.command()
@click.option('-c', '-command', type = click.Choice(['preprocess', 'pretrain', 'extractfeatures', 'trainmodel']))
def main(c):
    if c == 'preprocess':
        preprocess()
    elif c == 'pretrain':
        pretrain()
    elif c == 'extractfeatures':
        extractfeatures()
    else:
        trainmodel()

    


if __name__ == "__main__":
    main()