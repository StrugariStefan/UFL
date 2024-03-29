import pickle
from ab_llloyds import algorithm1
import pickle
from abc import ABCMeta, abstractmethod
import time
import warnings

class IClassifier:
    __metaclass__ = ABCMeta

    @classmethod 
    def __init__(self, class_name, name = None):
        self.name = name
        self.class_name = class_name
        self.clf = None
        self.accuracy = None
        self.best_c = None

    @classmethod
    def version(self): return "1.0"

    @classmethod
    def load_model(self):
        if not os.path.exists("models"):
            os.makedirs("models")

        filename = "models\\" + self.class_name + "_" + self.name + ".sav"
        if Path(filename).is_file() == True:
            model = pickle.load(open(filename, 'rb'))
            return model
        else:
            raise FileNotFoundError()

    @classmethod
    def __save_model__(self, model):
        if not os.path.exists("models"):
            os.makedirs("models")
        filename = "models\\" + self.class_name + "_" + self.name + ".sav"
        pickle.dump(model, open(filename, 'wb'))
        

    @abstractmethod
    def __call__(self, x_train, y_train, x_test, y_test, to_compute_training_error = False): raise NotImplementedError

    @abstractmethod
    def predict(self, img_repr): raise NotImplementedError


class Lloyds(IClassifier):
    def __init__(self, name = None):
        super().__init__(self.__class__.__name__, name)

    def __call__(self, x_train, y_train, x_test, y_test):
        print ("Lloyds started...")
        k = len(np.unique(y_train)) 
        clusters, voronoi_tiling, clus_indexes = algorithm1(V = x_train, d = 2, k = k, alfa = 2, beta = 2)
        hamming_error = self.__get_hamming_error__(clus_indexes, x_train, y_train, k)
        print ("Lloyds ended")

        return (1.0 - hamming_error)

    def __get_hamming_error__(clus_indexes, x, y, k):
        from sympy.utilities.iterables import multiset_permutations

        if k > 5:
            print ("K limit exceeded")
            return 1.0

        optimal_clus = y[:]

        values = list(set(optimal_clus))
        optimal_clus = list(map(lambda oci: values.index(oci), optimal_clus))

        min_hamming_distance = len(x)
        a = np.arange(k)
        ms_per_gen = list(multiset_permutations(a))
        n = len(ms_per_gen)

        for i, p in enumerate(ms_per_gen):
            potential_clus = [p[i] for i in optimal_clus]
            score = sum([0 if clus_indexes[i] == potential_clus[i] else 1 for i in range(len(clus_indexes)) ])

            if score < min_hamming_distance:
                min_hamming_distance = score
                best_clusering = potential_clus

        return min_hamming_distance / len(x)

    def predict(self, img_repr):
        pass

class Svc(IClassifier):

    def __init__(self, c_range = [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8], name = None):
        super().__init__(self.__class__.__name__, name)
        self.c_range = c_range
        self.clf = None
        self.accuracy = None

    def __call__(self, x_train, y_train, x_test, y_test, to_compute_training_error = False):
        from sklearn.svm import LinearSVC

        max_score = 0
        best_clf = None
        best_c = 1.0
        for c in self.c_range:
            start_time = time.time()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf = LinearSVC(C = c)
                clf.fit(x_train, y_train)

                y_pred = clf.predict(x_test)
                score = sum(y_pred == y_test)

                if to_compute_training_error:
                    y_pred_train = clf.predict(x_train)
                    train_score = sum(y_pred_train == y_train)

            if score > max_score:
                max_score = score
                best_clf = clf
                best_c = c
                if to_compute_training_error:
                    self.train_score = train_score / len(x_train)
                
        self.clf, self.accuracy, self.best_c = best_clf, max_score / y_test.shape[0], best_c

        return (max_score / y_test.shape[0])

    def predict(self, img_repr):
        if self.clf != None:
            return self.clf.predict(img_repr)[0]
        return None

class LogisticRegresssion(IClassifier):

    def __init__(self, c_range = [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8], name = None):
        super().__init__(self.__class__.__name__, name)
        self.c_range = c_range
        self.clf = None
        self.accuracy = None

    def __call__(self, x_train, y_train, x_test, y_test, to_compute_training_error = False):
        from sklearn.linear_model import LogisticRegression

        max_score = 0
        best_clf = None
        best_c = 1.0
        for c in self.c_range:
            start_time = time.time()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf = LogisticRegression(C = c)
                clf.fit(x_train, y_train)

                y_pred = clf.predict(x_test)
                score = sum(y_pred == y_test)

                if to_compute_training_error:
                    y_pred_train = clf.predict(x_train)
                    train_score = sum(y_pred_train == y_train)

            if score > max_score:
                max_score = score
                best_clf = clf
                best_c = c
                if to_compute_training_error:
                    self.train_score = train_score / len(x_train)

        self.clf, self.accuracy, self.best_c = best_clf, max_score / y_test.shape[0], best_c

        return (max_score / y_test.shape[0])

    def predict(self, img_repr):
        if self.clf != None:
            return self.clf.predict(img_repr)[0]
        return None


class Knn(IClassifier):

    def __init__(self, neighbors = range(1, 23, 2), name = None):
        super().__init__(self.__class__.__name__, name)
        self.neighbors = neighbors
        self.clf = None
        self.accuracy = None

    def __call__(self, x_train, y_train, x_test, y_test, to_compute_training_error = False):
        from sklearn.neighbors import KNeighborsClassifier

        max_score = 0
        best_clf = None
        best_c = 1
        for neighbors in self.neighbors:
            start_time = time.time()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf = KNeighborsClassifier(n_neighbors = neighbors)
                clf.fit(x_train, y_train)

                y_pred = clf.predict(x_test)
                score = sum(y_pred == y_test)

                if to_compute_training_error:
                    y_pred_train = clf.predict(x_train)
                    train_score = sum(y_pred_train == y_train)
            if score > max_score:
                max_score = score
                best_clf = clf
                best_c = neighbors
                if to_compute_training_error:
                    self.train_score = train_score / len(x_train)
                
        self.clf, self.accuracy, self.best_c = best_clf, max_score / y_test.shape[0], best_c

        return (max_score / y_test.shape[0])

    def predict(self, img_repr):
        if self.clf != None:
            return self.clf.predict(img_repr)[0]
        return None

classification_algorithms = {
    "svc": Svc,
    "logisticRegression": LogisticRegresssion,
    "knn": Knn
}