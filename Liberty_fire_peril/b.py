import weka.core.jvm as jvm
jvm.start()
import wekaexamples.helper as helper
from weka.core.converters import Loader
from weka.classifiers import Classifier, SingleClassifierEnhancer, MultipleClassifiersCombiner, FilteredClassifier
from weka.classifiers import Evaluation
from weka.filters import Filter
from weka.core.classes import Random
#import weka.plot.classifiers as plot_cls
#import weka.plot.graph as plot_graph
#import weka.core.types as types

# load a dataset
iris_file = "ntrain.csv"
print("Loading dataset: " + iris_file)
loader = Loader(classname="weka.core.converters.CSVLoader")
iris_data = loader.load_file(iris_file)
iris_data.set_class_index(iris_data.num_attributes() - 1)
                                            
# build a classifier and output model
print ("Training J48 classifier on iris")
classifier = Classifier(classname="weka.classifiers.functions.LinearRegression")
#classifier = Classifier(classname="weka.classifiers.trees.J48", options=["-C", "0.5"])
# Instead of using 'options=["-C", "0.3"]' in the constructor, we can also set the "confidenceFactor"
# property of the J48 classifier itself. However, being of type float rather than double, we need
# to convert it to the correct type first using the double_to_float function:
#classifier.set_property("confidenceFactor", types.double_to_float(0.3))
classifier.build_classifier(iris_data)
print(classifier)
print(classifier.graph())
#plot_graph.plot_dot_graph(classifier.graph())
    

evaluation = Evaluation(iris_data)                     # initialize with priors
evaluation.crossvalidate_model(classifier, iris_data, 10, Random(42))  # 10-fold CV
print(evaluation.to_summary())
print("pctCorrect: " + str(evaluation.percent_correct()))
print("incorrect: " + str(evaluation.incorrect()))
jvm.stop()
