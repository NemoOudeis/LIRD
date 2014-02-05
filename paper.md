#Machine Learning Final Project Paper
##Introduction

###Who we are
[Name1]
[IMAT1]

[Name2]
[IMAT2]

###What we are about to do: Letter recognition (Task 3)
We will use [scikit-learn](http://scikit-learn.org/stable/install.html) to apply selected machine learning algorithms on the [provided Dataset](http://archive.ics.uci.edu/ml/datasets/Letter+Recognition) (20000 Sampled, 16 features).

All results were achieved in the following setup:

* Ubuntu & Mac OS X 10.9.1
* [scikit-learn 0.14.1](http://scikit-learn.org/stable/install.html)
* [Python 2.7](http://python.org/download/releases/2.7/)
* [NumPy 1.6.2](http://www.numpy.org/)

###Our Tools
We chose the following classifiers for the survey:

1. Support Vector Machines
2. Decision Trees
3. k Nearest Neighbors
4. Random & Ensemble Methods: Random Trees and Random Forests

SVMs were chosen as starting point, as it sets out to separate data into 2 classes. From there we investigate the different constructive approaches to multi-class classification i.e. tricks like one-vs-one or one-vs-rest cross validation in the SVM section. Decision Tree methods, another intuitive extension of binary classification to multiple classes, are examined next. 
As next step we investigate how to make use of the geometrical structure of data in a discussion of k Nearest Neighbour methods. Finally we peek into the possibilities of randomized algorithms and how known methods can be used to constructively create better classifiers with ensemble methods.

All classifiers are analysed with regard to preprocessing, optimizing parameters, performance and results. In every section the command line call used to create the stated (possibly contracted) will be given and respective command line arguments explained.

The project source code can be found at [GitHub](https://github.com/vsaw/LIRD) and most of this document is maintained in the corresponding wiki.

## [Support Vector Machines](https://github.com/vsaw/LIRD/wiki/Support-Vector-Machines)

Implemented as [`sklearn.svm.SVC`](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC). Different [kernels](http://scikit-learn.org/stable/modules/svm.html#svm-kernels) are available. scikit-learn states that *"multiclass support is handled according to a one-vs-one scheme"*.


### SVC(kernel='rbf')
These are the results when the default `SVC` is applied on the data set. The kernel is given by `exp(-gamma * |x - x'|^2)` where `gamma = 1/n_features`.

It seems to perform rather well on the data set with a classification error of 111/4000 = ~2.8% when using the default settings. Although, normalizing the data has a negative effect on it, the default `SVC` classifier has an classification error of 228/4000 = 5.7%.

For fitting: The *one-vs-one* training results in a rather long training and fitting time compared to the other classifiers. However normalizing the data reduced the fitting time to one third.


### SVC(kernel='sigmoid')
The kernel is given by `tanh(gamma * <x, x'> + coef0)` where `gamma = 1/n_features` and `coef0 = 0`.


#### Untuned sigmoid kernels
The classification was horrible. The classification error for unaltered input set was 3856/4000 = 96.4%. Normalizing the data did not help, the error did not change.

For fitting: For the unaltered data the sigmoid kernel was as fast as the rbf kernel. But interestingly normalizing the data made the classifier run ~35% longer.


#### Tuning the sigmoid kernel
A closer look on the confusion matrix revealed that in all cases everything will be classified to the same label. Now if we look at the kernel it can be seen that a `tanh` function is used, the `tanh` is pretty sensitve around `0` but quickly saturates at 1 if the input into the `tanh` is larger than 2. Now we pass the dot product of our vectors to the `tanh`, this can be rather large. Manual computation of the dot product on all the vectors of our data showed that it is in the range of 268 - 962. If we now scale this with `gamma = 1/n_features = 1/16` in our case we still end up way above the sensitive area of the `tanh` function. Choosing `gamma = 1/max( dot product of all vectors in our test set ) = 1/962` lets the sigmoid kernel successfully classify about 75% of the data!


### SVC(kernel='poly')
The kernel is given by `(gamma * <x, x'> + coef0)**degree` where `gamma = 1/n_features`, `coef0 = 0` and `degree = 3`.

Original data: 206/4000 = ~5.15%, normalized data: 453/4000 = ~11.3%. So again normalizing the data did not help.

For fitting: On original data it took 48% of the time needed by the rbf kernel. For the normalized data it took 75% of the time needed by the rbf kernel.


### SVC(kernel='linear')
The kernel is given by `<x, x'>`.

Original data: 644/4000 = ~16.1%, normalized data: 642/4000 = ~16.05%. So again normalizing the data did not help.

For fitting: On original data it took ~31% of the time needed by the rbf kernel. For the normalized data it took ~66% of the time needed by the rbf kernel.


LinearSVC
---------
Implemented as [`sklearn.svm.LinearSVC`](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html). Similar to `SVC(kernel='linear')` but with some implementation difference, most notably the use of *one-vs-rest* multiclass scheme.

Classification results were pretty bad compared to the `SVC(kernel='linear')`. On the original data set 1655/4000 = ~41.3% of the data was missclassified. The normalized data has a missclassification rate of 1214/4000 = ~30% which is a significant improvement. 

The performance in fitting on the original data was roughly the same as `SVC(kernel='rbf')`. On the normalized data it was ~66% slower than `SVC(kernel='rbf')`. It is worth noting that the `LinearSVC` was slower than the `SVC(kernel='linear')` even though the `LinearSVC` uses the cheaper *one-vs-rest* scheme unlike the `SVC(kernel='linear')` which uses the *one-vs-one* scheme. The *one-vs-one* scheme has a runtime complexity of `O(n_classes^2)` whilst *one-vs-rest* is in `O(n_classes)`. So the difference in the runtime must come from the underlying libraries because `LinearSVC` uses [liblinear](http://www.csie.ntu.edu.tw/~cjlin/liblinear/) and the `SVC` uses [libsvm](http://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf).


One-vs-One and One-vs-Rest Schemes
----------------------------------
On the first run it can be seen that there is huge difference in the error rate of the `SVC(kernel='linear')` and the `LinearSVC`. The reason behind this is that the SVC uses a *one-vs-one* multiclass scheme and the `LinearSVC` uses the *one-vs-rest* scheme.

scikit-learn has a [`OneVsRestClassifier`](http://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html) wrapper that turns the `SVC(kernel='linear')` into a *one-vs-rest* classifier. As expected the results are in the same order as of the `LinearSVC` so that *one-vs-one* has a significant advantage for given the data set.

<table>
	<tr>
		<th>Method</th>
		<th>Classification error rate original data</th>
		<th>Classification error rate normalized data</th>
	</tr>
	<tr>
		<td><!-- Method --><tt>SVC(kernel='linear')</tt></td>
		<td><!-- Error Rate orig. data -->16.1%</td>
		<td><!-- Error Rate norm. data -->16.05%</td>
	</tr>
	<tr>
		<td><!-- Method --><tt>LinearSVC</tt></td>
		<td><!-- Error Rate orig. data -->41.4%</td>
		<td><!-- Error Rate norm. data -->30,35%</td>
	</tr>
	<tr>
		<td><!-- Method --><tt>SVC(kernel='linear')</tt> (one-vs-rest)</td>
		<td><!-- Error Rate orig. data -->42.75%</td>
		<td><!-- Error Rate norm. data -->41,57%</td>
	</tr>
</table>

Summary
-------
The versatile support vector machines offer a lot of tuning to produce good classification results but can also perform abysmally bad. For the given data set the *one-vs-one* multiclass scheme performed significantly better and provided the best tuning compared to using different kernels or parameters to a kernel.

<table>
	<tr>
		<th>Method</th>
		<th>Classification error rate original data</th>
		<th>Fitting runtime original data compared to <tt>SVC(kernel='rbf')</tt></th>
		<th>
			Prediction runtime original data compared to fitting time of
			<tt>SVC(kernel='rbf')</tt>
		</th>
		<th>Classification error rate normalized data</th>
		<th>
			Fitting runtime normalized data compared to fitting time of
			<tt>SVC(kernel='rbf')</tt>
		</th>
		<th>
			Prediction runtime normalized data compared to fitting time of
			<tt>SVC(kernel='rbf')</tt>
		</th>
	</tr>
	<tr>
		<td><!-- Method --><tt>SVC(kernel='rbf')</tt></td>
		<td><!-- Error Rate orig. data -->2.775%</td>
		<td><!-- Fitting time orig. data -->1 = 29s</td>
		<td><!-- Prediction time orig. data -->0.21</td>
		<td><!-- Error Rate norm. data -->5.7%</td>
		<td><!-- Fitting time norm. data -->0.41</td>
		<td><!-- Prediction time norm. data -->0.24</td>
	</tr>
	<tr>
		<td><!-- Method --><tt>SVC(kernel='poly')</tt></td>
		<td><!-- Error Rate orig. data -->5.15%</td>
		<td><!-- Fitting time orig. data -->0.48</td>
		<td><!-- Prediction time orig. data -->0.03</td>
		<td><!-- Error Rate norm. data -->11.3%</td>
		<td><!-- Fitting time norm. data -->0.31</td>
		<td><!-- Prediction time norm. data -->0.10</td>
	</tr>
	<tr>
		<td><!-- Method --><tt>SVC(kernel='linear')</tt></td>
		<td><!-- Error Rate orig. data -->16.1%</td>
		<td><!-- Fitting time orig. data -->0.31</td>
		<td><!-- Prediction time orig. data -->0.03</td>
		<td><!-- Error Rate norm. data -->16.05%</td>
		<td><!-- Fitting time norm. data -->0.27</td>
		<td><!-- Prediction time norm. data -->0, less than 1s</td>
	</tr>
	<tr>
		<td><!-- Method --><tt>LinearSVC</tt></td>
		<td><!-- Error Rate orig. data -->41.4%</td>
		<td><!-- Fitting time orig. data -->1.03</td>
		<td><!-- Prediction time orig. data -->0, less than 1s</td>
		<td><!-- Error Rate norm. data -->30,35%</td>
		<td><!-- Fitting time norm. data -->0.68</td>
		<td><!-- Prediction time norm. data -->0, less than 1s</td>
	</tr>
	<tr>
		<td><!-- Method --><tt>SVC(kernel='linear')</tt> (one-vs-rest)</td>
		<td><!-- Error Rate orig. data -->42.75%</td>
		<td><!-- Fitting time orig. data -->5</td>
		<td><!-- Prediction time orig. data -->0.24</td>
		<td><!-- Error Rate norm. data -->41,57%</td>
		<td><!-- Fitting time norm. data -->2.51</td>
		<td><!-- Prediction time norm. data -->0.21</td>
	</tr>
	<tr>
		<td><!-- Method --><tt>SVC(kernel='sigmoid')</tt></td>
		<td><!-- Error Rate orig. data -->96.4%</td>
		<td><!-- Fitting time orig. data -->0.96</td>
		<td><!-- Prediction time orig. data -->0.17</td>
		<td><!-- Error Rate norm. data -->96.4%</td>
		<td><!-- Fitting time norm. data -->1.31</td>
		<td><!-- Prediction time norm. data -->0.24</td>
	</tr>
</table>

## [Decision Trees](https://github.com/vsaw/LIRD/wiki/Decision-Trees)

###Warnings from [SciKit Docu](http://scikit-learn.org/stable/modules/tree.html#tips-on-practical-use)
- Decision tree learners create biased trees if some classes dominate. It is therefore recommended to balance the dataset prior to fitting with the decision tree.
- Decision trees tend to overfit on data with a large number of features. Getting the right ratio of samples to number of features is important, since a tree with few samples in high dimensional space is very likely to overfit.
- Consider performing dimensionality reduction (PCA, ICA, or Feature selection) beforehand to give your tree a better chance of finding features that are discriminative.

See SciKit [user guide](http://scikit-learn.org/stable/modules/tree.html) and [sklearn.tree](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.tree) module documentation.

###Preprocessing
Decision Trees only work with numeric labels, so we used a [label encoder](http://scikit-learn.org/stable/modules/preprocessing.html#label-encoding) from the [preprocessing](http://scikit-learn.org/stable/modules/preprocessing.html) package

Moreover we used [feature selection](http://scikit-learn.org/stable/modules/feature_selection.html#feature-selection) to reduce the feature dimensionality by only considering "relevant" features.

###Optimization
After experimenting with all [parameters](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) of the classifier we found that:
- The function to measure the quality of a split **criterion** gini and entropy both are about equal
- The minimum number of samples required to split an internal node **min_samples_split=2** (default) always yields the best results, irrespective of training set size
- The minimum number of samples required to be at a leaf node **min_samples_leaf=1** (default) always best, irrespective of training set size
- The maximum depth of the tree **max_depth** influences the result seemingly random, no proper optimization possible
- The number of features to consider when looking for the best split **max_features** yields best results for values between 85 - 90 %, better than sqrt(#feature) (default) and log2(#feature)
- [feature selection](http://scikit-learn.org/stable/modules/feature_selection.html#tree-based-feature-selection) reduces dimensionality by ~50% but still delivers as good, sometimes even better, results
- only about 2-5 % of improvement after tuning

As to why does **max_depth** has a seemingly random influence on the result:
First of all it is clear, that very small values of **max_depth** will lead to very bad results (because it limits the number separating hyperplanes to a maximum of 2^**max_depth**) but after a certain threshold classifier with potentially very deep trees perform just as good as those with comparably shallow trees. 

###Performance & Results
Decision Tress train and predict exceptionally fast, with time less than 1 second for any of the tried instances with 80% data for training and 20% for validation

Even with very small training set size Decision Trees deliver comparably good results (almost 50% correct classification rate for 198000 samples after training with only 200 samples).

As shown in the table below the accuracy of prediction only increases very slowly after a certain point and feature selection was used in all "best" classifiers of the given tests.

<table>
	<tr>
		<th>Training/Validation Set Size</th>
		<th>Classification score</th>
		<th>Clssifier Specification (best classifier of instance)</th>
	</tr>
	<tr>
		<td><!-- T/V Set Size --><tt>1% - 99%</tt></td>
		<td><!-- Score -->47.73%</td>
		<td><!-- Spec -->Max Features 0.85% with entropy criterion and feature selection</td>
	</tr>
        <tr>
		<td><!-- T/V Set Size --><tt>2% - 98%</tt></td>
		<td><!-- Score -->57.41%</td>
		<td><!-- Spec -->Max Features 0.87% with gini criterion and feature selection</td>
	</tr>
        <tr>
		<td><!-- T/V Set Size --><tt>3% - 97%</tt></td>
		<td><!-- Score -->62.87%</td>
		<td><!-- Spec -->Max Features 0.85% with entropy criterion and feature selection</td>
	</tr>
        <tr>
		<td><!-- T/V Set Size --><tt>10% - 90%</tt></td>
		<td><!-- Score -->73.62%</td>
		<td><!-- Spec -->Max Features 0.87% with gini criterion and feature selection</td>
	</tr>	
        <tr>
		<td><!-- T/V Set Size --><tt>21% - 79%</tt></td>
		<td><!-- Score -->80.19%</td>
		<td><!-- Spec -->Max Features 0.90% with gini criterion and feature selection</td>
	</tr>
        <tr>
		<td><!-- T/V Set Size --><tt>80% - 20%</tt></td>
		<td><!-- Score -->88.67%</td>
		<td><!-- Spec -->Max Features 0.87% with entropy criterion and feature selection</td>
	</tr>
</table>


**Remark:**
Normalizing the dataset has no influence on the result of decision tree classifier, which implies that decision trees do not need any regularity with regard to distribution of the data.

When you think about it, for a single feature's `f_i` mean to be of by some `mu_i` the hyperplanes of all cuts are moved by `e_i*mu_i` and the (im)purity remains the same. Analogously the (im)purity is invariant to scaling by constants, as all points in space (and hyperplanes) keep their relative positioning when scaling one dimension.

###Command line
```
# decision tree on original data with feature selection before classification with detailed output
python lird.py -v 3 tree --data orig --select-feature on
```

## [k Nearest Neighbors](https://github.com/vsaw/LIRD/wiki/Nearest-Neighbors)

TODO @ VALE

## [Random Tree & Ensemble Methods](https://github.com/vsaw/LIRD/wiki/Random-&-Ensemble-Methods)


### Random Tree
From the [Reference](http://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeClassifier.html#sklearn.tree.ExtraTreeClassifier)

> "An extremely randomized tree classifier.

> Extra-trees differ from classic decision trees in the way they are built. When looking for the best split to separate the samples of a node into two groups, random splits are drawn for each of the max_features randomly selected features and the best split among those is chosen. When max_features is set 1, this amounts to building a totally random decision tree.

> Warning: Extra-trees should only be used within ensemble methods."

Even though random decision trees should (intuitively) be worse than regular decision trees algorithms they performed pretty well on the given data set, only lagging about 5% in classification score (up to ~83%).

For a dataset with a quite few samples and few features (like ours) they are not relevant, but they enable us to analyse bigger and more complex datasets (i.e. multiple orders of magnitude more) and allow for ensemble methods, because of their performance.

###Ensemble Methods

[Reference](http://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees)

We consider [random forest classifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier) and [extra random trees classifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier) for the analysis of ensemble methods.
Both methods construct multiple trees from a sample of the training data set in some random fashion and combine them (by averaging) to create superior classifiers. They differ in the way the random trees are constructed:

####Random Forest
When splitting a node the classifier only considers a random subset of features and choses the best split (with some (im)purity measure,e.g. gini coefficient). This way all trees differ significantly, as each split considers different features.

> As a result of this randomness, the bias of the forest usually slightly increases (with respect to the bias of a single non-random tree) but, due to averaging, its variance also decreases, usually more than compensating for the increase in bias, hence yielding an overall better model.

####Extra Random Trees
Here a split will only consider a random subset of features, sample a couple of random cuts and chose the best of them (according to the features), thus adding more randomness into the mix.

###Preprocessing 
See [Decision Trees](https://github.com/vsaw/LIRD/wiki/Decision-Trees)

###Optimization
The parameters considered in [Decision Trees](https://github.com/vsaw/LIRD/wiki/Decision-Trees) will not be considered again here, we will concentrate on the ensemble method specific parameters.

- **n_estimators** is the number of trees constructed. Intuition tells us that the more the better up to a certain threshold. Our experiments confirm this assumption, with the threshold around 20 trees
- **bootstrap** and **oob_score** influence way samples are drawn from the dataset to construct the trees. They did not have a significant impact on the result, i.e. <0.5 %. But bootstrap=false consistently delivered the better results.

Feature selection does not work too well on ensemble methods (compared to pure decision trees), which is not surprising. Feature selection reduces features by almost 50%, after that ensemble methods selects further subsets of the "relevant" (i.e. the feature selector deemed the features relevant) to build random trees. In this process *each* tree disregards relevant features in construction, whereas ensemble methods without feature selection do not have this problem in *every* tree. So feature selection degrades (almost) all trees used in ensembling the classifier, which shows in the averaged result.

###Performance & Results
Just like decision trees random trees and ensemble methods are very time efficient (50 random forests in 8 seconds, 50 extra random trees in 2 seconds).  

As you can see ensemble methods take random trees, that are about 10 % worse than decision trees, and combine them in a smart way such that the resulting classifier is about 10 % better than decision trees (this is all very rough, but it gets the point across).

Another interesting result is that random forests perform better than extra random trees for estimator count of a10 or less. As soon as this border is breached extra random trees outperform random forests. This manifests the intuition that averaging of randomness (extra random forests) can only be better then systematic approach/algorithm (random forests) when the amount of samples used for averaging is big enough (-> law of big numbers).

<table>
	<tr>
		<th>Training/Validation Set Size</th>
		<th>Ensamble</th>
                <th>Decision Trees</th>
                <th>Random Tree</th>
		<th>Ensaemble Classifier Specification</th>
	</tr>
	<tr>
		<td><!-- T/V Set Size --><tt>1% - 99%</tt></td>
		<td><!-- Score -->60.56%</td>
		<td><!-- DT -->47.73%</td>
		<td><!-- RT -->42.08%</td>
		<td><!-- Spec -->46 Extra Random Trees</td>
	</tr>
        <tr>
		<td><!-- T/V Set Size --><tt>2% - 98%</tt></td>
		<td><!-- Score -->72.61%</td>
		<td><!-- DT -->57.41%</td>
		<td><!-- RT -->50.29%</td>
		<td><!-- Spec -->46 Extra Random Trees</td>
	</tr>
        <tr>
		<td><!-- T/V Set Size --><tt>3% - 97%</tt></td>
		<td><!-- Score -->77.58%</td>
		<td><!-- DT -->62.87%</td>
		<td><!-- RT -->53.09%</td>
		<td><!-- Spec -->42 Extra Random Trees</td>
	</tr>
        <tr>
		<td><!-- T/V Set Size --><tt>10% - 90%</tt></td>
		<td><!-- Score -->87.4%</td>
		<td><!-- DT -->73.62%</td>
		<td><!-- RT -->66.94%</td>
		<td><!-- Spec -->50 Extra Random Trees</td>
	</tr>	
        <tr>
		<td><!-- T/V Set Size --><tt>21% - 79%</tt></td>
		<td><!-- Score -->91.72%</td>
		<td><!-- DT -->80.19%</td>
		<td><!-- RT -->73.48%</td>
		<td><!-- Spec -->50 Extra Random Trees</td>
	</tr>
        <tr>
		<td><!-- T/V Set Size --><tt>80% - 20%</tt></td>
		<td><!-- Score -->96.88%</td>
		<td><!-- DT -->88.67%</td>
		<td><!-- RT -->83.35%</td>
		<td><!-- Spec -->38 Extra Random Trees</td>
	</tr>
</table>

###Command line
```
# -v/--verbose level from 1-3
# random trees on orginial and normalized data
python lird.py -v 3 random

# ensemble method on original data without feature selection, forest size between 5 and 50
python lird.py -v 3 ensemble --data orig --select-feature off --min-trees 5 --max-trees 50
```