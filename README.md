# README - COSI 114a HW3

The code provided in this repository contains the solutions to HW3 for COSI 114a - Fundamentals of Natural Language Processing I. The assignment had us implement a naive Bayes classifier and apply it to the tasks of sentence segmentation and sentiment analysis. 

As this assignment was done for a class, some helper files and testing files were provided. All student-written solutions to the assignment were written in the ``` hw3.py ``` file. 

## Installation and Execution 

Get the files from GitHub and in your terminal/console move into the project folder. To run the test file included with the files given to students, run the following: 

``` bash 
python test_hw3.py 
```

Doing this will run the set of tests in the test file that was provided to students to test their code. Running the test file will print a score to the user's console indicating the success of the program for each given test case. Make sure to unzip the ``` test_data.zip ``` file before running the above code. It has been compressed to save space on the upload, but is necessary for testing the code. 

Note: The test file provided only made up a portion of the final grade for the assignment. More extensive tests were done during final grading, but students weren't given access to those tests. Furthermore, these instructions assume that the user has python downloaded and is able to run the ``` python ``` command in their terminal. 


## Assignment Description 

### Accuracy, Precision, Recall, and F1

The first task was to implement the functions to compute accuracy, precision, recall, and F1 score. The methods doing so are all under their respective names. The implementation was done from scratch, as students weren't allowed to use or rely on scikit-learn for its implementation. 

Each of the functions take two arguments: a ``` Sequence ``` of predicted labels and a ``` Sequence ``` of expected labels. Additionally, the ``` F1 ```, ``` precision ```, and ``` recall ``` functions also take the string label to be considered as positive as an argument. Each of the methods return a ``` float ```. 

For all functions, a check is done to ensure that the predicted and expected label sequences are the same length and non-empty. If they are not the same length or are empty, a ``` ValueError ``` exception is raised. It is also set so that none of the functions will crash due to a zero division error. If the denominator is zero, it returns 0.0 instead. 

Note: It is not assumed that the positive label appears in either the predicted or expected values. 

### Extracting Features from a Text 

The following three feature extractor classes were implemented: 

1. ``` BaselineSegmentationFeatureExtractor ``` 
2. ``` UnigramAirlineSentimentFeatureExtractor ```
3. ``` BigramAirlineSentimentFeatureExtractor ```

Each of these classes have a static method called ``` extract_features ```, which takes an input Python representations of the raw data and returns a ``` ClassificationInstance ```. ``` ClassificationInstance ``` is a general representation of data point to be classified. It has ``` features ```, which are a tuple of strings, and a string ``` label ```. 

#### Airline Sentiment Features 

In NLP, a common set of features is the Bag of Words representation, which represents a document as an unordered collection of words that appear in the document, meaning word order is completely ignored. It's like a binary indicator of whether a given word is present in the document or not. Often, the tokens are lowercased before computing the BoW representation to ensure that tokens that are the same word but have differing capitalizations are treated as an instance of the same linguistic entity, instead of having them represent different things. The implementations of ``` UnigramAirlineSentimentFeatureExtractor ``` and ``` BigramAirlineSentimentFeatureExtractor ``` abides by this, so all tokens are lowercased and no feature is repeated when extracting bag-of-words features. 

Things to Note: 
1. All text was lowercased before the features were computer. 
2. The order of the features doesn't matter; the only thing that matters is that each feature is present no more than once for a given instance. A set was used to implement this. 
3. For the bigrams, each feature is a string created by calling ``` str ``` on a tuple. This is done so that there isn't a mix of types later on. 

#### Sentence Segmentation Features 

Features were extracted as strings for sentence segmentation, but this was no longer done in the form of using the bag-of-words/n-grams features. Instead, for ``` BaselineSegmentationFeatureExtractor ```, each feature is represented as a string of the form ``` "feature_name=feature_value". 

For ``` BaselineSegmentationFeatureExtractor ```, the following three basic features were implemented: 
1. Split Token 
    * Candidate token that would divide the sentence 
    * EX: ``` split_tok=! ``` and ``` split_tok=. ```
2. Right Token 
    * The token immediately to the right of the split token, aka the right context 
    * EX: ``` right_tok=Louis ``` and ``` right_tok=The ``` 
3. Left Token 
    * The token immediately to the left of the split token, aka the left context 
    * EX: ``` left_tok=St ``` and ``` left_tok=dog ``` 

### Instance Counter 

Training for models often comes down to computing counts over the training data and then normalizing the counts to obtain valid probability distributions. For naive Bayes, this process involves computing how many times a label was seen in the data and how many times each feature was observed with each label. To facilitate the this, an ``` InstanceCounter ``` class was implemented, which iterates over the data and counts the labels and features for each of the training instances seen, providing all the counts needed for a naive Bayes model. 

The ``` count_instances ``` method iterates over ``` ClassificationInstance ``` objects and count features and labels. The following methods have also been implemented in order to assist ``` count_instances ```: 

* ``` label_count(self, label: str) -> int ``` - Given a label, returns how many times that label was seen. Used to compute prior probabilities over labels. 
* ``` total_labels(self) -> int ``` - Returns the number of total labels seen in the training data (just a number of instances seen, as each only has one label). This is not the number of unique labels. 
* ``` feature_label_joint_count(self, feature: str, label: str) -> int ``` - Given a feature and a label, how many times does feature show up when only counting over instances of class ``` label ```? For example, how many times does the unigram feature "good" appear among instances with the label "positive"? 
* ``` labels(self) -> list[str]``` - Returns a list of the unique labels observed. The order of the labels doesn't matter, and for ease, this method returns an instance variable already containing the correct information. 
* ``` feature_vocab_size(self) -> int ``` - Returns the size of the feature vocabulary. This is the number of unique features that have been seen across all instances, regardless of label. 
* ``` feature_set(self) -> set[str] ``` - Returns a set that contains all the features observed across all labels. This is needed by the classifier's ``` log_posterior_prob ``` method to filter out features that weren't seen during training. The size of this set should be the same value returned by ``` feature_vocab_size ```. 
* ``` total_feature_count_for_label(self, label: str) -> int ``` - Returns the total count of all features seen for a given label. 

### Naive Bayes Classifier 

#### Initialization 

The ``` init ``` method of the ``` NaiveBayesClassifier ``` has been provided in the starter code. It uses the ``` InstanceCounter ``` implemented as described above to compute counts over the training data. The constructor also takes a ``` float ``` argument ``` k ```, which is the value that is used for add-k smoothing. 

It is assumed that ``` k ``` is always a positive number throughout the implementation of this class, and that ``` train ``` is called with an instance counter that has had ``` count_instances ``` called on it before any other methods are called. 

#### Training 

The training method has also been provided in the starter code. It uses ``` count_instances ``` to set counts on the instance counter. 

#### Helper Methods 

The following methods were implemented to aid in classification so that the assignment could be completed: 

* ``` prior_prob(self, label: str) -> float ``` - Returns the prior probability for a label, which is the count of the label divided by the total number of instances seen during training 
* ``` likelihood_prob(self, feature: str, label: str) -> float ``` - Returns the likelihood for a single feature, which is computed as $\frac{Count(f, C) + k}{N + Vk}$. This is the count of the feature with the given label (computed by ``` feature_label_joint_count ``` plus the ``` k ``` value set it ``` __ init__ ```) divided by the total count of all the features observed with that class plus the feature vocabulary size (the number of unique features observed) multiplied by ``` k ```. 
* ``` log_posterior_prob(self, features: Sequence[str], label: str) -> float ``` - Returns the posterior probability, which is computed by combining the prior probability with the likelihood probability of all features. Any features that never appearing in training are skipped, and their probability is not included in the calculation. However, that only applies to features that were never seen in training at all, not features that were seen with one label in training but not all labels (these are handled by smoothing). 
    * To avoid numerical instability from multiplying when calculating the posterior probability, ``` math.log ``` and the log identity of ``` log(xy) = log(x) + log(y) ``` are used instead to compute the results. 

#### Classification 

The ``` classify ``` method has been implemented, which returns the most probable label given a list of features. This method utilizes the methods mentioned above in order to work by using the python ``` max() ``` function on the posterior probabilities it's been passed. 

#### Testing 

The ``` test ``` function has also been implemented. This method takes an iterable of ``` ClassificationInstance ``` objects and returns a tuple of two lists. The two lists contain the predicated and true labels of classification instances, respectively. 

