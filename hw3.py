import json
import math
from collections import defaultdict, Counter
from math import log
from typing import (
    Iterable,
    Any,
    Sequence,
    Generator, Set, Tuple,
)

############################################################
# The following classes and methods are provided as helpers.
# Do not modify them! The stubs for what you need to implement are later in the file.

# DO NOT MODIFY
START_TOKEN = "<start>"
# DO NOT MODIFY
END_TOKEN = "<end>"


# DO NOT MODIFY
class AirlineSentimentInstance:
    """Represents a single instance from the airline sentiment dataset.

    Each instance contains the sentiment label, the name of the airline,
    and the sentences of text. The sentences are stored as a tuple of
    tuples of strings. The outer tuple represents sentences, and each
    sentences is a tuple of tokens."""

    def __init__(
            self, label: str, airline: str, sentences: Sequence[Sequence[str]]
    ) -> None:
        self.label: str = label
        self.airline: str = airline
        # These are converted to tuples so they cannot be modified
        self.sentences: tuple[tuple[str, ...], ...] = tuple(
            tuple(sentence) for sentence in sentences
        )

    def __repr__(self) -> str:
        return f"<AirlineSentimentInstance: {str(self)}>"

    def __str__(self) -> str:
        return f"label={self.label}; airline={self.airline}; sentences={self.sentences}"

    # You should never call this function directly. It's called by data loading functions.
    @classmethod
    def from_dict(cls, json_dict: dict[str, Any]) -> "AirlineSentimentInstance":
        return AirlineSentimentInstance(
            json_dict["label"], json_dict["airline"], json_dict["sentences"]
        )


# DO NOT MODIFY
class SentenceSplitInstance:
    """Represents a potential sentence boundary in context.

    Each instance is labeled with whether it is ('y') or is not ('n') a sentence
    boundary, the characters to the left of the boundary token, the potential
    boundary token itself (punctuation that could be a sentence boundary), and
    the characters to the right of the boundary token."""

    def __init__(
            self, label: str, left_context: str, token: str, right_context: str
    ) -> None:
        self.label: str = label
        self.left_context: str = left_context
        self.token: str = token
        self.right_context: str = right_context

    def __repr__(self) -> str:
        return f"<SentenceSplitInstance: {str(self)}>"

    def __str__(self) -> str:
        return " ".join(
            [
                f"label={self.label};",
                f"left_context={repr(self.left_context)};",
                f"token={repr(self.token)};",
                f"right_context={repr(self.right_context)}",
            ]
        )

    # You should never call this function directly. It's called by data loading functions.
    @classmethod
    def from_dict(cls, json_dict: dict[Any, Any]) -> "SentenceSplitInstance":
        return SentenceSplitInstance(
            json_dict["label"],
            json_dict["left"],
            json_dict["token"],
            json_dict["right"],
        )


# DO NOT MODIFY
def load_sentiment_instances(
        datapath: str,
) -> Generator[AirlineSentimentInstance, None, None]:
    """Load airline sentiment instances from a JSON file."""
    with open(datapath, encoding="utf8") as infile:
        json_list = json.load(infile)
        for json_item in json_list:
            yield AirlineSentimentInstance.from_dict(json_item)


# DO NOT MODIFY
def load_segmentation_instances(
        datapath: str,
) -> Generator[SentenceSplitInstance, None, None]:
    """Load sentence segmentation instances from a JSON file."""
    with open(datapath, encoding="utf8") as infile:
        json_list = json.load(infile)
        for json_item in json_list:
            yield SentenceSplitInstance.from_dict(json_item)


# DO NOT MODIFY
class ClassificationInstance:
    """Represents a label and features for classification."""

    def __init__(self, label: str, features: Iterable[str]) -> None:
        self.label: str = label
        # Features can be passed in as any iterable and they will be
        # stored in a tuple
        self.features: tuple[str, ...] = tuple(features)

    def __repr__(self) -> str:
        return f"<ClassificationInstance: {str(self)}>"

    def __str__(self) -> str:
        return f"label={self.label}; features={self.features}"


############################################################
# The stubs below this are the ones that you should fill in.
# Do not modify anything above this line other than to add any needed imports.


def accuracy(predictions: Sequence[str], expected: Sequence[str]) -> float:
    """Compute the accuracy of the provided predictions."""
    correct = 0
    pred_len = len(predictions)
    expt_len = len(expected)
    print(predictions)
    print(expected)
    if pred_len != expt_len or pred_len == 0 or expt_len == 0:
        raise ValueError
    else:
        for p, e in zip(predictions, expected):
            if p == e:
                correct += 1
        if pred_len == 0:
            return 0.0
        return float(correct / pred_len)


def recall(predictions: Sequence[str], expected: Sequence[str], label: str) -> float:
    """Compute the recall of the provided predictions."""
    correct = 0
    count = 0
    pred_len = len(predictions)
    expt_len = len(expected)
    if pred_len != expt_len or pred_len == 0 or expt_len == 0:
        raise ValueError
    else:
        for p, e in zip(predictions, expected):
            if e == label:
                count += 1
                if p == e:
                    correct += 1
        if count == 0:
            return 0.0
        return float(correct / count)


def precision(predictions: Sequence[str], expected: Sequence[str], label: str) -> float:
    """Compute the precision of the provided predictions."""
    correct = 0
    count = 0
    pred_len = len(predictions)
    expt_len = len(expected)
    if pred_len != expt_len or pred_len == 0 or expt_len == 0:
        raise ValueError
    else:
        for p, e in zip(predictions, expected):
            if p == label:
                count += 1
                if p == e:
                    correct += 1
        if count == 0:
            return 0.0
        return float(correct / count)


def f1(predictions: Sequence[str], expected: Sequence[str], label: str) -> float:
    """Compute the F1-score of the provided predictions."""
    pres = precision(predictions, expected, label)
    rec = recall(predictions, expected, label)
    sum = pres + rec
    prod = pres * rec
    if sum == 0:
        return 0.0
    return float(2 * (prod / sum))


class UnigramAirlineSentimentFeatureExtractor:
    @staticmethod
    def extract_features(instance: AirlineSentimentInstance) -> ClassificationInstance:
        """Extract unigram features from an instance."""
        label = instance.label
        features = set(word.lower() for sentence in instance.sentences for word in sentence)
        return ClassificationInstance(label, features)


class BigramAirlineSentimentFeatureExtractor:
    # Helper code to generate bigrams
    @staticmethod
    def bigrams(sentence: Sequence[str]) -> set[tuple[str, str]]:
        """Return the bigrams contained in a sequence."""
        tpl_set = set()
        sentence = list(sentence)
        sentence.insert(0, START_TOKEN)
        sentence.insert(len(sentence), END_TOKEN)
        for ind in range(len(sentence) - 1):
            tpl_set.add(tuple([sentence[ind].lower(), sentence[ind + 1].lower()]))
        return tpl_set

    @staticmethod
    def extract_features(instance: AirlineSentimentInstance) -> ClassificationInstance:
        """Extract bigram features from an instance."""
        label = instance.label
        features = set()
        for sentence in instance.sentences:
            bgrms = BigramAirlineSentimentFeatureExtractor.bigrams(sentence)
            for tpl in bgrms:
                features.add(str(tpl))
        return ClassificationInstance(label, features)


class BaselineSegmentationFeatureExtractor:
    @staticmethod
    def extract_features(instance: SentenceSplitInstance) -> ClassificationInstance:
        """Extract features for all three tokens from an instance."""
        label = instance.label
        features = set()
        features.add("left_tok=" + instance.left_context)
        features.add("split_tok=" + instance.token)
        features.add("right_tok=" + instance.right_context)
        return ClassificationInstance(label, features)


class InstanceCounter:
    """Holds counts of the labels and features seen during training.

    See the assignment for an explanation of each method."""

    def __init__(self) -> None:
        self.label_counter = Counter()
        self.feature_counter = Counter()
        self.feat_label_joint_counter = Counter()
        self.feat_for_label_counter = Counter()
        self.label_list = set()
        self.feat_set = set()

    def count_instances(self, instances: Iterable[ClassificationInstance]) -> None:
        # You should fill in this loop. Do not try to store the instances!
        for instance in instances:
            self.label_counter[instance.label] += 1
            self.label_list.add(instance.label)
            for feature in instance.features:
                self.feat_for_label_counter[instance.label] += 1
                self.feat_set.add(feature)
                self.feature_counter[feature] += 1
                joint_name = feature + ", " + instance.label
                self.feat_label_joint_counter[joint_name] += 1
        self.label_list = list(self.label_list)

    def label_count(self, label: str) -> int:
        count = 0
        if label in self.label_counter:
            count = self.label_counter[label]
        return count

    def total_labels(self) -> int:
        return sum(self.label_counter.values())

    def feature_label_joint_count(self, feature: str, label: str) -> int:
        count = 0
        joint_name = feature + ", " + label
        if joint_name in self.feat_label_joint_counter:
            count = self.feat_label_joint_counter[joint_name]
        return count

    def labels(self) -> list[str]:
        return self.label_list

    def feature_vocab_size(self) -> int:
        return len(self.feature_counter)

    def feature_set(self) -> set[str]:
        return self.feat_set

    def total_feature_count_for_label(self, label: str) -> int:
        count = 0
        if label in self.feat_for_label_counter:
            count += self.feat_for_label_counter[label]
        return count


class NaiveBayesClassifier:
    """Perform classification using naive Bayes.

    See the assignment for an explanation of each method."""

    # DO NOT MODIFY
    def __init__(self, k: float):
        self.k: float = k
        self.instance_counter: InstanceCounter = InstanceCounter()

    # DO NOT MODIFY
    def train(self, instances: Iterable[ClassificationInstance]) -> None:
        self.instance_counter.count_instances(instances)

    def prior_prob(self, label: str) -> float:
        return float(self.instance_counter.label_count(label) / self.instance_counter.total_labels())

    def likelihood_prob(self, feature: str, label: str) -> float:
        return float(((self.instance_counter.feature_label_joint_count(feature, label) + self.k) /
                      (self.instance_counter.total_feature_count_for_label(label) +
                       (self.instance_counter.feature_vocab_size() * self.k) )))

    def log_posterior_prob(self, features: Sequence[str], label: str) -> float:
        pri_prob = float(math.log(self.prior_prob(label)))
        outcome = pri_prob
        for feature in features:
            if feature in self.instance_counter.feature_set():
                lklhd_prob = float(math.log(self.likelihood_prob(feature, label)))
                outcome += lklhd_prob
        return float(outcome)

    def classify(self, features: Sequence[str]) -> str:
        probs = list()
        labels = self.instance_counter.label_list
        for label in labels:
            probs.append(tuple([self.log_posterior_prob(features, label), label]))
        max_tuple = max(probs)
        return str(max_tuple[1])

    def test(
            self, instances: Iterable[ClassificationInstance]
    ) -> tuple[list[str], list[str]]:
        predicted = list()
        actual = list()
        for instance in instances:
            predicted.append(self.classify(instance.features))
            actual.append(instance.label)
        return tuple([predicted, actual])


# MODIFY THIS AND DO THE FOLLOWING:
# 1. Inherit from UnigramAirlineSentimentFeatureExtractor or BigramAirlineSentimentFeatureExtractor
#    (instead of object) to get an implementation for the extract_features method.
# 2. Set a value for self.k below based on your tuning experiments.
class TunedAirlineSentimentFeatureExtractor(UnigramAirlineSentimentFeatureExtractor):
    def __init__(self) -> None:
        self.k = float(0.5)
        self.feat_extractor = UnigramAirlineSentimentFeatureExtractor()

