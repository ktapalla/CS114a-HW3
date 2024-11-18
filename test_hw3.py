#! /usr/bin/env python

# Version 1.1
# 10/10/2022

import math
import os
import unittest
from typing import Iterable, Iterator, TypeVar

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from grader import Grader, points, timeout
from hw3 import (
    accuracy,
    precision,
    recall,
    f1,
    load_segmentation_instances,
    ClassificationInstance,
    load_sentiment_instances,
    InstanceCounter,
    NaiveBayesClassifier,
    UnigramAirlineSentimentFeatureExtractor,
    BaselineSegmentationFeatureExtractor,
    TunedAirlineSentimentFeatureExtractor,
    BigramAirlineSentimentFeatureExtractor,
    SentenceSplitInstance,
    AirlineSentimentInstance,
)

T = TypeVar("T")
SENTENCE_SPLIT_DIR = os.path.join("test_data", "sentence_splits")
AIRLINE_SENTIMENT_DIR = os.path.join("test_data", "airline_sentiment")


class TestScoringMetrics(unittest.TestCase):
    @points(1)
    def test_accuracy(self):
        """Accuracy is a float and has the correct value."""
        predicted = ["T", "T", "T", "T", "F", "F", "F", "F", "T", "T"]
        actual = ["T", "T", "T", "T", "T", "F", "F", "F", "F", "F"]
        self.assertEqual(float, type(accuracy(predicted, actual)))
        self.assertAlmostEqual(0.7, accuracy(predicted, actual))

    @points(1)
    def test_precision(self):
        """Precision is a float and has the correct value for each class."""
        predicted = ["T", "T", "T", "T", "F", "F", "F", "F", "T", "T"]
        actual = ["T", "T", "T", "T", "T", "F", "F", "F", "F", "F"]
        self.assertEqual(float, type(precision(predicted, actual, "T")))
        self.assertAlmostEqual(2 / 3, precision(predicted, actual, "T"))
        self.assertAlmostEqual(0.75, precision(predicted, actual, "F"))

    @points(1)
    def test_recall(self):
        """Recall is a float and has the correct value for each class."""
        predicted = ["T", "T", "T", "T", "F", "F", "F", "F", "T", "T"]
        actual = ["T", "T", "T", "T", "T", "F", "F", "F", "F", "F"]
        self.assertEqual(float, type(recall(predicted, actual, "T")))
        self.assertAlmostEqual(0.8, recall(predicted, actual, "T"))
        self.assertAlmostEqual(3 / 5, recall(predicted, actual, "F"))

    @points(1)
    def test_f1_score(self):
        """F1 is a float and has the correct value for each class."""
        predicted = ["T", "T", "T", "T", "F", "F", "F", "F", "T", "T"]
        actual = ["T", "T", "T", "T", "T", "F", "F", "F", "F", "F"]
        self.assertEqual(float, type(f1(predicted, actual, "T")))
        self.assertAlmostEqual(8 / 11, f1(predicted, actual, "T"))
        self.assertAlmostEqual(2 / 3, f1(predicted, actual, "F"))


class TestSegmentationFeatureExtractor(unittest.TestCase):
    sentence_instances: list[SentenceSplitInstance]

    @classmethod
    def setUpClass(cls):
        # Store these to avoid reloading the data every test
        cls.sentence_instances = list(
            load_segmentation_instances(os.path.join(SENTENCE_SPLIT_DIR, "dev.json"))
        )

    def setUp(self):
        self.feature_extractor = BaselineSegmentationFeatureExtractor()

    @points(1)
    def test_type_instance_sentence_split_classification(self):
        """Feature extraction for sentence segmentation produces ClassificationInstance objects."""
        instance = self.feature_extractor.extract_features(self.sentence_instances[0])
        self.assertEqual(ClassificationInstance, type(instance))

    @points(1)
    def test_instance_label_negative_sentence_split_classification(self):
        """The label of the ClassificationInstance representing dev sentence 1 is 'n'."""
        instance = self.feature_extractor.extract_features(self.sentence_instances[0])
        self.assertEqual("n", instance.label)

    @points(1)
    def test_features_correct_sentence_split_classification(self):
        """Correct features are extracted for dev sentence 1."""
        features = self.feature_extractor.extract_features(
            self.sentence_instances[0]
        ).features
        self.assertSetEqual(
            {"split_tok=.", "left_tok=D", "right_tok=Forrester"},
            set(features),
        )

    @points(1)
    def test_predicted_labels_valid_sentence_split(self):
        """All predicted labels are valid for sentence segmentation."""
        for instance in self.sentence_instances:
            classification_instance = self.feature_extractor.extract_features(instance)
            self.assertIn(classification_instance.label, {"y", "n"})


class TestAirlineSentimentUnigramFeatureExtractor(unittest.TestCase):
    sentiment_instances: list[AirlineSentimentInstance]

    @classmethod
    def setUpClass(cls):
        # Store these to avoid reloading the data every test
        cls.sentiment_instances = list(
            load_sentiment_instances(os.path.join(AIRLINE_SENTIMENT_DIR, "dev.json"))
        )

    def setUp(self):
        self.feature_extractor = UnigramAirlineSentimentFeatureExtractor()

    @points(0.5)
    def test_instance_label_negative_airline_sentiment_classification(self):
        """The label of the ClassificationInstance representing dev sentence 1 is 'negative'."""
        instance = self.feature_extractor.extract_features(self.sentiment_instances[0])
        self.assertEqual("negative", instance.label)

    @points(1)
    def test_instance_correct_features_airline_sentiment_classification(self):
        """Correct features are extracted for dev sentence 1."""
        features = self.feature_extractor.extract_features(
            self.sentiment_instances[0]
        ).features
        self.assertSetEqual(
            {
                "#",
                "&",
                ",",
                "2",
                ";",
                "?",
                "@nrhodes85",
                "@usairways",
                "above",
                "actually",
                "amp",
                "apologizes",
                "beyond",
                "but",
                "customers",
                "does",
                "for",
                "funny",
                "go",
                "is",
                "it",
                "just",
                "n't",
                "notimpressed",
                "steps",
                "take",
                "that",
            },
            set(features),
        )

    @points(0.5)
    def test_predicted_labels_valid_airline_sentiment_classification(self):
        """All predicted labels are valid for airline sentiment classification."""
        for instance in self.sentiment_instances:
            classification_instance = self.feature_extractor.extract_features(instance)
            self.assertIn(classification_instance.label, {"positive", "negative"})


class TestAirlineSentimentBigramFeatureExtractor(unittest.TestCase):
    @points(2)
    def test_bigram_feature_extractor(self):
        """The bigram feature extractor extracts labels and features correctly."""
        # Load a single sentiment instance
        sentiment_instance = next(
            load_sentiment_instances(
                os.path.join("test_data", "sentiment_test_data.json")
            )
        )
        extractor = BigramAirlineSentimentFeatureExtractor()
        classification_instance = extractor.extract_features(sentiment_instance)
        self.assertEqual("positive", classification_instance.label)
        self.assertEqual(str, type(classification_instance.features[0]))
        self.assertSetEqual(
            {
                "('ever', '!')",
                "('best', 'flight')",
                "('i', 'love')",
                "('flying', '!')",
                "('<start>', 'best')",
                "('flight', 'ever')",
                "('!', '<end>')",
                "('<start>', 'i')",
                "('love', 'flying')",
            },
            set(classification_instance.features),
        )


class SegmentationTestFeatureExtractor:
    """Simple baseline sentiment feature extractor for testing."""

    @staticmethod
    def extract_features(inst: SentenceSplitInstance) -> ClassificationInstance:
        return ClassificationInstance(
            inst.label, [f"left_tok={inst.left_context}", f"split_tok={inst.token}"]
        )


class SentimentTestFeatureExtractor:
    """Simple baseline sentiment feature extractor for testing."""

    @staticmethod
    def extract_features(inst: AirlineSentimentInstance) -> ClassificationInstance:
        return ClassificationInstance(
            inst.label,
            # First five tokens of the first sentence
            inst.sentences[0][:5],
        )


class TestInstanceCounter(unittest.TestCase):
    inst_counter: InstanceCounter
    labels: frozenset[str]

    @classmethod
    def setUpClass(cls) -> None:
        # Create instance counter and count the instances
        feature_extractor = SegmentationTestFeatureExtractor()
        counter = InstanceCounter()
        instances = DefensiveIterable(
            feature_extractor.extract_features(inst)
            for inst in load_segmentation_instances(
                os.path.join(SENTENCE_SPLIT_DIR, "dev.json")
            )
        )
        counter.count_instances(instances)
        # Store attributes in the class
        cls.inst_counter = counter
        cls.labels = frozenset(["y", "n"])

    @points(2)
    @timeout(0.01)
    def test_label_counts_y(self):
        """The correct number of instances of the label 'y' is observed."""
        self.assertEqual(6110, self.inst_counter.label_count("y"))

    @points(2)
    @timeout(0.01)
    def test_label_counts_n(self):
        """The correct number of instances of the label 'n' is observed."""
        self.assertEqual(811, self.inst_counter.label_count("n"))

    @points(1)
    @timeout(0.01)
    def test_total_labels(self):
        """The correct total number of labels is observed."""
        self.assertEqual(6921, self.inst_counter.total_labels())

    @points(2)
    @timeout(0.01)
    def test_feature_label_joint_count_1(self):
        """A period appears as the sentence boundary in the correct number of cases."""
        self.assertEqual(
            5903, self.inst_counter.feature_label_joint_count("split_tok=.", "y")
        )

    @points(2)
    @timeout(0.01)
    def test_feature_label_joint_count_2(self):
        """A period appears as a non-boundary in the correct number of cases."""
        self.assertEqual(
            751, self.inst_counter.feature_label_joint_count("split_tok=.", "n")
        )

    @points(3)
    @timeout(0.01)
    def test_labels(self):
        """All observed labels are valid and the total number of observed labels is correct."""
        labels = self.inst_counter.labels()
        self.assertEqual(list, type(labels))
        self.assertSetEqual(self.labels, set(labels))

    @points(1.5)
    @timeout(0.01)
    def test_feature_vocab_size(self):
        """The correct total number of features is returned."""
        self.assertEqual(2964, self.inst_counter.feature_vocab_size())

    @points(1.5)
    @timeout(0.01)
    def test_feature_set(self):
        """The correct set of features is returned."""
        self.assertEqual(set, type(self.inst_counter.feature_set()))
        self.assertEqual(2964, len(self.inst_counter.feature_set()))

    @points(3)
    @timeout(0.01)
    def test_total_feature_count_for_label(self):
        """The correct total number of features is observed for both classes."""
        self.assertEqual(12220, self.inst_counter.total_feature_count_for_label("y"))
        self.assertEqual(1622, self.inst_counter.total_feature_count_for_label("n"))


class TestNaiveBayesSegmentation(unittest.TestCase):
    classifier: NaiveBayesClassifier

    @classmethod
    def setUpClass(cls) -> None:
        # Create and train classifier
        feature_extractor = SegmentationTestFeatureExtractor()
        train_instances = (
            feature_extractor.extract_features(inst)
            for inst in load_segmentation_instances(
                os.path.join(SENTENCE_SPLIT_DIR, "train.json")
            )
        )
        cls.classifier = NaiveBayesClassifier(0.1)
        cls.classifier.train(train_instances)

    @points(3)
    @timeout(0.01)
    def test_prior_probability(self):
        """Prior class probabilities are computed correctly."""
        self.assertAlmostEqual(0.8666957485868004, self.classifier.prior_prob("y"))
        self.assertAlmostEqual(0.13330425141319954, self.classifier.prior_prob("n"))

    @points(3)
    @timeout(0.01)
    def test_likelihood_prob(self):
        """Likelihood probabilities are computed correctly."""
        self.assertAlmostEqual(
            0.47980707462129846,
            self.classifier.likelihood_prob("split_tok=.", "y"),
        )
        self.assertAlmostEqual(
            0.0034616739540377565,
            self.classifier.likelihood_prob("split_tok=!", "y"),
        )
        self.assertAlmostEqual(
            0.4487923621648963,
            self.classifier.likelihood_prob("split_tok=.", "n"),
        )
        self.assertAlmostEqual(
            0.005570308548305513,
            self.classifier.likelihood_prob("split_tok=!", "n"),
        )

    @points(3)
    @timeout(0.01)
    def test_log_posterior_probability_segmentation(self):
        """Posterior log-probabilities are computed correctly."""
        self.assertAlmostEqual(
            -0.8774384718891249,
            self.classifier.log_posterior_prob(["split_tok=."], "y"),
        )
        self.assertAlmostEqual(
            -5.809070293303792,
            self.classifier.log_posterior_prob(["split_tok=!"], "y"),
        )
        self.assertAlmostEqual(
            -2.8163161020012994,
            self.classifier.log_posterior_prob(["split_tok=."], "n"),
        )
        self.assertAlmostEqual(
            -7.205425990641993,
            self.classifier.log_posterior_prob(["split_tok=!"], "n"),
        )

    @points(2)
    @timeout(0.01)
    def test_classify(self):
        """Two candidate boundaries are classified correctly."""
        self.assertEqual(
            "y",
            self.classifier.classify(["left_tok=products", "split_tok=."]),
        )
        self.assertEqual("n", self.classifier.classify(["left_tok=Dr", "split_tok=."]))

    @points(3)
    @timeout(0.01)
    def test_naivebayes_test(self):
        """Naive Bayes classification works correctly."""
        result = self.classifier.test(
            [
                ClassificationInstance("y", ["left_tok=outstanding", "split_tok=."]),
                ClassificationInstance("y", ["left_tok=fairly", "split_tok=?"]),
                ClassificationInstance("n", ["left_tok=U.S", "split_tok=."]),
                ClassificationInstance("y", ["left_tok=!", "split_tok=!"]),
                ClassificationInstance("n", ["left_tok=Mx.", "split_tok=."]),
            ]
        )
        self.assertEqual(tuple, type(result))
        self.assertEqual(list, type(result[0]))
        self.assertEqual(list, type(result[1]))
        self.assertEqual(len(result[0]), len(result[1]))
        for item in result[0]:
            self.assertEqual(str, type(item))
        for item in result[1]:
            self.assertEqual(str, type(item))
        self.assertEqual((["y", "y", "n", "y", "y"], ["y", "y", "n", "y", "n"]), result)


class TestNaiveBayesSentiment(unittest.TestCase):
    classifier: NaiveBayesClassifier

    @classmethod
    def setUpClass(cls) -> None:
        # Create and train classifier
        feature_extractor = SentimentTestFeatureExtractor()
        train_instances = (
            feature_extractor.extract_features(inst)
            for inst in load_sentiment_instances(
                os.path.join(AIRLINE_SENTIMENT_DIR, "train.json")
            )
        )
        cls.classifier = NaiveBayesClassifier(0.1)
        cls.classifier.train(train_instances)

    @points(3)
    @timeout(0.01)
    def test_prior_probability(self):
        """Prior class probabilities are computed correctly."""
        self.assertAlmostEqual(
            0.20602253032928944, self.classifier.prior_prob("positive")
        )
        self.assertAlmostEqual(0.7939774696707106, self.classifier.prior_prob("negative"))

    @points(3)
    @timeout(0.01)
    def test_likelihood_prob(self):
        """Likelihood probabilities are computed correctly."""
        self.assertAlmostEqual(
            0.014438786335667303,
            self.classifier.likelihood_prob("thank", "positive"),
        )
        self.assertAlmostEqual(
            1.0608953957139827e-05,
            self.classifier.likelihood_prob("bad", "positive"),
        )
        self.assertAlmostEqual(
            0.0006067094932191293,
            self.classifier.likelihood_prob("thank", "negative"),
        )
        self.assertAlmostEqual(
            0.0005243507384835008,
            self.classifier.likelihood_prob("bad", "negative"),
        )

    @points(3)
    @timeout(0.01)
    def test_log_posterior_probability(self):
        """Posterior log-probabilities are computed correctly."""
        self.assertAlmostEqual(
            -5.817606943468801,
            self.classifier.log_posterior_prob(["thank"], "positive"),
        )
        self.assertAlmostEqual(
            -13.033581946120268,
            self.classifier.log_posterior_prob(["bad"], "positive"),
        )
        self.assertAlmostEqual(
            -7.638160669701965,
            self.classifier.log_posterior_prob(["thank"], "negative"),
        )
        self.assertAlmostEqual(
            -7.7840499431730885,
            self.classifier.log_posterior_prob(["bad"], "negative"),
        )

    @points(4)
    @timeout(0.01)
    def test_classify(self):
        """The tokens 'thank' and 'bad' are classified correctly and a string label is returned."""
        self.assertEqual("positive", self.classifier.classify(["thank"]))
        self.assertEqual("negative", self.classifier.classify(["bad"]))
        self.assertEqual(str, type(self.classifier.classify(["thank"])))


class TestPerformanceSegmentation(unittest.TestCase):
    train_instances: list[ClassificationInstance]
    dev_instances: list[ClassificationInstance]

    @classmethod
    def setUpClass(cls) -> None:
        # Create and train classifier
        feature_extractor = SegmentationTestFeatureExtractor()
        # We load the data into lists so the feature extraction time occurs here and not
        # when the tests are timed
        cls.train_instances = [
            feature_extractor.extract_features(inst)
            for inst in load_segmentation_instances(
                os.path.join(SENTENCE_SPLIT_DIR, "train.json")
            )
        ]
        # Load dev data
        cls.dev_instances = [
            feature_extractor.extract_features(inst)
            for inst in load_segmentation_instances(
                os.path.join(SENTENCE_SPLIT_DIR, "dev.json")
            )
        ]

    @points(4)
    @timeout(0.75)
    def test_segmentation_performance_y(self):
        """Segmentation performance is sufficiently good for the 'y' label."""
        classifier = NaiveBayesClassifier(2.0)
        classifier.train(self.train_instances)
        predicted, expected = classifier.test(self.dev_instances)
        acc, prec, rec, f1_score, report = classification_report(predicted, expected, "y")
        print("Baseline segmentation performance:")
        print(report)

        self.assertLessEqual(0.9860, acc)
        self.assertLessEqual(0.9848, prec)
        self.assertLessEqual(0.9994, rec)
        self.assertLessEqual(0.9920, f1_score)


class TestPerformanceSentiment(unittest.TestCase):
    dev_instances: list[ClassificationInstance]
    classifier: NaiveBayesClassifier

    @classmethod
    def setUpClass(cls) -> None:
        # Create and train classifier
        feature_extractor = SentimentTestFeatureExtractor()
        # We load the data into lists so the feature extraction time occurs here and not
        # when the tests are timed
        train_instances = [
            feature_extractor.extract_features(inst)
            for inst in load_sentiment_instances(
                os.path.join(AIRLINE_SENTIMENT_DIR, "train.json")
            )
        ]
        # Train
        cls.classifier = NaiveBayesClassifier(0.05)
        cls.classifier.train(train_instances)

        # Load dev data
        cls.dev_instances = [
            feature_extractor.extract_features(inst)
            for inst in load_sentiment_instances(
                os.path.join(AIRLINE_SENTIMENT_DIR, "dev.json")
            )
        ]

    @points(2)
    @timeout(0.1)
    def test_sentiment_performance_positive(self):
        """Baseline performance on sentiment classification is sufficiently good for the 'positive' label."""
        predicted, expected = self.classifier.test(self.dev_instances)
        acc, prec, rec, f1_score, report = classification_report(
            predicted, expected, "positive"
        )
        print("Baseline positive sentiment performance:")
        print(report)

        self.assertLessEqual(0.8542, acc)
        self.assertLessEqual(0.6598, prec)
        self.assertLessEqual(0.5687, rec)
        self.assertLessEqual(0.6109, f1_score)

    @points(2)
    @timeout(0.1)
    def test_sentiment_performance_negative(self):
        """Baseline performance on sentiment classification is good for the 'negative' label."""
        predicted, expected = self.classifier.test(self.dev_instances)
        acc, prec, rec, f1_score, report = classification_report(
            predicted, expected, "negative"
        )

        print("Baseline negative sentiment performance:")
        print(report)

        self.assertLessEqual(0.8542, acc)
        self.assertLessEqual(0.8949, prec)
        self.assertLessEqual(0.9260, rec)
        self.assertLessEqual(0.9102, f1_score)



class TestTunedAirlineSentiment(unittest.TestCase):
    @points(0)
    @timeout(1.0)
    def test_tuned_airline_sentiment(self):
        """Test that a valid value has been set for k and report performance."""
        extractor = TunedAirlineSentimentFeatureExtractor()
        # Check that the value for k has been changed from NaN
        self.assertFalse(math.isnan(extractor.k))
        # Check that the extractor inherits from a feature extractor (bigram or unigram)
        self.assertIsInstance(
            extractor,
            (
                UnigramAirlineSentimentFeatureExtractor,
                BigramAirlineSentimentFeatureExtractor,
            ),
        )

        sentiment_train_instances = (
            extractor.extract_features(inst)
            for inst in load_sentiment_instances(
                os.path.join(AIRLINE_SENTIMENT_DIR, "train.json")
            )
        )

        self.sentiment_dev_instances = [
            extractor.extract_features(inst)
            for inst in load_sentiment_instances(
                os.path.join(AIRLINE_SENTIMENT_DIR, "dev.json")
            )
        ]
        self.sentiment_classifier = NaiveBayesClassifier(extractor.k)
        self.sentiment_classifier.train(sentiment_train_instances)
        predicted, expected = self.sentiment_classifier.test(self.sentiment_dev_instances)
        for positive_label in self.sentiment_classifier.instance_counter.labels():
            acc, prec, rec, f1_score, report = classification_report(
                predicted, expected, positive_label
            )
            print(
                f"Tuned sentiment classification performance for k of "
                f"{extractor.k} for label {repr(positive_label)}:"
            )
            print(report)
            print()


def classification_report(
    predicted: list[str],
    expected: list[str],
    positive_label: str,
) -> tuple[float, float, float, float, str]:
    """Return accuracy, P, R, F1 and a classification report."""
    acc = accuracy_score(y_pred=predicted, y_true=expected)
    prec = precision_score(y_pred=predicted, y_true=expected, pos_label=positive_label)
    rec = recall_score(y_pred=predicted, y_true=expected, pos_label=positive_label)
    f1 = f1_score(y_pred=predicted, y_true=expected, pos_label=positive_label)
    report = "\n".join(
        [
            f"Accuracy:  {acc * 100:0.2f}",
            f"Precision: {prec * 100:0.2f}",
            f"Recall:    {rec * 100:0.2f}",
            f"F1:        {f1 * 100:0.2f}",
        ]
    )
    return acc, prec, rec, f1, report


class DefensiveIterable(Iterable[T]):
    def __init__(self, source: Iterable[T]):
        self.source: Iterable[T] = source

    def __iter__(self) -> Iterator[T]:
        return iter(self.source)

    def __len__(self):
        # This object should never be put into a sequence, so we sabotage the
        # __len__ function to make it difficult to do so. We specifically raise
        # ValueError because TypeError and NotImplementedError appear to be
        # handled by the list function.
        raise ValueError(
            "You cannot put this iterable into a sequence (list, tuple, etc.). "
            "Instead, iterate over it using a for loop."
        )


def main() -> None:
    tests = [
        TestScoringMetrics,
        TestSegmentationFeatureExtractor,
        TestAirlineSentimentUnigramFeatureExtractor,
        TestAirlineSentimentBigramFeatureExtractor,
        TestInstanceCounter,
        TestNaiveBayesSegmentation,
        TestNaiveBayesSentiment,
        TestPerformanceSegmentation,
        TestPerformanceSentiment,
        TestTunedAirlineSentiment,
    ]
    grader = Grader(tests)
    grader.print_results()


if __name__ == "__main__":
    main()
