from sklearn import preprocessing
import re
import spacy
import sklearn.feature_extraction.text
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


class CategoricalAnalyzer:
    def __call__(self, value):
        assert not isinstance(value, list), 'don\'t give me a list!'
        return [value]


class MulticategoricalAnalyzer:
    def __call__(self, value):
        assert isinstance(value, list), 'hey! give me a list!'
        return value


class FeaturePipeline(Pipeline):

    def get_feature_names(self):
        name, trans = self.steps[-1]
        if not hasattr(trans, 'get_feature_names'):
            raise AttributeError('Transformer %s (type %s) does not '
                                 'provide get_feature_names.' % (str(name), type(trans).__name__))
        return trans.get_feature_names()


class ColumnSelector(BaseEstimator, TransformerMixin):

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.columns]
        elif isinstance(X, pd.Series):
            if isinstance(self.columns, list):
                return pd.DataFrame([
                    [X[column] for column in self.columns]
                ], columns=self.columns, index=[X.name])
            else:
                return pd.Series([X[self.columns]], name=self.columns, index=[X.name])
        else:
            raise ValueError('Hey! Give me either a DataFrame (table) or a Series (row)!')

    def get_feature_names(self):
        if isinstance(self.columns, list):
            return self.columns
        else:
            return [self.columns]

class Selector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on text columns in the data
    """

    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]


class SingleCatSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on text columns in the data
    """

    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key].map(lambda v: [v])


class NumberSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on numeric columns in the data
    """

    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.key]]

    def get_feature_names(self):
        return [self.key]


class PipelineEx(Pipeline):

    def get_feature_names(self):
        """Get feature names from the last step.

        Returns
        -------
        feature_names : list of strings
            Names of the features produced by transform.
        """
        name, trans = self.steps[-1]
        if not hasattr(trans, 'get_feature_names'):
            raise AttributeError("Transformer %s does not provide"
                                 " get_feature_names." % str(name))
        return trans.get_feature_names()


class StandardScalerEx(preprocessing.StandardScaler):

    def __init__(self, key):
        self.key = key
    #
    # def fit(self, X, y=None):
    #     return self
    #
    # def transform(self, X):
    #     return X[[self.key]]

    def get_feature_names(self):
        return [self.key]


class MultiLabelBinarizerPipelineFriendly(preprocessing.MultiLabelBinarizer):
    def fit(self, X, y=None):
        """this would allow us to fit the model based on the X input."""
        super(MultiLabelBinarizerPipelineFriendly, self).fit(X)

    def transform(self, X, y=None):
        return super(MultiLabelBinarizerPipelineFriendly, self).transform(X)

    def fit_transform(self, X, y=None):
        return super(MultiLabelBinarizerPipelineFriendly, self).fit(X).transform(X)

    def get_feature_names(self):
        # for c in self.classes:
        #     try:
        #         unicode(c)
        #     except:
        #         print c
        return [unicode(c) for c in self.classes]


class Atomizer:
    _TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")
    _ALL_DIGITS_PATTERN = re.compile('\d+')
    _spacy_nlp = spacy.load('en')

    def __init__(self, ngram_range=(1, 1), stop_words=None, boost_terms=None, map_terms=None, preprocessor=None):
        self.ngram_range = ngram_range
        self.stop_words = stop_words
        self.boost_terms = boost_terms
        self.map_terms = map_terms  # TODO not implemented
        self.preprocessor = preprocessor

    def __call__(self, raw_document):
        ngrams = self._build_ngrams(self._tokenize(self._preprocess(raw_document)))
        if self.boost_terms is not None:
            return self._boost(ngrams)
        else:
            return ngrams

    @classmethod
    def normalize_token(cls, token):
        if token.startswith('_'):
            return token
        else:
            return cls._lemmatize(token.lower())

    @classmethod
    def normalize_multitoken(cls, multitoken):
        return ' '.join([
            cls.normalize_token(token)
            for token in multitoken.split(' ')
        ])

    @classmethod
    def _lemmatize(cls, token):
        return cls._spacy_nlp(unicode(token))[0].lemma_

    def _preprocess(self, document):
        if self.preprocessor is not None:
            return self.preprocessor(document)
        else:
            return document

    def _tokenize(self, document):
        return [self.normalize_token(token)
                for token
                in self._TOKEN_PATTERN.findall(document)
                if self._accept_token(token)]

    def _boost(self, stringified_ngrams):
        boosted_ngrams = []

        # upweighting: counting a term as if it occurred multiple times
        for ngram in stringified_ngrams:
            for i in xrange(0, self.boost_terms.get(ngram, 1)):
                boosted_ngrams.append(ngram)

        return boosted_ngrams

    def _accept_token(self, token):

        # reject too short tokens
        if len(token) == 1: return False

        # reject standalone numbers
        if self._ALL_DIGITS_PATTERN.match(token): return False

        return True

    def _accept_ngram(self, ngram):

        # drop n-grams containing stop words
        if '-STOP-' in ngram: return False

        # drop n-grams containing pronouns
        if '-PRON-' in ngram: return False

        # drop "doubles"
        if len(ngram) == 2:
            if ngram[0] == ngram[1]: return False

        return True

    def _build_ngrams(self, tokens):
        """
        Turn tokens into a sequence of n-grams
        """

        # handle stop words
        if self.stop_words is not None:
            tokens = [w
                      if w not in self.stop_words
                      else '-STOP-'
                      for w in tokens]

        stringified_ngrams = []

        min_n, max_n = self.ngram_range
        if max_n == 1:
            # handle unigrams
            for token in tokens:
                if self._accept_ngram([token]):
                    stringified_ngrams.append(token)
        else:
            # handle n-grams
            n_tokens = len(tokens)
            for n in xrange(min_n,
                            min(max_n + 1, n_tokens + 1)):
                for i in xrange(n_tokens - n + 1):
                    ngram = tokens[i: i + n]
                    if self._accept_ngram(ngram):
                        stringified_ngrams.append(' '.join(ngram))

        return stringified_ngrams


def pipe_feature_processing(categorical_data_columns, multicategorical_data_columns,
                            numeric_data_columns,
                            categorical=True, multicategorical=True, numeric=True):

    features = []

    if categorical == True:
        for col in categorical_data_columns:
            pipe_col = PipelineEx([
                ('selector', Selector(key=col)),
                ('label_binarizer', CountVectorizer(analyzer=CategoricalAnalyzer(), binary=True, min_df=5)),
            ])
            features.append((col, pipe_col))

    if multicategorical == True:
        for col in multicategorical_data_columns:
            pipe_col = PipelineEx([
                ('selector', Selector(key=col)),
                ('label_binarizer', CountVectorizer(analyzer=MulticategoricalAnalyzer(), binary=True, min_df=5)),
            ])
            features.append((col, pipe_col))

    if numeric == True:
        for col in numeric_data_columns:
            pipe_col = PipelineEx([
                ('selector', NumberSelector(key=col)),
                # ('polinomial', PolynomialFeatures(degree=2)),
                # ('scaler', StandardScaler())
            ])
            features.append((col, pipe_col))

    return features

description = FeaturePipeline([
                ('selector', ColumnSelector('description')),
                ('vec', TfidfVectorizer(
                    min_df=0.01, max_df=0.5,
                    analyzer=Atomizer(
                        ngram_range=(1, 2),
                        stop_words=sklearn.feature_extraction.text.ENGLISH_STOP_WORDS
                    )))
            ])

proj_description = FeaturePipeline([
                ('selector', ColumnSelector('proj_description')),
                ('vec', TfidfVectorizer(
                    min_df=0.01, max_df=0.5,
                    analyzer=Atomizer(
                        ngram_range=(1, 2),
                        stop_words=sklearn.feature_extraction.text.ENGLISH_STOP_WORDS
                    )))
            ])

position_name = FeaturePipeline([
                ('selector', ColumnSelector('position_name')),
                ('vec', TfidfVectorizer(
                    min_df=5, max_df=0.5,
                    analyzer=Atomizer(
                        ngram_range=(1, 2),
                        stop_words=sklearn.feature_extraction.text.ENGLISH_STOP_WORDS
                    )))
            ])

comments = FeaturePipeline([
                ('selector', ColumnSelector('comments')),
                ('vec', TfidfVectorizer(
                    min_df=5, max_df=0.5,
                    analyzer=Atomizer(
                        ngram_range=(1, 2),
                        stop_words=sklearn.feature_extraction.text.ENGLISH_STOP_WORDS
                    )))
            ])

def features(categorical_data_columns, multicategorical_data_columns, numeric_data_columns):

    features = FeatureUnion([('description', description),
                      ('proj_description', proj_description),
                      ('comments', comments),
                      ('position_name', position_name)] +
                        pipe_feature_processing(
                                categorical_data_columns=categorical_data_columns,
                                multicategorical_data_columns=multicategorical_data_columns,
                                numeric_data_columns=numeric_data_columns))
    return features


def pipe_feature_processing_new(categorical_data_columns, multicategorical_data_columns,
                            categorical=True, multicategorical=True):

    features = []
    if categorical == True:
        for col in categorical_data_columns:
            pipe_col = FeaturePipeline([
                ('selector', ColumnSelector(col)),
                ('label_binarizer', CountVectorizer(analyzer=CategoricalAnalyzer(), binary=True, min_df=5)),
            ])
            features.append((col, pipe_col))

    if multicategorical == True:
        for col in multicategorical_data_columns:
            pipe_col = FeaturePipeline([
                ('selector', ColumnSelector(col)),
                ('label_binarizer', CountVectorizer(analyzer=MulticategoricalAnalyzer(), binary=True, min_df=5)),
            ])
            features.append((col, pipe_col))

    return features

def new_features(categorical_data_columns, multicategorical_data_columns, numeric_data_columns):
    features = FeatureUnion([('description', description),
                      ('proj_description', proj_description),
                      ('position_name', position_name),
                      ('comments', comments),
                      ('numeric', ColumnSelector(numeric_data_columns))] +
                        pipe_feature_processing_new(
                                categorical_data_columns=categorical_data_columns,
                                multicategorical_data_columns=multicategorical_data_columns))
    return features