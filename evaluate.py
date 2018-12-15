import argparse
import os

import pickle

import dataset
import features
import models
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--feature-type',
        choices=['tfidf', 'lda', 'doc2vec', 'all'],
        type=str,
        default='tfidf',
        help='features type to be extracted')


    args = parser.parse_args()

    files_prefix = os.path.join(os.path.curdir, 'sst')
    data_file_path = os.path.join(files_prefix, 'data.txt')
    labels_file_path = os.path.join(files_prefix, 'label.txt')

    train_data_file_path = os.path.join(files_prefix, 'train.txt')
    test_data_file_path = os.path.join(files_prefix, 'test_real.txt')

    lexicon_path = os.path.join(files_prefix, 'lexicon.txt')

    lexicon = dataset.loda_lexicon(lexicon_path)

    test_sentences, test_labels = dataset.load(data_file=test_data_file_path)
    test_sentence_words, test_sentences = dataset.preprocess(test_sentences, prefix=files_prefix, output_file='data_test_final.txt')
    test_one_hot_labels = features.make_one_hot(test_labels, 2)

    if args.feature_type == "tfidf":
        tfidf_model_file_path = os.path.join(files_prefix, 'tf_idf.model')
        test_tf_idf_features = features.tf_idf_features(test_sentences, lexicon, model_path=tfidf_model_file_path, mode="test")

        prediected_labels = models.logistic_regression(None, None,  test_tf_idf_features, test_one_hot_labels, lexicon , \
                                                   files_prefix='tf_idf_', n_hidden=16, train=False)
    elif args.feature_type == "lda":
        num_topics = 1500
        lda_model_file_path = os.path.join(files_prefix, 'lda_{}.model'.format(num_topics))

        test_lda_features = features.lda_features(test_sentence_words, lexicon, lda_model_file_path, num_topics=num_topics, mode='test')
        prediected_labels = models.logistic_regression(None, None, test_lda_features, test_one_hot_labels, lexicon, \
                                                    files_prefix='lda_{}'.format(num_topics), train=False)
    elif args.feature_type == "doc2vec":
        vector_size = 100
        # d2v_model_file_path = os.path.join(files_prefix, 'doc2vec_{}.model'.format(vector_size))
        d2v_model_file_path = os.path.join(files_prefix, 'apnews_dbow', 'doc2vec.bin')
        train_d2v_file_path = os.path.join(files_prefix, 'd2v_train.txt'.format(vector_size))

        test_d2v_features = features.doc2vec_features(test_sentence_words, lexicon, vector_size=vector_size, mode="test", \
                                                 model_path=d2v_model_file_path, \
                                                 train_d2v_data_file_path=train_d2v_file_path, \
                                                 files_prefix=files_prefix)


        prediected_labels = models.logistic_regression(None, None, test_d2v_features, test_one_hot_labels,
                                               lexicon, \
                                               files_prefix='d2v_{}'.format(vector_size), train=False)

    elif args.feature_type == "all":
        tfidf_model_file_path = os.path.join(files_prefix, 'tf_idf.model')
        test_tf_idf_features = features.tf_idf_features(test_sentences, lexicon, model_path=tfidf_model_file_path, mode="test")

        num_topics = 1500
        lda_model_file_path = os.path.join(files_prefix, 'lda_{}.model'.format(num_topics))
        test_lda_features = features.lda_features(test_sentence_words, lexicon, lda_model_file_path, num_topics=num_topics, mode='test')

        vector_size = 100
        d2v_model_file_path = os.path.join(files_prefix, 'doc2vec_{}.model'.format(vector_size))
        # d2v_model_file_path = os.path.join(files_prefix, 'apnews_dbow', 'doc2vec.bin')
        train_d2v_file_path = os.path.join(files_prefix, 'd2v_train.txt'.format(vector_size))

        test_d2v_features = features.doc2vec_features(test_sentence_words, lexicon, vector_size=vector_size,
                                                      mode="test", \
                                                      model_path=d2v_model_file_path, \
                                                      train_d2v_data_file_path=train_d2v_file_path, \
                                                      files_prefix=files_prefix)

        test_features = np.concatenate((test_tf_idf_features, test_lda_features, test_d2v_features), axis=1)

        prediected_labels = models.logistic_regression(None, None, test_d2v_features, test_one_hot_labels,
                                                   lexicon, \
                                                   files_prefix='all_', train=False)

    for i, pl in enumerate(prediected_labels):
        print pl, test_sentences[i]