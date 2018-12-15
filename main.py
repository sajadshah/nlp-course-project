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
    test_data_file_path = os.path.join(files_prefix, 'test.txt')

    lexicon_path = os.path.join(files_prefix, 'lexicon.txt')

    # charmap_path = os.path.join(files_prefix, 'charmap.txt')
    # dataset.create_char_map_base(input_file=data_file_path, output_file=charmap_path)

    dataset.create_train_test_data_files(data_file_path, labels_file_path, train_data_file_path, test_data_file_path)

    sentences, labels = dataset.load(data_file=train_data_file_path)
    sentence_words, sentences = dataset.preprocess(sentences, prefix=files_prefix, output_file='data_train_final.txt')
    one_hot_labels = features.make_one_hot(labels, 2)
    lexicon = dataset.create_lexicon_file(sentence_words, output_file=lexicon_path)

    test_sentences, test_labels = dataset.load(data_file=test_data_file_path)
    test_sentence_words, test_sentences = dataset.preprocess(test_sentences, prefix=files_prefix, output_file='data_test_final.txt')
    test_one_hot_labels = features.make_one_hot(test_labels, 2)

    if args.feature_type == "tfidf":
        tfidf_model_file_path = os.path.join(files_prefix, 'tf_idf.model')
        tf_idf_features = features.tf_idf_features(sentences, lexicon, model_path=tfidf_model_file_path, mode="train")
        test_tf_idf_features = features.tf_idf_features(test_sentences, lexicon, model_path=tfidf_model_file_path, mode="test")

        predicted_labels = models.logistic_regression(tf_idf_features, one_hot_labels, test_tf_idf_features, test_one_hot_labels, lexicon , \
                                                   files_prefix='tf_idf_', learning_rate=0.0001, learning_rate_decay=0.5, learning_rate_decay_yellows=5,\
                                                   max_yellow_cards=10, batch_size=100, training_epochs=200, n_hidden=16)
    elif args.feature_type == "lda":
        # lda_features = features.lda_features_alt(sentences, lexicon, lda_model_file_path, num_topics=10, mode='train')
        # test_lda_features = features.lda_features_alt(test_sentences, lexicon, lda_model_file_path, num_topics=10, mode='test')

        num_topics = 1500
        lda_model_file_path = os.path.join(files_prefix, 'lda_{}.model'.format(num_topics))
        lda_features = features.lda_features(sentence_words, lexicon, lda_model_file_path, num_topics=num_topics, mode='train')
        test_lda_features = features.lda_features(test_sentence_words, lexicon, lda_model_file_path, num_topics=num_topics, mode='test')
        predicted_labels = models.logistic_regression(lda_features, one_hot_labels, test_lda_features, test_one_hot_labels, lexicon, \
                                                    files_prefix='lda_{}'.format(num_topics), learning_rate=0.0001, \
                                                    learning_rate_decay=0.9, learning_rate_decay_yellows=1, \
                                                    max_yellow_cards=20, batch_size=20, training_epochs=500, n_hidden=1024)
    elif args.feature_type == "doc2vec":
        vector_size = 100
        # d2v_model_file_path = os.path.join(files_prefix, 'doc2vec_{}.model'.format(vector_size))
        d2v_model_file_path = os.path.join(files_prefix, 'apnews_dbow', 'doc2vec.bin')
        d2v_temp_features_file_path = d2v_model_file_path + '_data.tmp'
        if False and os.path.exists(d2v_temp_features_file_path):
            temp_file = open(d2v_temp_features_file_path, "rb")
            d2v_features = pickle.load(temp_file)
            test_d2v_features = pickle.load(temp_file)
        else :
            train_d2v_file_path = os.path.join(files_prefix, 'd2v_train.txt'.format(vector_size))
            d2v_features = features.doc2vec_features(sentence_words, lexicon, vector_size=vector_size, mode="train", \
                                                 model_path=d2v_model_file_path, \
                                                 train_d2v_data_file_path=train_d2v_file_path, \
                                                 files_prefix=files_prefix)

        test_d2v_features = features.doc2vec_features(test_sentence_words, lexicon, vector_size=vector_size, mode="test", \
                                                 model_path=d2v_model_file_path, \
                                                 train_d2v_data_file_path=train_d2v_file_path, \
                                                 files_prefix=files_prefix)

        temp_file = open(d2v_temp_features_file_path, "wb")
        pickle.dump(d2v_features, temp_file)
        pickle.dump(test_d2v_features, temp_file)

        predicted_labels = models.logistic_regression(d2v_features, one_hot_labels, test_d2v_features, test_one_hot_labels,
                                               lexicon, \
                                               files_prefix='d2v_{}'.format(vector_size), learning_rate=0.0001, \
                                               learning_rate_decay=0.9, learning_rate_decay_yellows=2, \
                                               max_yellow_cards=20, batch_size=100, training_epochs=500000, n_hidden=16)

    elif args.feature_type == "all":
        tfidf_model_file_path = os.path.join(files_prefix, 'tf_idf.model')
        tf_idf_features = features.tf_idf_features(sentences, lexicon, model_path=tfidf_model_file_path, mode="train")
        test_tf_idf_features = features.tf_idf_features(test_sentences, lexicon, model_path=tfidf_model_file_path, mode="test")

        num_topics = 1500
        lda_model_file_path = os.path.join(files_prefix, 'lda_{}.model'.format(num_topics))
        lda_features = features.lda_features(sentence_words, lexicon, lda_model_file_path, num_topics=num_topics, mode='train')
        test_lda_features = features.lda_features(test_sentence_words, lexicon, lda_model_file_path, num_topics=num_topics, mode='test')

        vector_size = 100
        d2v_model_file_path = os.path.join(files_prefix, 'doc2vec_{}.model'.format(vector_size))
        # d2v_model_file_path = os.path.join(files_prefix, 'apnews_dbow', 'doc2vec.bin')
        d2v_temp_features_file_path = d2v_model_file_path + '_data.tmp'
        if os.path.exists(d2v_temp_features_file_path):
            temp_file = open(d2v_temp_features_file_path, "rb")
            d2v_features = pickle.load(temp_file)
            test_d2v_features = pickle.load(temp_file)
        else:
            train_d2v_file_path = os.path.join(files_prefix, 'd2v_train.txt'.format(vector_size))
            d2v_features = features.doc2vec_features(sentence_words, lexicon, vector_size=vector_size, mode="train", \
                                                     model_path=d2v_model_file_path, \
                                                     train_d2v_data_file_path=train_d2v_file_path, \
                                                     files_prefix=files_prefix)

        test_d2v_features = features.doc2vec_features(test_sentence_words, lexicon, vector_size=vector_size,
                                                      mode="test", \
                                                      model_path=d2v_model_file_path, \
                                                      train_d2v_data_file_path=train_d2v_file_path, \
                                                      files_prefix=files_prefix)

        features = np.concatenate((tf_idf_features, lda_features, d2v_features), axis=1)
        test_features = np.concatenate((test_tf_idf_features, test_lda_features, test_d2v_features), axis=1)

        predicted_labels = models.logistic_regression(d2v_features, one_hot_labels, test_d2v_features, test_one_hot_labels,
                                                   lexicon, \
                                                   files_prefix='all_', learning_rate=0.0001, \
                                                   learning_rate_decay=0.9, learning_rate_decay_yellows=2, \
                                                   max_yellow_cards=20, batch_size=100, training_epochs=500000,
                                                   n_hidden=16)

    with open(os.path.join(files_prefix, 'test_predicted_labels'), 'w') as f:
        for pl in predicted_labels:
            f.write('{}\n'.format(pl))