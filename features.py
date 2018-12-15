import gensim
from gensim import corpora
from gensim.models import Word2Vec, Doc2Vec, LdaModel
from gensim.models.doc2vec import TaggedDocument, TaggedLineDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups

import numpy as np
import os
import pickle

import dataset

def normalize(train_data, test_data):
    train_mean = np.mean(train_data, axis=0)
    train_std = np.std(train_data, axis=0)

    return (train_data - train_mean) / train_std, (test_data-train_mean) / train_std

def make_one_hot(labels, n_classes):
    n = len(labels)
    labels = np.asarray(labels, dtype=int)
    labels_one_hot = np.zeros((n, n_classes), dtype=int)
    labels_one_hot[np.arange(n), labels] = 1
    return labels_one_hot

def tf_idf_features(sentences, lexicon, model_path, mode="train"):
    tf = TfidfVectorizer(vocabulary=lexicon)
    if mode == "train" and not os.path.exists(model_path):
        tf.fit(sentences)
        pickle.dump(tf, open(model_path, "wb"))

    tf= pickle.load(open(model_path, "rb"))
    tfidf_matrix = tf.transform(sentences)
    # feature_names = tf.get_feature_names()
    # doc = 0
    # feature_index = tfidf_matrix[doc, :].nonzero()[1]
    # tfidf_scores = zip(feature_index, [tfidf_matrix[doc, x] for x in feature_index])
    # for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
    #     print w, s

    return tfidf_matrix.toarray()

def lda_features_alt(sentences, lexicon, num_topics = 20, mode='train'):
    tf = TfidfVectorizer(vocabulary=lexicon)
    tfidf_matrix = tf.fit_transform(sentences)

    lda = LatentDirichletAllocation(n_topics=num_topics)
    lda_matrix = lda.fit_transform(tfidf_matrix)

    return lda_matrix


def lda_features(sentence_words, lexicon, model_path, num_topics = 50, mode='train', ):
    dictionary = corpora.Dictionary([[lex] for lex in lexicon])
    corpus = [dictionary.doc2bow(words) for words in sentence_words]
    if mode == 'train' and not os.path.exists(model_path):
        ldamodel = LdaModel(corpus, num_topics=num_topics)
        ldamodel.save(model_path)
    else :
        ldamodel = LdaModel.load(model_path)

    features = []
    for sentence in corpus:
        lda_f = ldamodel[sentence]
        feats = np.zeros((num_topics,))
        for (n_t, s_t) in lda_f:
            feats[n_t] = s_t
        features.append(feats)

    result = np.asarray(features)
    return result

def train_word2vec_feature(sentence_words) :
    model = Word2Vec(sentence_words, min_count=1)
    # summarize the loaded model
    print(model)
    # summarize vocabulary
    words = list(model.wv.vocab)
    print(words)
    # access vector for one word
    print(model['sentence'])
    # save model
    model.save('model.bin')
    # load model
    new_model = Word2Vec.load('model.bin')
    print(new_model)

def create_doc2vec_model_batch(train_d2v_data_file_path, vector_size, lexicon, files_prefix, model_path):
    print('Training d2v model')
    model = Doc2Vec(min_count=1, window=10, size=vector_size, workers=4)
    # print('Loading dataset')
    # train_sentence_words, train_sentences = dataset.load_data_in_file(train_d2v_data_file_path, prefix=files_prefix)

    vocab = [TaggedDocument([l], 'WORD_{}'.format(index)) for index, l in enumerate(lexicon)]
    model.build_vocab(vocab)

    total_docs = dataset.simplecount(train_d2v_data_file_path)

    for ep in range(10):
        index = 0
        iter_on_data = dataset.iterate_on_data_in_file(train_d2v_data_file_path, files_prefix, 10000)
        for sentence_words in iter_on_data:
            documents = []
            for s_words in sentence_words:
                td = TaggedDocument(s_words, 'DOC_{}'.format(index))
                index += 1
                documents.append(td)

            print('Training model {} - {} / {}'.format(ep, index, total_docs))
            model.train(documents, total_examples=total_docs)
            model.save(model_path)

    return model


def create_doc2vec_model(train_d2v_data_file_path, vector_size, lexicon, files_prefix, model_path):
    print('Training d2v model')
    model = Doc2Vec(min_count=1, window=10, size=vector_size, workers=4)
    print('Loading dataset')
    sentence_words, _ = dataset.load_data_in_file(train_d2v_data_file_path, prefix=files_prefix)
    print('creating TaggedDocuments')
    documents = []
    for index, s_words in enumerate(sentence_words):
        td = TaggedDocument(s_words, 'DOC_{}'.format(index))
        documents.append(td)

    print('Building vocabulary')
    model.build_vocab(documents)

    total_docs = dataset.simplecount(train_d2v_data_file_path)

    for ep in range(1):
        print('Training model {} - {} / {}'.format(ep, index, total_docs))
        model.train(documents, total_examples=total_docs)
        model.save(model_path)

    return model

def create_doc2vec_model_v2(train_d2v_data_file_path, vector_size, lexicon, files_prefix, model_path):
    print('Training d2v model')
    docs = TaggedLineDocument(train_d2v_data_file_path)
    model = Doc2Vec(docs, size=vector_size, window=10, min_count=1, workers=8)
    model.save(model_path)

    return model

def doc2vec_features(sentence_words, lexicon, vector_size=100, model_path= None, mode="train", train_d2v_data_file_path = None, files_prefix = None) :
    if mode == "train" and not os.path.exists(model_path):
        d2v_model = create_doc2vec_model_v2(train_d2v_data_file_path, vector_size, lexicon, files_prefix, model_path)
    else :
        d2v_model = Doc2Vec.load(model_path)
    print('d2v model loaded from {}'.format(model_path))
    features = []
    for s_words in sentence_words:
        features.append(d2v_model.infer_vector(s_words, alpha=0.01, steps=1000))

    result = np.asarray(features)
    print('features ready shape: {}'.format(result.shape))
    return result
