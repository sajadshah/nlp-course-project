import os, io
import nltk
import random

import pickle


def load(data_file, labels_file = None):
    if labels_file == None:
        labels_file = data_file + '.labels'

    f = io.open(data_file, mode='r', encoding='utf-8')
    l = io.open(labels_file, mode='r', encoding='utf-8')
    sentences = []
    labels = []
    for sentence in f:
        sentences.append(sentence.strip())
    f.close()
    for label in l:
        labels.append(int(label.strip()))
    l.close()

    return sentences, labels


def create_train_test_data_files(data_file, labels_file, train_output, test_output):
    if os.path.exists(train_output) and os.path.exists(test_output):
        return

    sentences, labels = load(data_file, labels_file)
    indeces = range(0, len(sentences))
    random.shuffle(indeces)
    train_size = int(0.75 * len(sentences))
    trainf = io.open(train_output, mode='w', encoding='utf-8')
    trainl = io.open(train_output + '.labels', mode='w', encoding='utf-8')
    for i in range(train_size):
        trainf.write(sentences[indeces[i]] + '\n')
        trainl.write(unicode(labels[indeces[i]]) + '\n')
    trainf.close()

    testf = io.open(test_output, mode='w', encoding='utf-8')
    testl = io.open(test_output + '.labels', mode='w', encoding='utf-8')
    for i in range(train_size, len(sentences)):
        testf.write(sentences[indeces[i]] + '\n')
        testl.write(unicode(labels[indeces[i]]) + '\n')
    testf.close()

def loda_lexicon(lexicon_path):
    words = []
    if os.path.exists(lexicon_path):
        lf = io.open(lexicon_path, mode='r', encoding='utf-8')
        for w in lf:
            w = w.strip()
            if w not in words:
                words.append(w)
    return words



def create_lexicon_file(sentences_words, output_file):
    words = []
    if os.path.exists(output_file):
        lf = io.open(output_file, mode='r', encoding='utf-8')
        for w in lf:
            w = w.strip()
            if w not in words:
                words.append(w)
        return words

    for sentence_w in sentences_words:
        for w in sentence_w:
            if w not in words:
                words.append(w)

    lf = io.open(output_file, mode='w', encoding='utf-8')
    for w in words:
        lf.write(w + u'\n')
    lf.close()

    return words


def create_char_map_base(input_file='data.txt', output_file='char_map.txt'):
    if os.path.exists(output_file):
        return ;
    chars = []
    f = io.open(input_file,mode='r', encoding='utf-8')
    for sentence in f:
        for ch in sentence:
            if ch not in chars:
                chars.append(ch)
    f.close()
    chars = sorted(chars)
    char_map_file = io.open(output_file, mode="w", encoding='utf-8')
    for ch in chars:
        char_map_file.write(u'{}\n'.format(ch))
    char_map_file.close()

    return chars


def read_char_map(charmap_file):
    char_map = {}
    chf = io.open(charmap_file, mode="r", encoding='utf-8')
    for r in chf:
        if '-' in r:
            s, d = r.strip().split('-')
            char_map[s] = d
    return char_map


def read_remove_chars(remove_chars_file):
    remove_chars = []
    rmf = io.open(remove_chars_file, mode='r', encoding='utf-8')
    for r in rmf:
        remove_chars.append(r.strip())
    return remove_chars

def read_stop_words(stop_words_file):
    stop_words = []
    sf = io.open(stop_words_file, mode='r', encoding='utf-8')
    for r in sf:
        stop_words.append(r.strip())
    return stop_words


def normalize(sentences, charmap_file, remove_chars_file, output_file=None):
    char_map = read_char_map(charmap_file)
    remove_chars = read_remove_chars(remove_chars_file)

    normal_sentences = []
    for sentence in sentences:
        for s_char in char_map:
            if s_char in sentence:
                sentence = sentence.replace(s_char, char_map[s_char])
        for r_char in remove_chars:
            if r_char in sentence:
                sentence = sentence.replace(r_char, ' ')

        normal_sentences.append(sentence)
    if output_file != None:
        o = io.open(output_file, mode='w', encoding='utf-8')
        for sentence in normal_sentences:
            o.write(sentence)
        o.close()

    sentences_words = []
    for sentence in normal_sentences:
        words = nltk.word_tokenize(sentence)
        sentences_words.append(words)

    return sentences_words


def remove_stop_words(sentences_words, stop_words_file):
    stop_words = read_stop_words(stop_words_file)
    new_sentences = []
    for s_words in sentences_words:
        non_stop_words = []
        for w in s_words:
            w = w.lower()
            if w not in stop_words:
                non_stop_words.append(w)

        new_sentences.append(non_stop_words)
    return new_sentences


def stem(sentence_words):
    stemmer = nltk.PorterStemmer()
    new_sentences = []
    for words in sentence_words:
        new_words = []
        for w in words:
            n_w = stemmer.stem(w)
            new_words.append(n_w)
        new_sentences.append(new_words)
    return new_sentences

def preprocess(sentences, prefix, output_file=None):
    print('normalizing {} sentences'.format(len(sentences)))
    sentences_words = normalize(sentences, \
                          charmap_file=os.path.join(prefix, 'charmap.txt'), \
                          remove_chars_file=os.path.join(prefix, 'remove_chars.txt'), \
                          output_file=os.path.join(prefix, 'data_normal.txt'))
    print('removing stop words of {} sentences'.format(len(sentences)))
    sentence_words = remove_stop_words(sentences_words, \
                          stop_words_file=os.path.join(prefix, 'stop_words_short.txt'))

    print('stemming {} sentences'.format(len(sentence_words)))
    sentences_words = stem(sentence_words)

    print('removing stop words of {} sentences'.format(len(sentence_words)))
    sentence_words = remove_stop_words(sentences_words, \
                                       stop_words_file=os.path.join(prefix, 'stop_words_short.txt'))

    print('preparing to return loaded file')
    sentences = []
    if output_file is not None:
        of = io.open(os.path.join(prefix, output_file), mode='w', encoding='utf-8')
        for s_words in sentence_words:
            n_sentence = ' '.join(s_words)
            of.write(n_sentence + u'\n')
            sentences.append(n_sentence)
        of.close()
    else:
        for s_words in sentence_words:
            n_sentence = ' '.join(s_words)
            sentences.append(n_sentence)
    print('data loaded successfully')
    return sentence_words, sentences


def load_data_in_file(data_file, prefix):
    print('loading file {}'.format(data_file))
    temp_file_path = data_file + '_temp.tmp'

    if os.path.exists(temp_file_path):
        temp_file = open(temp_file_path, "rb")
        s_w = pickle.load(temp_file)
        s = pickle.load(temp_file)
        return s_w, s

    f = io.open(data_file, mode='r', encoding='utf-8')
    sentences = []
    for l in f:
        sentences.append(l)
    f.close()
    s_w, s = preprocess(sentences, prefix=prefix)

    temp_file = open(temp_file_path, "wb")
    pickle.dump(s_w, temp_file)
    pickle.dump(s, temp_file)
    return s_w, s


def iterate_on_data_in_file(data_file, prefix, batch_size):
    f = io.open(data_file, mode='r', encoding='utf-8')
    sentences = []
    for i, l in enumerate(f):
        sentences.append(l)
        if len(sentences) == batch_size:
            sen_, sen_words = preprocess(sentences, prefix=prefix)
            sentences = []
            yield sen_words
    f.close()
    sen_, sen_words = preprocess(sentences, prefix=prefix)
    yield sen_words


def simplecount(filename):
    lines = 0
    for line in open(filename):
        lines += 1
    return lines