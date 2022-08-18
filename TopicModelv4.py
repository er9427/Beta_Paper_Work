import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import os
from io import StringIO
import pandas as pd
import tomotopy as tp
import math
import re
from collections import Counter
import sys
from matplotlib import pyplot as plt
import fsspec

WORD = re.compile(r"\w+")

#Our stopwords, stemmer, and lemmatizer
stop_words = stopwords.words('english')
stop_words.extend(['I', "I'm", "I've", "I'd", "<br />", "<br>"])
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
round_to = 4
dpth_hlda = 4

# Text Pre-processing
def preprocessing(sentence):
    sentence = re.sub('[^A-Za-z\s]+', '', str(sentence).lower())
    word_tokens = word_tokenize(sentence)
    filtered_sentence = ''
    review_sentence = ''
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence = filtered_sentence + ' ' + lemmatizer.lemmatize(w)
    rset = word_tokenize(filtered_sentence)
    rset = nltk.pos_tag(rset)
    for i in range(len(rset)):
        if rset[i][1] == 'NN' or rset[i][1] == 'NNS' or rset[i][1] == 'NNP' or \
                rset[i][1] == 'NNPS':
            review_sentence = review_sentence + ' ' + rset[i][0]
    return review_sentence

#Converting Topic-to-Sentence
def convert_topic_sentence(topic):
    topic_sentence = ''
    for i in range(len(topic)):
        topic_sentence = topic_sentence + ',' + topic[i][0]
    return topic_sentence

#Review Vector Dist
def text_to_vector(text):
    words = WORD.findall(str(text))
    return Counter(words)

#Cosine similarity between review and hLDA topics
def cosine_similarity(vec1, vec2): #issue of freq
    intersection = set(vec1.keys()) & set(vec2.keys())
    # print(intersection)
    numerator = sum([vec1[x] * vec2[x] for x in intersection]) #review.text
    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

#Hierarchical Latent Dirichlet Allocation
def hlda():
    print('Step 2: Implementing Hierarchical Latent Dirichlet Allocation for the corpus')
    mdl = tp.HLDAModel(tw = tp.TermWeight.PMI, depth = dpth_hlda, seed = 724)
    corpus = pd.read_csv('stage/corpus.csv')
    for line in corpus['Reviews']:
        ch = str(line).strip().split()
        mdl.add_doc(ch)
    print('Added Docs')

    for i in range(0,100,10):
        mdl.train(10)
    print('Model trained')
    mdl.save('stage/Hierarchical_Topic_Model.bin')
    print('Step 3: HLDA model saved to Hierarchical_Topic_Model.bin')


# Function performing text normalization and distribution for reviews and hLDA on the whole corpus
def create_corpus():
    corpus_df = pd.DataFrame(columns = ['Reviews'])
    amazon_reviews = pd.read_csv('input/input.csv', header = 0)
    amazon_reviews["Topic Word Distribution per review"] = ""
    amazon_reviews["Topic Words Only"] = ""
    print('Step 1: Creating a Latent Dirichlet Allocation Topic Model for each Pre-Processed Review')
    for k in range(len(amazon_reviews)):
        rev = amazon_reviews.loc[k,"reviews.text"]
        topic_dist = []
        count = []
        tot = 0
        rev = preprocessing(rev)
        rev = rev.strip() # because of space in beginning
        corpus_df = corpus_df.append({'Reviews': rev}, ignore_index=True)
        rev = rev.split(' ')
        for st in rev: #create topic distribution
            if st not in topic_dist:
                topic_dist.append(st)
                count.append(1)
            else:
                ind = topic_dist.index(st)
                count[ind] += 1
        tot = sum(count)
        for c, cn in enumerate(count):
            count[c] = cn/tot
        combined = [topic_dist[i] + ", " + str(round(count[i], round_to)) for i, j in enumerate(topic_dist)]
        
        amazon_reviews.loc[k,"Topic Word Distribution per review"] = str(combined)
        amazon_reviews.loc[k,"Topic Words Only"] = str(topic_dist)

    corpus_df.to_csv(r'stage/corpus.csv', header='Review', index=None, sep='\n',mode='w')
    amazon_reviews.to_csv(r'stage/Amazon_reviews_topics.csv', index=None, sep=',',mode='w')
    hlda()

#Loading the model and finding cosine similarity of every review with the lowest level topic from the "table"
def load_model():
    amazon_reviews = pd.read_csv('stage/Amazon_reviews_topics.csv')
    reviews_df = pd.DataFrame(columns=["Level 3 topics", "Level 3 labels","Level 2 topics", "Level 2 labels", "Level 1 topics", "Level 1 labels"])
    mdl = tp.HLDAModel.load('stage/Hierarchical_Topic_Model.bin')
    print("Step 5: Model Loaded")
    extractor = tp.label.PMIExtractor()
    cands = extractor.extract(mdl)
    labeler = tp.label.FoRelevance(mdl, cands, smoothing=1e-2, mu=0.25)

    print("Step 6: Reading Amazon Reviews and finding the cosine similarity with topics at the lowest level") #term words cos sim, vector matching per review and level 4
    for review in amazon_reviews['Topic Word Distribution per review']:
    # for review in amazon_reviews['reviews.text']:
         max_cosine = 0
         max_sim_topic_id = 0
         for j in range(mdl.k):
             if mdl.level(j) == mdl.depth - 1:
                 topic = mdl.get_topic_words(j)
                 topic_sentence = convert_topic_sentence(topic)
                 sim = cosine_similarity(text_to_vector(review), text_to_vector(topic_sentence)) #cosine sim algorithm
                 if max_cosine <= sim:
                     max_cosine = sim
                     max_sim_topic_id = j
         level = mdl.depth - 1
         parent_nodes = []
         parent_nodes.append(max_sim_topic_id)
         while level >= 0:
             imm_parent = mdl.parent_topic(parent_nodes[-1])
             parent_nodes.append(imm_parent)
             level = level - 1
         parent_nodes.pop(0) # do not want level 4 topics

         topic_label = []
         for i in range(len(parent_nodes) - 1):
            topict = ''
            topict = mdl.get_topic_words(parent_nodes[i])
            for count, j in enumerate(topict):
                temptop = []
                temptup = (j)
                temptop = list(temptup)
                temptop[1] = round(temptop[1], round_to)
                topict[count] = tuple(temptop)
            topic_label.append(topict)
            ltopict = labeler.get_topic_labels(parent_nodes[i])
            for count2, k in enumerate(ltopict):
                ltemptop = []
                ltemptup = (k)
                ltemptop = list(ltemptup)
                ltemptop[1] = round(ltemptop[1], round_to)
                ltopict[count2] = tuple(ltemptop)
            # topic_label.append(mdl.get_topic_words(parent_nodes[i]))
            topic_label.append(ltopict)
            # topic_label.append(labeler.get_topic_labels(parent_nodes[i]))
         reviews_df_length = len(reviews_df)
         reviews_df.loc[reviews_df_length] = topic_label

    result = pd.concat([amazon_reviews, reviews_df], axis = 1)
    print("Step 7: Process finished")
    result.to_csv(r'output/Amazon_reviews_topics.csv', index=None, sep=',')

#Summary of the HLDA model for topic dimension
def summarize_model():
    mdl = tp.HLDAModel.load('stage/Hierarchical_Topic_Model.bin')
    mdl.summary()
    mo = mdl.perplexity
    print(mo)
    with open('perplexity.txt', 'w')  as f:
        f.write(str(mo))


create_corpus()
load_model()
summarize_model() #enable for summary of topics like in Part 1

