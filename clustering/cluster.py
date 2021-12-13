
import nltk
import numpy as np
import torch
import pickle
import os
import time

from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
from argparse import ArgumentParser
from allennlp.predictors.predictor import Predictor

# Compute Document ID to Text Mapping, given parsed gigaword entries
def load_documents(news_prefixes, gigaword_path):
    doc_to_text = {}
    for prefix in news_prefixes:
        flat_dir = os.path.join(gigaword_path, "data/" + prefix + "_eng_flat/")
        monthly_dirs = os.listdir(flat_dir)
        for dir in tqdm(monthly_dirs):
            dir_path = os.path.join(flat_dir, dir)
            files = os.listdir(dir_path)
            for f in files:
                with open(os.path.join(dir_path, f), "r") as doc:
                    doc_to_text[f] = ""
                    for line in doc:
                        doc_to_text[f] += line.strip() + " "
    return doc_to_text

# Split text into corpus of sentences, retain mapping of sentence to document id
def generate_corpus(doc_to_text):
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    corpus = []
    corpusidx_to_doc = {}
    idx = 0
    for doc_id, text in tqdm(doc_to_text.items()):
        sentences = sent_detector.tokenize(text)
        for s in sentences:
            corpus.append(s) # add each sentence to the overall corpus
            corpusidx_to_doc[idx] = doc_id # map sentence back to its doc
            idx += 1
    return corpus, corpusidx_to_doc

def corpus_idx_to_idx_in_doc(corpus_idx, corpusidx_to_doc):
    docid = corpusidx_to_doc[corpus_idx]
    i = corpus_idx - 1
    while i >= 0:
        if corpusidx_to_doc[i] != docid:
            return corpus_idx - i - 1
        i -= 1
    return -1

def load_corpus_embeddings(corpus_emb_path):
    corpus_dir = os.path.join(args.corpus_load_path)
    files = os.listdir(corpus_dir)
    for f in files:
        if 'corpus_emb' not in f:
            continue
        print(f)
        # corpus_embeddings = torch.load(os.path.join(dir_path, f), map_location=torch.device('cpu'))

def get_stopword_list():
    vectorizer = TfidfVectorizer(stop_words='english', use_idf=False, ngram_range =(1, 2))
    sw_list = list(vectorizer.get_stop_words())
    extra_stopwords = ['said', 'says', 'did', 'like']
    sw_list.extend(extra_stopwords)
    return sw_list

def get_keywords_for_doc(doc, top_k, sw_list):
    vectorizer = TfidfVectorizer(stop_words=sw_list, use_idf=False, ngram_range =(1, 2))
    doc_tf_vector = vectorizer.fit_transform([doc])
    doc_arr = doc_tf_vector.toarray()[0]
    ind = np.argpartition(doc_arr, -top_k)[-top_k:]
    top_words_scores = {}
    for i in ind:
        top_words_scores[vectorizer.get_feature_names_out()[i]] = doc_arr[i]
    return sorted(top_words_scores, key=top_words_scores.get, reverse=True)

def get_docs_in_cluster(cluster, corpusidx_to_doc, clustered_sentences):
  docs = []
  for idx in cluster:
    docs.append(corpusidx_to_doc[clustered_sentences[i][idx]])
  return set(docs)

def get_srl_for_sentences(predictor, sentences):
    sentence_srls = []
    for s in sentences:
        sentence_srls.append(predictor.predict(sentence=s))
    parsed_results = []
    for srl in sentence_srls:
        # print(srl)
        for parse in srl['verbs']:
            # Skip Parses w/o Arguments
            found_args = False
            for j, word in enumerate(srl['words']):
                if "ARG" in parse['tags'][j]:
                    found_args = True
            if not found_args:
                continue

            # Skip Parses based around reporting verbs
            reporting_verbs = ['said', 'says']
            if parse['verb'] in reporting_verbs:
                continue

            event = ""
            for j, word in enumerate(srl['words']):
                if "ARG" in parse['tags'][j] or 'V' in parse['tags'][j]:
                    event += word + " "
            event = event[:len(event)-1]
            # print(event)
            parsed_results.append(event)
    return parsed_results

def get_srl_for_cluster(clustered_sentences, cluster, cluster_num, subcluster_num, corpusidx_to_doc, doc_to_text, predictor, anchor_word):
    doc_srls = []
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    for subcluster_idx in cluster[subcluster_num]:
        doc_id = corpusidx_to_doc[clustered_sentences[cluster_num][subcluster_idx]]
        text = doc_to_text[doc_id]
        sentences = sent_detector.tokenize(text)
        sentence_srls = []
        anchor_sent_idx = -1
        for i, s in enumerate(sentences):
            if anchor_word in s.lower():
                anchor_sent_idx = i
                # print(anchor_sent_idx, s.lower())
                break
        if anchor_sent_idx == -1:
            continue
        for i, s in enumerate(sentences):
            if i < anchor_sent_idx - 3 or i > anchor_sent_idx + 3:
                continue
            if len(s.split(" ")) > 200:
                print('skipped')
                continue
            # print(s)
            # print("=======")
            sentence_srls.append(predictor.predict(sentence=s))
        # print(sentence_srls)
        doc_srls.append(sentence_srls)
    return doc_srls

def get_events_before_after_anchor(cluster_srls, anchor_event):
    # Given a document
    # Find first sentence where anchor_event is mentioned
    # events from sentences before in before set
    # events from sentences after in after set
    events_before = []
    events_after = []
    for doc_srl in cluster_srls:
        # print(doc_srl)
        # Find first sentence where anchor event is mentioned
        anchor_sent_idx = -1
        for i in range(len(doc_srl)):
            sentence = [w.lower() for w in doc_srl[i]['words']]
            if anchor_event in sentence:
                anchor_sent_idx = i
                # print(sentence)
                break
        # print(anchor_sent_idx)
        # if anchor event not in this doc, skip pulling events from it
        if anchor_sent_idx == -1:
            continue
        for i in range(len(doc_srl)):
            # print(doc_srl[i]['words'])
            # if i < anchor_sent_idx - 3 or i > anchor_sent_idx + 3:
            #     continue
            for parse in doc_srl[i]['verbs']:
                event = ""
                for j, word in enumerate(doc_srl[i]['words']):
                    if "ARG" in parse['tags'][j] or 'V' in parse['tags'][j]:
                        event += word + " "
                event = event[:len(event)-1]
                # Skip events with 2 words or less
                if len(event.split(" ")) < 3:
                    continue
                if i < anchor_sent_idx:
                    events_before.append(event)
                elif i > anchor_sent_idx:
                    events_after.append(event)
    return set(events_before), set(events_after)

def get_events_before_after_anchor_no_srl(clustered_sentences, cluster, cluster_num, subcluster_num, corpusidx_to_doc, doc_to_text, anchor, corpus):
    before = []
    after = []
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    not_found_anchor = 0
    for subcluster_idx in cluster[subcluster_num]:
        corpus_idx = clustered_sentences[cluster_num][subcluster_idx] # idx of sentence in clustering in overall corpus
        doc_id = corpusidx_to_doc[corpus_idx] # id of document sentence was from
        text = doc_to_text[doc_id]
        sentences = sent_detector.tokenize(text)

        # anchor_sent_idx = -1
        # for i, s in enumerate(sentences):
        #     words = [w.lower() for w in s.split(" ")]
        #     if anchor in words:
        #         anchor_sent_idx = i
        #         break
        # if anchor_sent_idx == -1:
        #     continue
        anchor_sent_idx = corpus_idx_to_idx_in_doc(corpus_idx, corpusidx_to_doc)
        if anchor_sent_idx == -1:
            print("Error finding idx of clustered sent in doc")
            continue
        # if cluster_num == 6:
        #     print(corpus[corpus_idx])
        #     print(sentences[anchor_sent_idx])
        #     print("$$$$$$$$$$$$$"*3)

        # Make sure anchor sentence has anchor in it
        words = [w.lower() for w in sentences[anchor_sent_idx].split(" ")]
        if anchor not in words:
            not_found_anchor += 1
            continue

        for i, s in enumerate(sentences):
            if i < anchor_sent_idx - 3 or i > anchor_sent_idx + 3:
                continue
            if i < anchor_sent_idx:
                before.append(s)
            elif i > anchor_sent_idx:
                after.append(s)
    # print("NOT FOUND ANCHOR:", not_found_anchor, "Out of", len(cluster[subcluster_num]))
    return set(before), set(after)

def cluster_events(events, model, predictor, dist_thresh=0.4):
    event_embeddings = model.encode(events, convert_to_tensor=True)
    event_embeddings = event_embeddings.to('cpu')
    event_embeddings = event_embeddings /  np.linalg.norm(event_embeddings, axis=1, keepdims=True)
    clustering_model = AgglomerativeClustering(n_clusters=None, affinity='cosine', linkage='average', distance_threshold=dist_thresh)
    clustering_model.fit(event_embeddings)
    cluster_assignment = clustering_model.labels_
    clustered_sentences = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []
        clustered_sentences[cluster_id].append(events[sentence_id])
    # clustered_sentences = dict(sorted(clustered_sentences.items(), key=lambda c: len(c[1]), reverse=True)) # largest clusters
    clustered_sentences = dict(sorted(clustered_sentences.items())) # in order of label
    print(len(clustered_sentences), "clusters for", len(events), "sentences")
    print("---"*10)
    count = 0
    # predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz", cuda_device=0)
    srl_events = 0
    sent_count = 0
    for i, cluster in clustered_sentences.items():
        print("Cluster ", i+1)
        srls = get_srl_for_sentences(predictor, cluster)
        sent_count += len(cluster)
        for j, c in enumerate(srls):
            print("\t", j, ":", c)
            srl_events += 1
        if count > 18:
            break
        count += 1
    print(srl_events, "Total SRL Events from", sent_count , "sentences")
    return clustered_sentences


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--gigaword-root", required=True, type=str)
    parser.add_argument("--corpus-save-path", required=False, type=str)
    parser.add_argument("--corpus-load-path", required=False, type=str)
    parser.add_argument("--anchor-event", required=False, type=str)
    args = parser.parse_args()

    # Get cpu or gpu device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    print("Loading Documents")
    # news_prefixes = ['afp', 'apw', 'cna', 'ltw', 'nyt', 'wpb', 'xin']
    news_prefixes = ['ltw']
    doc_to_text = load_documents(news_prefixes, args.gigaword_root)
    print("Number of Documents", len(doc_to_text))

    print("Loading Sentence Encoder")
    # model = SentenceTransformer('all-mpnet-base-v2')
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Generating Corpus")
    # corpus, corpusidx_to_doc = generate_corpus(doc_to_text)
    with open(os.path.join(args.corpus_load_path, 'ltw_corpus.pkl'), "rb") as corpus_file:
        corpus = pickle.load(corpus_file)
    with open(os.path.join(args.corpus_load_path, 'ltw_corpusidx.pkl'), "rb") as corpus_file:
        corpusidx_to_doc = pickle.load(corpus_file)
    print("Corpus Size:", len(corpus), "sentences")

    if args.corpus_save_path:
        print(int((len(corpus)/5000000)) + 1, "Total Batch(es) - Can only enocde ~5M sentences at a time")
        for i in range(0, len(corpus), 5000000):
            print("Encoding Corpus")
            corpus_embeddings = model.encode(corpus[i:i+5000000 if i+5000000 < len(corpus) else len(corpus)],
                                        batch_size = 64, 
                                        convert_to_tensor=True, 
                                        show_progress_bar=True, 
                                        device=device)
            print("Saving Corpus Embeddings")
            corpus_output_dir = os.path.join(args.corpus_save_path)
            os.makedirs(corpus_output_dir, exist_ok=True)
            torch.save(corpus_embeddings, corpus_output_dir + "/ltw_f_corpus_emb" + str(int(i/5000000)) +".pt")
    elif args.corpus_load_path:
        # corpus_dir = os.path.join(args.corpus_load_path)
        # files = os.listdir(corpus_dir)
        # for f in files:
        #     corpus_embeddings = torch.load(os.path.join(dir_path, f), map_location=torch.device('cpu'))
        print("Loading embeddings")
        corpus_embeddings0 = torch.load(args.corpus_load_path + '/ltw_f_corpus_emb0.pt', map_location=torch.device('cpu'))
        # print(corpus_embeddings0.shape)
        corpus_embeddings1 = torch.load(args.corpus_load_path + '/ltw_f_corpus_emb1.pt', map_location=torch.device('cpu'))
        # print(corpus_embeddings1.shape)
        corpus_embeddings2 = torch.load(args.corpus_load_path + '/ltw_f_corpus_emb2.pt', map_location=torch.device('cpu'))
        corpus_embeddings = torch.cat((corpus_embeddings0, corpus_embeddings1, corpus_embeddings2), dim=0)
        # print(corpus_embeddings.shape)
    else:
        print("Corpus embeddings either need to be loaded or computed (and saved)")
    # corpus_embeddings = torch.narrow(corpus_embeddings, 0, 0, 150000)
    # print(corpus_embeddings.shape)
    # communitites = util.community_detection(corpus_embeddings, threshold=0.7, min_community_size=10)
    # print("done")
    
    # num_clusters = 140
    # found_clustering = False
    # while (not found_clustering):
    #     clustering_model = MiniBatchKMeans(n_clusters=num_clusters, max_iter=300)
    #     # t0 = time.time()
    #     clustering_model.fit(corpus_embeddings)
    #     # t1 = time.time()
    #     # print("TOOK", t1-t0, "seconds")
    #     cluster_assignment = clustering_model.labels_
    #     clustered_sentences = [[] for i in range(num_clusters)]
    #     for sentence_id, cluster_id in enumerate(cluster_assignment):
    #         clustered_sentences[cluster_id].append(sentence_id)
    #     good = True
    #     for i, cluster in enumerate(tqdm(clustered_sentences)):
    #         if len(cluster) > 160000:
    #             good = False
    #     if good:
    #         found_clustering = True
    #         with open(os.path.join(args.corpus_load_path, 'clustered_news_narratives_' + str(num_clusters) + '.pkl'), "wb") as cluster_file:
    #             print("Got", num_clusters, "clusters")
    #             pickle.dump(clustered_sentences, cluster_file)
    #     else:
    #         num_clusters += 10

    # UNCOMMENT 
    cluster_file = open(os.path.join(args.corpus_load_path, 'clustered_news_narratives_170.pkl'), 'rb')
    clustered_sentences = pickle.load(cluster_file)
    cluster_file.close()

    # communities_path = os.path.join(args.corpus_load_path, 'communities_f_t7_min25/')
    # os.makedirs(communities_path, exist_ok=True)
    # print("Subclustering KMeans Clusters")
    # for i, cluster in enumerate(tqdm(clustered_sentences)):
    #     if i < 146:
    #         continue
    #     if len(cluster) < 100:
    #         communities = []
    #     else:
    #         embeddings = torch.index_select(corpus_embeddings, 0, torch.tensor(cluster))
    #         # print(embeddings.shape)
    #         communities = util.community_detection(embeddings, threshold=0.7, min_community_size=25, init_max_size=1000 if embeddings.shape[0] > 1000 else embeddings.shape[0])
    #     with open(os.path.join(communities_path, 'cluster_' + str(i) + '_communities.pkl'), 'wb') as outfile:
    #         pickle.dump(communities, outfile)


    # clustered_sentences = pickle.load(cluster_file)
    # cluster_file.close()
    # sw_list = get_stopword_list()
    # with open(os.path.join(args.corpus_load_path, 'ltw_clustering_results_25f.txt'), 'w') as f:
    #     print("LTW Clustering Results", file=f)
    #     print("Embedding Model = all-MiniLM-L6-v2", file=f)
    #     print("First clustered into 90 large subclusters via MiniBatchKMeans, then clustered by cosine similarity (https://sbert.net/examples/applications/clustering/README.html#fast-clustering)", file=f)
    #     print("Threshold = 0.7, Min Community Size = 25", file=f)
    #     print("", file=f)
    #     for i, cluster in enumerate(tqdm(clustered_sentences)):
    #         print("==="*40, file=f)
    #         print("", file=f)
    #         print("Large Cluster", i, file=f)
    #         cg_file = open(os.path.join(args.corpus_load_path, 'communities_f_t7_min25/cluster_' + str(i) + '_communities.pkl'), 'rb')
    #         cluster_group = pickle.load(cg_file)
    #         cg_file.close()
    #         for j, sub_cluster in enumerate(cluster_group):
    #             overall_doc = ""
    #             for idx in sub_cluster:
    #                 overall_doc += doc_to_text[corpusidx_to_doc[clustered_sentences[i][idx]]] + " "
    #             print("\t Cluster", j, "-", get_keywords_for_doc(overall_doc, 5, sw_list), file=f)
                # print("\t\t Docs: ", get_docs_in_cluster(sub_cluster, corpusidx_to_doc, clustered_sentences), file=f)
                # print("", file=f)

    if args.anchor_event:
        # Determine subclusters that contain anchor_event
        anchor_clusters = []
        with open(os.path.join(args.corpus_load_path, 'ltw_clustering_results_25f.txt'), 'rb') as clustering_results:
            cluster_num = 0
            for line in clustering_results:
                l = line.decode('UTF-8')
                if 'Large Cluster' in l:
                    cluster_num = int(l.split(" ")[2])
                    continue
                elif '- [' not in l:
                    continue
                l_split = l.split(' - ')
                subcluster_num = int(l_split[0].split(' ')[2])
                keywords = l_split[1][1:len(l_split[1]) - 2].split(', ')
                for i, word in enumerate(keywords):
                    if i > 2: # only check top 3 keywords
                        break
                    if args.anchor_event == word[1:len(word)-1]:
                        anchor_clusters.append((cluster_num, subcluster_num))
                        break
        print("Found", len(anchor_clusters), "subclusters for", args.anchor_event)
        print(anchor_clusters)
        events_before = []
        events_after = []
        predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz", cuda_device=0)
        print("Loaded SRL Parser")
        for clusters in tqdm(anchor_clusters):
            cg_file = open(os.path.join(args.corpus_load_path, 'communities_f_t7_min25/cluster_' + str(clusters[0]) + '_communities.pkl'), 'rb')
            cluster_group = pickle.load(cg_file)
            cg_file.close()
            # cluster_srls = get_srl_for_cluster(clustered_sentences, cluster_group, clusters[0], clusters[1], corpusidx_to_doc, doc_to_text, predictor, args.anchor_event)
            # eb, ea = get_events_before_after_anchor(cluster_srls, args.anchor_event)
            eb, ea = get_events_before_after_anchor_no_srl(clustered_sentences, cluster_group, clusters[0], clusters[1], corpusidx_to_doc, doc_to_text, args.anchor_event, corpus)
            events_before.extend(eb)
            events_after.extend(ea)
            # print(eb)
            # print("!!!")
            # print(ea)
            # print(len(eb), len(ea))
        events_before = set(events_before)
        events_after = set(events_after)
        # print(events_before)
        # print("==="*30)
        # print(events_after)
        # print("==="*30)
        print("Events Before:", len(events_before), "Events After:", len(events_after))
        cluster_events(list(events_before), model, predictor)
        print("==="*30)
        cluster_events(list(events_after), model, predictor)