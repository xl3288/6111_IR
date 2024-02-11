import pprint
from googleapiclient.discovery import build
import sys
from urllib.request import Request, urlopen
import urllib.error
from bs4 import BeautifulSoup
from sklearn import feature_extraction
import numpy as np
import pdb

search_engine_id = '56010a4c9f84b48fe'
api_key = 'AIzaSyAJIO83SwsYtTeAWmw7v0xryh2_boCizEI'


def get_query_res(query):
    service = build("customsearch", "v1", developerKey=api_key)
    query_res = (service.cse().list(q=query, cx=search_engine_id, num=10).execute())
    clean_res = []
    for item in query_res['items']:
        # deal with non-html
        if 'fileFormat' in item.keys() and item['fileFormat'].lower() not in ('html', 'json'):
            continue
        this_res = [item['title'], item['formattedUrl'], item['snippet']]
        try:
            print("Openning URL:", item['formattedUrl'])
            req = Request(item['formattedUrl'], headers={'User-Agent': 'Mozilla/5.0'})
            content = urlopen(req, timeout=10)
            content_bytes = content.read()
            content_str = content_bytes.decode("utf8")
            parsed_content_str = clean_html(content_str)
            this_res.append(parsed_content_str)
            content.close()
            # pdb.set_trace()
            clean_res.append(this_res)
        except urllib.error.HTTPError as err:
            print(f'A HTTPError was thrown: {err.code} {err.reason}')
            this_res.append("")  # TODO: do we want to append snippets again?
            clean_res.append(this_res)
    return clean_res


def print_params(query, precision):
    print("Parameters:")
    print("Client key = %s" % api_key)
    print("Engine key = %s" % search_engine_id)
    print("Query = %s" % query)
    print("Precision = %s" % precision)


def get_relevance_feedback(clean_res, query, target_prec):
    print("Google Search Results:")
    print("======================")

    feedbacks = {'Y': [], 'N': []}
    for i in range(len(clean_res)):
        print("Result %s" % (i + 1))
        print("[")
        print("URL: %s" % clean_res[i][1])
        print("Title: %s" % clean_res[i][0])
        print("Summary: %s" % clean_res[i][2])
        print("]")

        user_input = input("Relevant (Y/N)?")
        if user_input.title() == 'Y':
            feedbacks['Y'].append(clean_res[i][2])
            if clean_res[i][3] != "":
                feedbacks['Y'].append(clean_res[i][3])
        else:
            feedbacks['N'].append(clean_res[i][2])

    print("======================")
    print("FEEDBACK SUMMARY")
    print("Query: %s" % query)
    calc_prec = len(feedbacks['Y']) * 1.0 / len(clean_res)
    print("Precision = %s" % calc_prec)
    if calc_prec == 0.0:
        print("No relevant results among the top-10 pages, program will terminate")
        sys.exit()
    elif calc_prec < target_prec:
        print("Still below the desired precision of %s" % target_prec)
    else:
        print("Desired precision reached, done")

    return feedbacks


def run_Rocchio_algo(query, feedbacks):
    term_map = {}

    general_vectorizer = feature_extraction.text.TfidfVectorizer(stop_words='english')
    # get vocabulary mapping so all the vectors are in same feature space
    feedback_vector = general_vectorizer.fit_transform(feedbacks['Y'] + feedbacks['N'])

    vectorizer = feature_extraction.text.TfidfVectorizer(stop_words='english',
                                                         vocabulary=general_vectorizer.vocabulary_)
    original_query_vector = vectorizer.fit_transform([query]).toarray()
    positive_feedbacks_vector = vectorizer.fit_transform(feedbacks['Y']).toarray()
    negative_feedbacks_vector = vectorizer.fit_transform(feedbacks['N']).toarray()

    a = 1
    b = 0.8
    c = 0.1
    top_n = 10
    feedbacks_sum = lambda x: np.sum(x, axis=0)
    part1 = a * original_query_vector
    part2 = b / len(feedbacks['Y']) * feedbacks_sum(positive_feedbacks_vector)
    part3 = c / len(feedbacks['N']) * feedbacks_sum(negative_feedbacks_vector)
    modified_query_vector = part1 + part2 - part3

    # print('modified_query_vector:', modified_query_vector)

    features = np.array(vectorizer.get_feature_names_out())
    term_weights = np.argsort(modified_query_vector).flatten()[::-1]
    top_n_features = features[term_weights][:top_n]
    print('top_n_features:', top_n_features)

    return set(query + top_n_features)


def clean_html(html):
    # parse html content
    soup = BeautifulSoup(html, "html.parser")
    for data in soup(['style', 'script']):
        data.decompose()
    return ' '.join(soup.stripped_strings)


if __name__ == "__main__":
    print(sys.argv)

    # input format: precision "query", e.g. 0.7 "per se"
    # TODO: to add api_key and engine_id as inputs
    precision = float(sys.argv[1])
    query = sys.argv[2]

    for i in range(10):  # terminate after 10 times
        print(i)
        print_params(query, precision)
        clean_res = get_query_res(query)
        # print(clean_res)
        feedbacks = get_relevance_feedback(clean_res, query, precision)
        new_query = run_Rocchio_algo(query, feedbacks)
        query = new_query


