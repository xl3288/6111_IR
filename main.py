import pprint
from googleapiclient.discovery import build
import sys

search_engine_id = '56010a4c9f84b48fe'
api_key = 'AIzaSyAJIO83SwsYtTeAWmw7v0xryh2_boCizEI'

def main():
    print (sys.argv)
    # input format: "query" precision, e.g "per se" 0.7

    query = sys.argv[1]
    precision = float(sys.argv[2])
    print(query, precision)

    service = build("customsearch", "v1", developerKey=api_key)
    res = (service.cse().list(q=query, cx=search_engine_id, num=10).execute())
    pprint.pprint(res)

if __name__ == "__main__":
    main()