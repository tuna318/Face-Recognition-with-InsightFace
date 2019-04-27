import os
import argparse
from icrawler.builtin import GoogleImageCrawler

# pa = argparse.ArgumentParser()
# pa.add_argument('-k', '--keyword', required = True,
#                 help = 'Object to search on google')
# pa.add_argument('-n', '--number', required = True,
#                 help = 'Numbers of picture to be crawled')
# args = vars(pa.parse_args())

actors = ["Eddard Ned Stark", "Robert Baratheon", "Tyrion Lannister", "Cersei Lannister",
          "Catelyn Stark", "Jaime Lannister", "Daenerys Targaryen", "Jon Snow", "Robb Stark",
          "Sansa Stark", "Arya Stark", "Bran Stark", "Joffrey Baratheon", "Jorah Mormont",
          "Theon Greyjoy", "Petyr Baelish", "Sandor Clegane (The Hound)", "Samwell Tarly",
          "Renly Baratheon", "Jeor Mormont", "Gendry", "Lysa Arryn", "Bronn", "Grand Maester Pycelle",
          "Varys", "Barristan Selmy", "Khal Drogo", "Hodor", "Maester Luwin", "Brienne of Tarth",
          "Davos Seaworth", "Tywin Lannister", "Stannis Baratheon", "Margaery Tyrell", "Ygritte",
          "Podrick Payne", "Melisandre", "Yara Greyjoy", "Grey Worm", "Missandei", "Tormund",
          "Ramsay Snow", "Qyburn", "Euron Greyjoy"]

if __name__ == '__main__':
    for actor in actors:
        google_crawler = GoogleImageCrawler(
            feeder_threads = 1,
            parser_threads = 1,
            downloader_threads = 4,
            storage = {'root_dir' : 'datasets/train/%s'%actor})
        filters = dict(size = 'medium')
        google_crawler.crawl(keyword = actor, filters = filters, max_num = int(60))
