from io import open
import glob
import unicodedata
import string

ALL_LETTERS = string.ascii_letters + " .,;'"

def findFiles(path):
    return glob.glob(path)


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in ALL_LETTERS
    )


def readLines(filename):
    lines = open(filename, encoding = 'utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]



def extract(path):
    category_lines = {}
    all_categories = []

    for filename in findFiles(path):
        category = filename.split('/')[-1].split('.')[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines

    return ALL_LETTERS, len(ALL_LETTERS), category_lines, all_categories
