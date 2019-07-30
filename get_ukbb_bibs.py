from urllib.request import urlopen
from bs4 import BeautifulSoup as bs
import bs4

# define all urls since not all of them are the same
urls = ['https://www.ukbiobank.ac.uk/published-papers/',
        'https://www.ukbiobank.ac.uk/publications-2018/',
        'https://www.ukbiobank.ac.uk/publications-2017/',
        'https://www.ukbiobank.ac.uk/publications-from-2016/',
        'https://www.ukbiobank.ac.uk/publications-from-2015/',
        'https://www.ukbiobank.ac.uk/publications-from-2014/',
        'https://www.ukbiobank.ac.uk/publications-from-2013/',
        'https://www.ukbiobank.ac.uk/2012-2/']


def get_bib_from_tag(tag):
    return '\n'.join([i.strip() for i in tag.contents if isinstance(
        i, bs4.element.NavigableString)])


#loop over the urls and read them
bibs = []
for url in urls:
    print('Processing', url)
    with urlopen(url) as u:
        handle = u.read()
        soup = bs(handle)
        # find the bibentry class
        mydivs = soup.findAll("div", {"class": "tp_bibtex_entry"})
        for tag in mydivs:
            bib = get_bib_from_tag(tag)
            if not bib in bibs:
                bibs.append(bib)

print('%d records found' % len(bibs))
with open('ukbb.bib', 'w') as out:
    out.write('\n'.join(bibs))

