import nltk
from nltk.data import load
from Tagger import AbstractTagger


class FullTagger(AbstractTagger):
    def __init__(self):
        AbstractTagger.__init__(self)
        tag_list = load('help/tagsets/upenn_tagset.pickle').keys()
        self.possibleTagsList = list(dict.fromkeys(map(self.__map_tag, tag_list)))

    def possible_tags(self):
        return self.possibleTagsList

    def map_sentence(self, sentence):
        list_of_pos_tags = nltk.pos_tag(sentence)
        list_of_tags = []
        for i in range(0, len(list_of_pos_tags)):
            tag_tuple = list_of_pos_tags[i]
            tag = self.__map_tag(tag_tuple[1])
            list_of_tags.append(tag)
        return list_of_tags

    def __map_tag(self, tag):
        if tag == 'VBN':
            return 'VBN'
        elif tag == 'VBZ':
            return 'VBZ'
        elif tag == 'VBG':
            return 'VBG'
        elif tag == 'VBP':
            return 'VBP'
        elif tag == 'VBD':
            return 'VBD'
        elif tag == 'MD':
            return 'MD'
        elif tag == 'NN':
            return 'NN'
        elif tag == 'NNPS':
            return 'NNPS'
        elif tag == 'NNP':
            return 'NNP'
        elif tag == 'NNS':
            return 'NNS'
        elif tag == 'JJS':
            return 'JJS'
        elif tag == 'JJR':
            return 'JJR'
        elif tag == 'JJ':
            return 'JJ'
        elif tag == 'RB':
            return 'RB'
        elif tag == 'RBR':
            return 'RB'
        elif tag == 'RBS':
            return 'RB'
        elif tag == '-':
            return 'EMPTY'
        elif tag == 'CD':
            return 'CD'
        elif tag == 'IN':
            return 'IN'
        elif tag == 'PDT':
            return 'PDT'
        elif tag == 'CC':
            return 'CC'
        elif tag == 'EX':
            return 'EX'
        elif tag == 'POS':
            return 'POS'
        elif tag == 'RP':
            return 'RP'
        elif tag == 'FW':
            return 'FW'
        elif tag == 'DT':
            return 'DT'
        elif tag == 'UH':
            return 'UH'
        elif tag == 'TO':
            return 'TO'
        elif tag == 'PRP':
            return 'PRP'
        elif tag == 'PRP$':
            return 'PRP$'
        elif tag == '$':
            return '$'
        elif tag == 'WP':
            return 'WP'
        elif tag == 'WP$':
            return 'WP$'
        elif tag == 'WDT':
            return 'WDT'
        elif tag == 'WRB':
            return 'WRB'
        else:
            return 'OTHER'
