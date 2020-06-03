import nltk
import numpy as np
from Tagger import AbstractTagger


class UniversalTagger(AbstractTagger):
    def __init__(self):
        AbstractTagger.__init__(self)
        self.possibleTagsList = np.array(['ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRT', 'PRON', 'VERB', '.', 'X', 'EMPTY']).tolist()

    def possible_tags(self):
        return self.possibleTagsList

    def map_sentence(self, sentence):
        list_of_pos_tags = nltk.pos_tag(sentence, tagset='universal')
        list_of_tags = []
        for i in range(0, len(list_of_pos_tags)):
            if sentence[i] == '-':
                list_of_tags.append('EMPTY')
            else:
                tag_tuple = list_of_pos_tags[i]
                tag = tag_tuple[1]
                if tag in self.possibleTagsList:
                    list_of_tags.append(tag)
                else:
                    print("Unexpected tag:", tag)
                    raise NotImplementedError('Unexpected tag!')
        return list_of_tags
