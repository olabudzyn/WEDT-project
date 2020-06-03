
class AbstractTagger(object):
    def possible_tags(self):
        raise NotImplementedError('subclasses must override possible_tags()!')

    def map_sentence(self, sentence):
        raise NotImplementedError('subclasses must override map_sentence()!')
