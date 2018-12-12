class Mode:
    Text = 'txt'
    Binary = 'bin'

class Format:
    Word2Vec = 'Word2Vec'
    Glove = 'Glove'

def getFileSize(inf):
    curIx = inf.tell()
    inf.seek(0, 2)  # jump to end of file
    file_size = inf.tell()
    inf.seek(curIx)
    return file_size

class CLI_Formats:
    _labeled_formats = {
        'word2vec-binary' : 'Binary word2vec format',
        'word2vec-text'   : 'Text word2vec format',
    }

    @staticmethod
    def options():
        return CLI_Formats._labeled_formats.keys()

    @staticmethod
    def default():
        return 'word2vec-binary'

    @staticmethod
    def parse(key):
        if key == 'word2vec-binary':
            format = Format.Word2Vec
            mode = Mode.Binary
        elif key == 'word2vec-text':
            format = Format.Word2Vec
            mode = Mode.Text
        else:
            format = None
            mode = None
        return (format, mode)
