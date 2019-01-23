'''
Given embeddings learned from node2vec (which are keyed to node IDs),
remap them using a supplied vocabulary file to words and write out a
new embedding file.
'''

import codecs
import pyemblib
import configlogger
from drgriffis.common import log

def readVocab(f):
    vocab = []
    with codecs.open(f, 'r', 'utf-8') as stream:
        for line in stream:
            if len(line.strip()) > 0:
                vocab.append(line.strip())
    return vocab

if __name__ == '__main__':
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog -i INPUT -o OUTPUT -v VOCAB')
        parser.add_option('-i', '--input', dest='inputf',
                help='(REQUIRED) input node2vec embedding file')
        parser.add_option('-o', '--output', dest='outputf',
                help='(REQUIRED) output word2vec embedding file')
        parser.add_option('-v', '--vocab', dest='vocabf',
                help='(REQUIRED) file mapping node IDs to vocabulary words')
        pyemblib.CLI_Formats.addCLIOption(parser, '--output-format', dest='output_format',
                help='format of output embeddings')
        parser.add_option('-l', '--logfile', dest='logfile',
                help='name of file to write log contents to (empty for stdout)',
                default=None)
        (options, args) = parser.parse_args()
        if (not options.inputf) or (not options.outputf) or (not options.vocabf):
            parser.print_help()
            exit()
        return options
    options = _cli()

    log.start(logfile=options.logfile)
    configlogger.writeConfig(log, [
        ('Input embeddings', options.inputf),
        ('Vocabulary file', options.vocabf),
        ('Output embeddings', options.outputf),
        ('Output embeddings format', options.output_format),
    ])

    log.startTimer('Reading node2vec embeddings from %s...' % options.inputf)
    e = pyemblib.read(options.inputf, format=pyemblib.Format.Word2Vec, mode=pyemblib.Mode.Text)
    log.stopTimer(message='Read {0:,} embeddings in {1}s.\n'.format(
        len(e), '{0:.2f}'
    ))

    log.writeln('Reading vocabulary mapping from %s...' % options.vocabf)
    vocab = readVocab(options.vocabf)
    log.writeln('Read {0:,} vocab mappings.\n'.format(len(vocab)))

    e = {
        vocab[int(k)]: v
            for (k,v) in e.items()
    }
    log.writeln('Writing remapped embeddings to %s...' % options.outputf)
    (fmt, mode) = pyemblib.CLI_Formats.parse(options.output_format)
    pyemblib.write(e, options.outputf, format=fmt, mode=mode, verbose=True)
    log.writeln('Done!')

    log.stop()
