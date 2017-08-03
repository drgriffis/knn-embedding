'''
Get the top k nearest neighbors for a set of embeddings and save to a file
'''

import multiprocessing as mp
import tensorflow as tf
import numpy as np
import codecs
import os
from scipy.stats import spearmanr
from nearest_neighbors import NearestNeighbors
from drgriffis.science import embeddings
from drgriffis.common import log, util

class _SIGNALS:
    HALT = -1
    COMPUTE = 1

def KNearestNeighbors(emb_arr, top_k, neighbor_file, threads=2, batch_size=5):
    '''docstring goes here
    '''
    # set up threads
    log.writeln('1 | Setup')
    index_subsets = util.prepareForParallel(list(range(len(emb_arr))), threads-1, data_only=True)
    nn_q = mp.Queue()
    nn_writer = mp.Process(target=_nn_writer, args=(neighbor_file, len(emb_arr), nn_q))
    computers = [
        mp.Process(target=_threadedNeighbors, args=(index_subsets[i], emb_arr, batch_size, top_k, nn_q))
            for i in range(threads - 1)
    ]
    nn_writer.start()
    log.writeln('2 | Compute')
    util.parallelExecute(computers)
    nn_q.put(_SIGNALS.HALT)
    nn_writer.join()

def _nn_writer(neighborf, total, nn_q):
    stream = open(neighborf, 'w')
    result = nn_q.get()
    log.track(message='  >> Processed {0}/%d samples' % total, writeInterval=500)
    while result != _SIGNALS.HALT:
        (ix, neighbors) = result
        stream.write('%s\n' % ','.join([str(d) for d in [ix, *neighbors]]))
        log.tick()
        result = nn_q.get() 
    log.flushTracker()

def _threadedNeighbors(thread_indices, emb_arr, batch_size, top_k, nn_q):
    sess = tf.Session()
    grph = NearestNeighbors(sess, emb_arr)

    ix = 0
    while ix < len(thread_indices):
        batch = thread_indices[ix:ix+batch_size]
        nn = grph.nearestNeighbors(batch, top_k=top_k, no_self=True)
        for i in range(len(batch)):
            nn_q.put((batch[i], nn[i]))
        ix += batch_size

if __name__ == '__main__':
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog EMB1')
        parser.add_option('-t', '--threads', dest='threads',
                help='number of threads to use in the computation (min 2, default: %default)',
                type='int', default=2)
        parser.add_option('-o', '--output', dest='outputf',
                help='file to write nearest neighbor results to (default: %default)',
                default='output.csv')
        parser.add_option('--vocab', dest='vocabf',
                help='file to read ordered vocabulary from (will be written if does not exist yet)')
        parser.add_option('--glove-vocab', dest='glove_vocabf',
                help='vocab file if using GloVe vectors')
        parser.add_option('-k', '--nearest-neighbors', dest='k',
                help='number of nearest neighbors to calculate (default: %default)',
                type='int', default=25)
        parser.add_option('-l', '--logfile', dest='logfile',
                help='name of file to write log contents to (empty for stdout)',
                default=None)
        parser.add_option('--text', dest='embtext',
                help='read embeddings in text format instead of binary',
                action='store_true', default=False)
        (options, args) = parser.parse_args()
        if len(args) != 1:
            parser.print_help()
            exit()
        (embf,) = args
        return embf, options.glove_vocabf, options.vocabf, options.threads, options.outputf, options.k, options.embtext, options.logfile
    embf, glove_vocabf, vocabf, threads, outputf, k, embtext, logfile = _cli()
    log.start(logfile=logfile, stdout_also=True)

    if glove_vocabf:
        emb = embeddings.read(embf, format=embeddings.Format.Glove, vocab=glove_vocabf, mode=embeddings.glove.GloveMode.SumContexts)
    else:
        mode = embeddings.Mode.Text if embtext else embeddings.Mode.Binary
        emb = embeddings.read(embf, mode=mode)

    if not os.path.isfile(vocabf):
        log.writeln('Writing ordered vocabulary to %s' % vocabf)
        embeddings.listVocab(emb, vocabf)
    else:
        log.writeln('Reading ordered vocabulary from %s' % vocabf)
    ordered_vocab = util.readList(vocabf, encoding='utf-8')

    emb_arr = np.array([
        emb[v] for v in ordered_vocab
    ])

    batch_size = 25
    KNearestNeighbors(emb_arr, k, outputf, threads=threads, batch_size=batch_size)
