'''
Get the top k nearest neighbors for a set of embeddings and save to a file
'''

import multiprocessing as mp
import tensorflow as tf
import numpy as np
import codecs
import os
from nearest_neighbors import NearestNeighbors
import pyemblib
import configlogger
from drgriffis.common import log, util

class _SIGNALS:
    HALT = -1
    COMPUTE = 1

def KNearestNeighbors(emb_arr, node_IDs, top_k, neighbor_file, threads=2, batch_size=5, completed_neighbors=None):
    '''docstring goes here
    '''
    # set up threads
    log.writeln('1 | Thread initialization')
    all_indices = list(range(len(emb_arr)))
    if completed_neighbors:
        filtered_indices = []
        for ix in all_indices:
            if not ix in completed_neighbors:
                filtered_indices.append(ix)
        all_indices = filtered_indices
        log.writeln('  >> Filtered out {0:,} completed indices'.format(len(emb_arr) - len(filtered_indices)))
        log.writeln('  >> Filtered set size: {0:,}'.format(len(all_indices)))
    #index_subsets = util.prepareForParallel(list(range(len(emb_arr))), threads-1, data_only=True)
    index_subsets = util.prepareForParallel(all_indices, threads-1, data_only=True)
    nn_q = mp.Queue()
    nn_writer = mp.Process(target=_nn_writer, args=(neighbor_file, node_IDs, nn_q))
    computers = [
        mp.Process(target=_threadedNeighbors, args=(index_subsets[i], emb_arr, batch_size, top_k, nn_q))
            for i in range(threads - 1)
    ]
    nn_writer.start()
    log.writeln('2 | Neighbor computation')
    util.parallelExecute(computers)
    nn_q.put(_SIGNALS.HALT)
    nn_writer.join()

def _nn_writer(neighborf, node_IDs, nn_q):
    stream = open(neighborf, 'w')
    stream.write('# File format is:\n# <word vocab index>,<NN 1>,<NN 2>,...\n')
    result = nn_q.get()
    log.track(message='  >> Processed {0}/{1:,} samples'.format('{0:,}', len(node_IDs)), writeInterval=50)
    while result != _SIGNALS.HALT:
        (ix, neighbors) = result
        stream.write('%s\n' % ','.join([
            str(d) for d in [
                node_IDs[ix], *[
                    node_IDs[nbr]
                        for nbr in neighbors
                ]
            ]
        ]))
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

def writeNodeMap(emb, f):
    ordered = tuple([
        k.strip()
            for k in emb.keys()
            if len(k.strip()) > 0
    ])
    node_id = 1  # start from 1 in case 0 is reserved in node2vec
    with codecs.open(f, 'w', 'utf-8') as stream:
        for v in ordered:
            stream.write('%d\t%s\n' % (
                node_id, v
            ))
            node_id += 1
    
def readNodeMap(f, as_ordered_list=False):
    node_map = {}
    with codecs.open(f, 'r', 'utf-8') as stream:
        for line in stream:
            (node_id, v) = [s.strip() for s in line.split('\t')]
            node_map[int(node_id)] = v

    if as_ordered_list:
        keys = list(node_map.keys())
        keys.sort()
        node_map = [
            node_map[k]
                for k in keys
        ]
    return node_map

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
        parser.add_option('-k', '--nearest-neighbors', dest='k',
                help='number of nearest neighbors to calculate (default: %default)',
                type='int', default=25)
        parser.add_option('--batch-size', dest='batch_size',
                type='int', default=25,
                help='number of points to process at once (default %default)')
        parser.add_option('--embedding-mode', dest='embedding_mode',
                type='choice', choices=[pyemblib.Mode.Text, pyemblib.Mode.Binary], default=pyemblib.Mode.Binary,
                help='embedding file is in text ({0}) or binary ({1}) format (default: %default)'.format(pyemblib.Mode.Text, pyemblib.Mode.Binary))
        parser.add_option('--partial-neighbors-file', dest='partial_neighbors_file',
                help='file with partially calculated nearest neighbors (for resuming long-running job)')
        parser.add_option('-l', '--logfile', dest='logfile',
                help='name of file to write log contents to (empty for stdout)',
                default=None)
        (options, args) = parser.parse_args()
        if len(args) != 1:
            parser.print_help()
            exit()
        (embf,) = args
        return embf, options

    embf, options = _cli()
    log.start(logfile=options.logfile)
    configlogger.writeConfig(log, [
        ('Input embedding file', embf),
        ('Input embedding file mode', options.embedding_mode),
        ('Output neighbor file', options.outputf),
        ('Ordered vocabulary file', options.vocabf),
        ('Number of nearest neighbors', options.k),
        ('Batch size', options.batch_size),
        ('Number of threads', options.threads),
        ('Partial nearest neighbors file for resuming', options.partial_neighbors_file),
    ], 'k Nearest Neighbor calculation with cosine similarity')

    t_sub = log.startTimer('Reading embeddings from %s...' % embf)
    emb = pyemblib.read(embf, mode=options.embedding_mode, errors='replace')
    log.stopTimer(t_sub, message='Read {0:,} embeddings in {1}s.\n'.format(len(emb), '{0:.2f}'))

    if not os.path.isfile(options.vocabf):
        log.writeln('Writing node ID <-> vocab map to %s...\n' % options.vocabf)
        writeNodeMap(emb, options.vocabf)
    else:
        log.writeln('Reading node ID <-> vocab map from %s...\n' % options.vocabf)
    node_map = readNodeMap(options.vocabf)

    # get the vocabulary in node ID order, and map index in emb_arr
    # to node IDs
    node_IDs = list(node_map.keys())
    node_IDs.sort()
    ordered_vocab = [
        node_map[node_ID]
            for node_ID in node_IDs
    ]

    emb_arr = np.array([
        emb[v] for v in ordered_vocab
    ])

    if options.partial_neighbors_file:
        completed_neighbors = set()
        with open(options.partial_neighbors_file, 'r') as stream:
            for line in stream:
                if line[0] != '#':
                    (neighbor_id, _) = line.split(',', 1)
                    completed_neighbors.add(int(neighbor_id))
    else:
        completed_neighbors = set()

    log.writeln('Calculating k nearest neighbors.')
    KNearestNeighbors(
        emb_arr,
        node_IDs,
        options.k,
        options.outputf,
        threads=options.threads,
        batch_size=options.batch_size,
        completed_neighbors=completed_neighbors
    )
    log.writeln('Done!\n')

    log.stop()
