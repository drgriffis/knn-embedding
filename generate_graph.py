'''
Given NN data calculated from embedding samples (by nn_saver.py),
generates a node2vec-format k-NN graph as follows:
 (1) Every word (vocabulary index) becomes a node
 (2) For each neighbor v of node w: (w,v) is a directed edge in the graph

This script supports combining multiple nearest neighborhood files.
If multiple neighborhood files are provided as input, the following
changes are made in graph generation:
 - The neighbor sets of node w are unioned across all neighborhood samples
 - Each edge (w,v) is assigned weight 1/M, where M is the number of neighborhood
   sample files were v is a nearest neighbor of w.
'''

from dependencies import configlogger
from dependencies.drgriffis.common import log

def readNeighbors(samplef, k):
    neighborhoods = {}
    with open(samplef, 'r') as stream:
        for line in stream:
            if line[0] != '#':
                (source, *neighbors) = [int(s) for s in line.split(',')]
                neighborhoods[source] = neighbors[:k]
    return neighborhoods
    
def buildGraph(neighbor_files, k):
    log.writeln('Building neighborhood graph...')
    graph = {}

    # construct frequency-weighted edges
    log.track(message='  >> Loaded {0}/%d neighborhood files' % len(neighbor_files), writeInterval=1)
    for neighbor_file in neighbor_files:
        neighborhoods = readNeighbors(neighbor_file, k)
        for (source, neighbors) in neighborhoods.items():
            if graph.get(source, None) is None:
                graph[source] = {}
            for nbr in neighbors:
                graph[source][nbr] = graph[source].get(nbr, 0) + 1
        log.tick()
    log.flushTracker()

    log.writeln('  >> Normalizing edge weights...')
    max_count = float(len(neighbor_files))
    for (source, neighborhood) in graph.items():
        for (nbr, freq) in neighborhood.items():
            graph[source][nbr] = freq/max_count

    log.writeln('Graph complete!')
    return graph

def writeGraph(graph, outf):
    with open(outf, 'w') as stream:
        for (source, neighbors) in graph.items():
            for (neighbor, edge_weight) in neighbors.items():
                stream.write('%s\n' % ' '.join([
                    str(source), str(neighbor), str(edge_weight)
                ]))

if __name__ == '__main__':
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog NN_FILE_1 [NN_FILE_2 [NN_FILE_3 [...]]]')
        parser.add_option('-o', '--output', dest='outputf',
                help='file to write weighted graph to (default: %default)',
                default='output.grph')
        parser.add_option('-k', dest='k',
                help='number of neighbors to use for edge construction (default: %default)',
                type='int', default=10)
        parser.add_option('-l', '--logfile', dest='logfile',
                help='name of file to write log contents to (empty for stdout)',
                default=None)
        (options, args) = parser.parse_args()
        if len(args) == 0:
            parser.print_help()
            exit()
        neighbor_files = args
        return neighbor_files, options
    neighbor_files, options = _cli()
    log.start(logfile=options.logfile)
    configlogger.writeConfig(log, [
        *[
            ('Neighborhood sample file %d' % (i+1), neighbor_files[i])
                for i in range(len(neighbor_files))
        ],
        ('Output file', options.outputf),
        ('Number of neighbors to include in edge construction', options.k),
    ], 'Nearest neighborhood graph generation')

    graph = buildGraph(neighbor_files, options.k)

    log.write('Writing graph to %s...' % options.outputf)
    writeGraph(graph, options.outputf)
    log.writeln('Done!')

    log.stop()
