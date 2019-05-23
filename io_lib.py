'''
I/O methods for neighbor files and node map files
'''

import codecs

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

def writeNeighborFileLine(stream, node_ID, neighbors):
    stream.write('%s\n' % ','.join([
        str(d) for d in [
            node_IDs[ix], *[
                node_IDs[nbr]
                    for nbr in neighbors
            ]
        ]
    ]))

def readNeighborFile(f, k=None, node_map=None):
    '''Read a neighbor file into a dictionary mapping
    { node: [neighbor list] }

    If k is supplied, restricts to the first k neighbors
    listed (i.e., the closest k neighbors)

    If node_map is supplied (as a dict), maps node IDs
    to labels in node_map.
    '''
    neighbors = {}
    with codecs.open(f, 'r', 'utf-8') as stream:
        for line in stream:
            if line[0] != '#':
                (node_ID, *neighbor_IDs) = [int(s) for s in line.split(',')]
                if node_map:
                    node_ID = node_map.get(node_ID, node_ID)
                    neighbor_IDs = [
                        node_map.get(nbr_ID, nbr_ID)
                            for nbr_ID in neighbor_IDs
                    ]
                if k:
                    neighbor_IDs = neighbor_IDs[:k]
                neighbors[node_ID] = neighbor_IDs
    return neighbors
