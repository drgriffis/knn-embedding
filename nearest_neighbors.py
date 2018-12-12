'''
'''

import numpy as np
import tensorflow as tf
import multiprocessing as mp

class NearestNeighbors:
    
    def __init__(self, session, embed_array):
        # unit norm the embedding array
        embed_array = np.array([
            vec / np.linalg.norm(vec)
                for vec in embed_array
        ])

        self._session = session
        self._prints = []
        
        self._dim = embed_array.shape[1]

        self._build(embed_array.shape)

        self._session.run(tf.global_variables_initializer())

        # fill the (static) embedding matrix
        self._session.run(self._embed_matrix.assign(self._embed_ph), feed_dict={self._embed_ph: embed_array})

    def _build(self, emb_shape):
        self._sample_indices = tf.placeholder(
            shape=[None,],
            dtype=tf.int32
        )
        self._embed_ph = tf.placeholder(
            shape=emb_shape,
            dtype=tf.float32
        )
        self._embed_matrix = tf.Variable(
            tf.constant(0.0, shape=emb_shape),
            trainable=False
        )
        self._sample_points = tf.gather(
            self._embed_matrix,
            self._sample_indices
        )

        self._sample_distances = self._distance(self._sample_points, self._embed_matrix)

    def _distance(self, a, b):
        # first, L2-norm both inputs
        #normed_a = tf.nn.l2_normalize(a, 1)
        #normed_b = tf.nn.l2_normalize(b, 1)
        normed_a = a
        normed_b = b   # already unit-normed
        # get full pairwise distance matrix
        pairwise_distance = 1 - tf.matmul(normed_a, normed_b, transpose_b=True)
        return pairwise_distance

    def _print(self, *nodes):
        for n in nodes:
            if type(n) is tuple and len(n) == 2:
                self._prints.append(tf.Print(0, [n[0]], message=n[1], summarize=100))
            else:
                self._prints.append(tf.Print(0, [n], summarize=100))

    def _exec(self, nodes, feed_dict=None):
        all_nodes = [p for p in self._prints]
        all_nodes.extend(nodes)
        outputs = self._session.run(all_nodes, feed_dict=feed_dict)
        return outputs[len(self._prints):]

    def nearestNeighbors(self, batch_indices, top_k=None, no_self=True):
        (pairwise_distances,) = self._exec([
                self._sample_distances
            ],
            feed_dict = {
                self._sample_indices: batch_indices
            }
        )

        nearest_neighbors = []
        for i in range(len(batch_indices)):
            distance_vector = pairwise_distances[i]
            sorted_neighbors = np.argsort(distance_vector)
            # if skipping the query, remove it from the neighbor list
            # (should be in the 0th position; if it's not, just move on)
            if no_self: 
                if sorted_neighbors[0] == batch_indices[i]: sorted_neighbors = sorted_neighbors[1:]
            if top_k is None: nearest_neighbors.append(sorted_neighbors)
            else: nearest_neighbors.append(sorted_neighbors[:top_k])
        return nearest_neighbors
