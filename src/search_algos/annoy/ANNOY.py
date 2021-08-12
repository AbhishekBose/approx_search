import os
import sys

import annoy
import numpy
sys.path.append(os.path.dirname(os.getcwd()))
from ANN import ANN


class ANNOY(ANN):
    def __init__(self, vectors):
        self.dimension = vectors.shape[1]
        self.vectors = vectors.astype('float32')
        self.index = annoy.AnnoyIndex(self.dimension)

    def build(self, number_of_trees=5):
        print("Building index")
        for i, vec in enumerate(self.vectors):
            self.index.add_item(i, vec.tolist())
        self.index.build(number_of_trees)

    def query(self, vector, k=10):
        indices, distances = self.index.get_nns_by_vector(vector,
                                                          k, include_distances=True)
        return indices, distances

    def query_batch(self, query_vec, k=10):
        raise NotImplementedError


if __name__ == "__main__":
    # test with dummy data
    data = numpy.random.randn(10000, 100).astype(numpy.float32)
    ann = ANNOY(data)
    ann.build()
    ids, distances = ann.query(data[0])
    print(ids)
    print(distances)
