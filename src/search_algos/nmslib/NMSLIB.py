import nmslib
import numpy
import sys
import os

sys.path.append(os.path.dirname(os.getcwd()))
from ANN import ANN


class NMSLIB(ANN):
    def __init__(self, vectors):
        self.index = nmslib.init(method='hnsw', space='cosinesimil')
        self.dimension = vectors.shape[1]
        self.vectors = vectors.astype('float32')

    def build(self):
        print("Building index")
        self.index.addDataPointBatch(self.vectors)
        self.index.createIndex({'post': 2}, print_progress=True)

    def query(self, query_vec, k=10):
        ids, distances = self.index.knnQuery(query_vec, k=k)
        return ids, distances

    def query_batch(self, query_vec, k=10):
        neighbours = self.index.knnQueryBatch(query_vec, k=k, num_threads=4)
        return neighbours


if __name__ == "__main__":
    # test with dummy data
    data = numpy.random.randn(10000, 100).astype(numpy.float32)
    nms = NMSLIB(data)
    nms.build()
    ids, distances = nms.query(data[0])
