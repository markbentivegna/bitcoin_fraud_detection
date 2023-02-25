from utilities.dataset_util import DatasetUtility
import numpy as np
import time
dataset_util = DatasetUtility()
u_tuples = dataset_util.get_u_tuples()
edge_list = dataset_util.get_edge_list()
N = dataset_util.get_transaction_count()

adj_matrix = np.zeros((N,N))
start = time.time()
for edge in edge_list:
    try:
        source_index = int(dataset_util.lookup_node_index(edge[0]))
        destination_index = int(dataset_util.lookup_node_index(edge[1]))
        adj_matrix[source_index][destination_index] = 1
    except Exception as e:
        print(e)

end = time.time()
print(f"took {end - start} seconds to generate adjacency matrix")
print("foo")
