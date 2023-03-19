import os 
import time 

import torch 

from resources import constants
from utilities.dataset_util import DatasetUtilityPyTorch as DUPT
from utilities.graph_util import find_ccs

'''
data_util = DUPT(dataset=constants.AUGMENTED_BITCOIN_DATASET_DIR)
st = time.time()
data = data_util.build_dataset()
en = time.time()

print(f"\nTook {en - st} seconds to generate graph")
print("\tEdge Index:".ljust(20) + f"(2 x {data.edge_index.size(1)})")
print(f"\tFeature matrix:".ljust(20) + f"({data.x.size(0)} x {data.x.size(1)})")
print(f"\tClasses:".ljust(20) + f"(1 x {data.y.size(0)})")

torch.save(data, 'data.pt')
fs = os.path.getsize('data.pt') 
print(f"Output: {fs//1000000}MB")

splits = data_util.split_subgraphs(data)
for s in splits:
    if s.ts == -1:
        torch.save(s, 'resources/augmented.pt')
'''

g = torch.load('resources/augmented.pt')
ccs = find_ccs(g)
print(len(ccs))

cc_map = torch.zeros(g.x.size(0), dtype=torch.long)
for i,cc in enumerate(ccs):
    idx = torch.tensor(list(cc))
    cc_map[idx] = i 

g.connected_components = cc_map 
torch.save(g, 'resources/augmented.pt')