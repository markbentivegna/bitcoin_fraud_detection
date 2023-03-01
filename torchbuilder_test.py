import os 
import time 

import torch 

from utilities.dataset_util import DatasetUtilityPyTorch as DUPT

data_util = DUPT()
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