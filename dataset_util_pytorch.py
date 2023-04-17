import torch
from torch_geometric.data import Data
from tqdm import tqdm 

from resources import constants




class DatasetUtilityPyTorch():
    def __init__(self, dataset=constants.BITCOIN_DATASET_DIR):
        self.DATA = dataset + '/'
        self.nodemap = dict()
        self.nid = 0 
        self.nid_to_node_name = []

        self.N_EDGES = constants.N_EDGES if dataset==constants.BITCOIN_DATASET_DIR \
            else constants.N_AUGMENTED_EDGES


    def _get_or_add(self, node):
        if not node in self.nodemap:
            self.nodemap[node] = self.nid 
            self.nid_to_node_name.append(int(node))
            self.nid += 1
        
        return self.nodemap[node]


    def build_edge_list(self):
        f = open(self.DATA + constants.EDGELIST_FILE_ORIGINAL, 'r')
        f.readline() # Skip header

        srcs,dsts = [],[]
        prog = tqdm(desc='Building edge list', total=self.N_EDGES)
        line = f.readline()
        while(line):
            src,dst = line.strip().split(',')

            # I think any new transaction from the augmented data
            # involves completely new nodes for both src and dst
            # E.g. the first transaction with a negative is listed as 
            # -28784671,152039820
            # but there is no record of 152039820. There is, however
            # a record associated with node -152039820 so I think that is
            # the (poorly documented) implication
            if src.startswith('-'):
                dst = '-'+dst

             # Convert to sequential IDs
            src = self._get_or_add(src) 
            dst = self._get_or_add(dst) 

            srcs.append(src) 
            dsts.append(dst)
            
            prog.update()
            line = f.readline()

        f.close()
        prog.close()

        # All nodes now added. Finalize by making it a tensor
        self.nid_to_node_name = torch.tensor(self.nid_to_node_name)
        return torch.tensor([srcs,dsts])
    

    def build_features(self):
        f = open(self.DATA + constants.FEATURES_FILE_ORIGINAL, 'r')
        x = torch.zeros(len(self.nodemap), constants.FEAT_DIM)

        prog = tqdm(desc='Building features', total=len(self.nodemap))
        line = f.readline()
        while(line):
            tokens = line.strip().split(',')
            node, feats = tokens[0], tokens[1:]
            feats = torch.tensor([float(f) for f in feats])

            # This *shouldn't* throw a key error assuming
            # build_edge_list was called first
            if (nid := self.nodemap.get(node)) is None:
                print("Skipping feature for node %s that doesn't appear in edgelist" % node)
            x[nid] = feats 

            prog.update()
            line = f.readline()

        f.close() 
        prog.close()
        return x 
    

    def build_labels(self):
        f = open(self.DATA + constants.CLASSES_FILE_ORIGINAL, 'r') 
        f.readline() # Skip headers 
        ys = torch.zeros(len(self.nodemap))

        prog = tqdm(desc='Building ground truth', total=len(self.nodemap))
        line = f.readline()

        # Tiny bit faster than if statements
        ymap = {'unknown':0, '1':1, '2':2, 'suspicious': 3}

        while(line):
            node,y = line.strip().split(',')
            ys[self.nodemap[node]] = ymap[y]

            prog.update()
            line = f.readline()

        f.close()
        prog.close()
        return ys 
    
    def build_dataset(self):
        ei = self.build_edge_list()
        
        # These two can be run in parallel but build_labels takes
        # <1s so it's not really worth it
        x = self.build_features()
        y = self.build_labels()
        
        return Data(
            x=x, edge_index=ei, y=y,
            num_nodes=x.size(0)
        )


    def split_subgraphs(self, data):
        '''
        Time stamps associated with nodes in each edge
        supposed to be fully connected components accross each 
        timestamp. Want to pull out individual subgraphs spanning
        specific timestamps
        '''
        times = data.x[data.edge_index[0]][:,0]
        spans = times.unique()

        graphs = []
        for span in spans:
            ei = data.edge_index[:, times==span]
            nodes, reindex = ei.unique(return_inverse=True)

            # Don't bother including timestamp. It's superfluous
            x = data.x[nodes][:,1:]
            y = data.y[nodes]
            graphs.append(Data(
                x=x, y=y, edge_index=reindex, 
                ts=span.item(), nid_to_node_name=self.nid_to_node_name[nodes]
            ))

        return graphs 