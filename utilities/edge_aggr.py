import torch
from torch_scatter import scatter

def edge_feats_to_node_feats(edge_index, edge_features, reduce='mean'):
    '''
    Given a 2 x E edge index, and a E x d matrix of edge features
    aggregate all edge features into a node feature 

    Reduce may be one of ['sum', 'mean', 'min', 'max']
    '''
    inbound = scatter(
        edge_features, 
        edge_index[1], 
        reduce=reduce,
        dim=0
    )
    
    outbound = scatter(
        edge_features, 
        edge_index[0], 
        reduce=reduce,
        dim=0
    )
    
    bidirectional = scatter(
        edge_features.repeat(2,1),
        edge_index.flatten(),
        reduce=reduce, 
        dim=0
    )
    
    return torch.cat([
        inbound, outbound, bidirectional
    ], dim=1)


if __name__ == '__main__':
    # Test to make sure it works 
    ei = torch.tensor([
        [0,1,2,0,1,1],
        [2,1,0,0,1,2]
    ])
    ef = torch.tensor([
        [0,0,1], 
        [0,1,0],
        [1,0,0],
        [1,0,0],
        [0,1,0],
        [0,0,1]
    ], dtype=torch.float)

    print(
        edge_feats_to_node_feats(ei, ef)
    )

    '''
    Expected output: 
        dst   |  src       |  bi 
      [1,0,0   .5, 0,  .5    .75, 0, .25]
      [0,1,0,  0 ,.33, .66,   0, .8, .2 ]
      [0,0,1,  1 , 0,  0,    .33, 0  .66]

    Actual output
    tensor([
        [1.0000, 0.0000, 0.0000, 0.5000, 0.0000, 0.5000, 0.7500, 0.0000, 0.2500],
        [0.0000, 1.0000, 0.0000, 0.0000, 0.6667, 0.3333, 0.0000, 0.8000, 0.2000],
        [0.0000, 0.0000, 1.0000, 1.0000, 0.0000, 0.0000, 0.3333, 0.0000, 0.6667]
    ])
    '''