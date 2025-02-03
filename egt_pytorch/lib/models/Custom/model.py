from ..egt_graph import EGT_GRAPH

class EGT_Custom(EGT_GRAPH):
    def __init__(self, **kwargs):
        super().__init__(output_dim=1, **kwargs)
        
    def output_block(self, g):
        h = super().output_block(g)
        return h.squeeze(-1)
