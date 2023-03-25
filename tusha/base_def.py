from enum import Enum


COUNTER_FACTUAL_NUM_POINTS = 100
COUNTER_FACTUAL_SMOOTH_NUM_POINTS = 55  # note that COUNTER_FACTUAL_SMOOTH_NUM_POINTS<COUNTER_FACTUAL_NUM_POINTS

class PredictionSummary:
    def __init__(self,predictor,target_pred,mu_pred,target_hdi,mu_hdi,mu_mean,predictor_vals,cat_codes):
        self.predictor = predictor
        self.target_pred = target_pred
        self.mu_pred = mu_pred
        self.target_hdi = target_hdi
        self.mu_hdi = mu_hdi
        self.mu_mean = mu_mean
        self.predictor_vals = predictor_vals
        self.cat_codes = cat_codes
        self.predictors = []


class FeatureType(Enum):
    BOOL = 1
    CATEGORICAL = 2
    NUMERICAL = 3
    ORDINAL = 4
    COUNT = 5

    def is_categorical(self):
        return self == FeatureType.BOOL or self == FeatureType.CATEGORICAL


class FeatureInfo:
    def __init__(self,featureType):
        self.featureType = featureType

class DAGNode:
    def __init__(self,name,info,parent_vars,child_vars):
        self.name = name
        self.info = info
        self.parent_vars = parent_vars
        self.child_vars = child_vars

from graphlib import TopologicalSorter

class DAGraph:
    #relation (parent,child)
    def __init__(self,nodes_info,relations):
        self.nodes = {name:DAGNode(name,info,[],[]) for name,info in nodes_info.items()}

        for parent,child in relations:
            self.nodes[parent].child_vars.append(child)
            self.nodes[child].parent_vars.append(parent)

    def topological_order(self):
        graph = {n:node.child_vars for n,node in self.nodes.items()}

        ts = TopologicalSorter(graph)
        return list(ts.static_order())[::-1]

    def get_downstream_vars(self,node_name):
        
        downstream_vars = set([])
        visited = set([])

        def recurs_downstream(node_name):
            if node_name in visited:
                return
            visited.add(node_name)

            node = self.nodes[node_name]
            for child_name in node.child_vars:
                downstream_vars.add(child_name)
                recurs_downstream(child_name)
        
        recurs_downstream(node_name)

        return downstream_vars




