import pymc as pm
import numpy as np
import arviz as az

def create_model_example1():
    RANDOM_SEED = 8927
    rng = np.random.default_rng(RANDOM_SEED)

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sigma=1)
        obs = pm.Normal("obs", mu=mu, sigma=1, observed=rng.standard_normal(100))
        idata = pm.sample()

    # return az.plot_trace(idata)
    return idata


def find_parent(graph):
    for node,parents in graph.items():
        if len(parents)<=0:
            return node

def extract_node(graph):
    parent = find_parent(graph)
    #remove parent
    graph = {node:[p for p in parents if p!=parent] for node,parents in graph.items() if node!=parent}

    return parent,graph


def create_model(rv_graph,rv_types):
    with pm.Model() as model:

        vars = []

        graph = rv_graph.copy()

        rvs = {}

        while graph:
            node,graph = extract_node(graph)

            parents = rv_graph[node]
            if len(parents)>0:
                mu = pm.Deterministic(node+'_mu', np.sum([rvs[p] for p in parents]))

            else:
                mu = 0

            if rv_types[node] == 'StudentT':
                rvs[node] = pm.StudentT(node, nu=15, mu=mu, sigma=10)
            else:
                rvs[node] = pm.Normal(node, mu=mu, sigma=1)
    return model


def example_run1():
    rv_graph = {'a':['b','c','d'],'b':['c','d'],'c':['d'],'d':[]}
    rv_types = {'a':'StudentT','b':'Normal','c':'Normal','d':'Normal'}

    model = create_model(rv_graph,rv_types)

    with model:
            idata = pm.sample_prior_predictive(samples=1000, random_seed=rng)