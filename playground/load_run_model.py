import cloudpickle
import pymc as pm

fname = '/home/guyyos/data/complete_model.obj'

def sample_model(model):
    with model:
        idata = pm.sample(1000, tune=1000)
        # idata = jax.sample_numpyro_nuts(1000, tune=1000) # faster
        # idata = jax.sample_blackjax_nuts() # not working

        pm.sample_posterior_predictive(idata, extend_inferencedata=True)

    # post = az.extract(idata)
    # pm.model_to_graphviz(model)
    return model, idata


with open(fname, 'rb') as handle:
    model = cloudpickle.load(handle)

print('loaded file')

model, idata = sample_model(model)