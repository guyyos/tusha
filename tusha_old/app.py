import dash_bootstrap_components as dbc
import dash
from dash import DiskcacheManager, CeleryManager, Input, Output, html
import os
from flask_caching import Cache

os.environ['REDIS_URL'] = 'redis://127.0.0.1:6379' #'redis://127.0.0.1:6379'

app_name = 'tusha_app'# __name__
print(app_name)
print('guyguy make sure to run in this folder >  celery -A app.celery_app worker --loglevel=INFO')
if False and 'REDIS_URL' in os.environ:
    # Use Redis & Celery if REDIS_URL set as an env variable
    from celery import Celery
    celery_app = Celery(app_name, broker=os.environ['REDIS_URL'], backend=os.environ['REDIS_URL'])
    background_callback_manager = CeleryManager(celery_app)
    print('guyguy: Using celery!! ')

else:
    # Diskcache for non-production apps when developing locally
    import diskcache
    cache = diskcache.Cache("./cache")
    background_callback_manager = DiskcacheManager(cache)
    print('guyguy: Using diskcache!! ')



app = dash.Dash(app_name, suppress_callback_exceptions=True,  external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],#external_stylesheets=[dbc.themes.FLATLY,dbc.icons.BOOTSTRAP],
                background_callback_manager=background_callback_manager,
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )

app.title = 'Tusha Causal'

cache = Cache(app.server, config={
    # try 'filesystem' if you don't want to setup redis
    'CACHE_TYPE': 'redis',
    'CACHE_REDIS_URL': os.environ.get("REDIS_URL", "redis://127.0.0.1:6379")
})
# app.config.suppress_callback_exceptions = True
