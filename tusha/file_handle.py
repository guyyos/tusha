import os
import cloudpickle
from app import UPLOAD_DIRECTORY


def get_upload_dir(session_id):
    return UPLOAD_DIRECTORY+'/'+session_id+'/objs/'

def get_full_name(session_id,name):
    dirname = get_upload_dir(session_id)
    fname = dirname+name+'.obj'
    return fname


def clean_user_model(session_id):
    
    fname = get_full_name(session_id,'complete_model')
    if os.path.isfile(fname):
        os.remove(fname)


def load_file(name, session_id):
    fname = get_full_name(session_id,name)

    if not os.path.isfile(fname):
        return None

    with open(fname, 'rb') as handle:
        obj = cloudpickle.load(handle)
    return obj


def save_file(name, session_id, content):
    dirname = get_upload_dir(session_id)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    fname = get_full_name(session_id,name)

    """Decode and store a file uploaded with Plotly Dash."""
    with open(fname, 'wb') as handle:
        cloudpickle.dump(content, handle)
