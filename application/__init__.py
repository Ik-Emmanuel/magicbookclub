from flask import Flask
import requests
import json
from IPython.display import display, HTML
import urllib
import importlib
import time
import getopt
import sys
from requests import get, post
from pathlib import Path



app = Flask(__name__, instance_relative_config=True)
SESSION_TYPE = 'filesystem'
app.config.from_object(__name__)




from application import routes

if __name__=="__main__":
    app.run( debug=True) 