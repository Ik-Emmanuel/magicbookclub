from application import app
import datetime
from flask import render_template, request, json, jsonify, Response, redirect, flash, url_for, session, make_response
import os
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
import time


import pickle

import numpy as np
import scipy
import pandas as pd
import seaborn as sns
sns.set_style('white')
import math
import random
import sklearn
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors


booksdata = pd.read_csv('BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
booksdata.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']
usersdata = pd.read_csv('BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
usersdata.columns = ['userID', 'Location', 'Age']
ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
ratings.columns = ['userID', 'ISBN', 'bookRating']
booksdata=booksdata

@app.route('/')
def index():
    return render_template('index.html' )


@app.route('/user')
def user():
    return render_template('firstform.html' )

@app.route('/books',  methods=["GET", "POST"])
def books():
    name= "User"
    name = request.form.get("fname")
    age = request.form.get("age")
    location = request.form.get("location")

    
    samplebooks = booksdata.sample(n=12)

    booksdict = samplebooks.to_dict()
   
    return render_template('bookpage.html' , name=name, age=age, location=location, data=booksdict )


@app.route('/postbookselect',  methods=["GET", "POST"])
def postbookselect():
    bookindex = request.form.get("book")
    booksingle = booksdata.filter(like=bookindex, axis=0)
    Title = booksingle['bookTitle'].values[0]
    Url = booksingle['imageUrlL'].values[0]
    Publisher = booksingle['publisher'].values[0]
    Author = booksingle['bookAuthor'].values[0]
    year =  booksingle['yearOfPublication'].values[0]
    ISBN = booksingle['ISBN'].values[0]

  
    return render_template('bookselect.html' , data=bookindex, title= Title, url=Url, publisher=Publisher, author=Author, year=year, ISBN=ISBN)


@app.route('/api/book_select/<value>')
def sent(value):
    try:

        if value:
            booksingle = booksdata.filter(like=value, axis=0)
            # print(value)

            # print(booksingle)
            # print(booksingle['ISBN'].values[0])
            # print(booksingle['bookTitle'].values[0])
            # print(booksingle['imageUrlL'].values[0])

            Title = booksingle['bookTitle'].values[0]
            Url = booksingle['imageUrlL'].values[0]

            result = {'message': Title, 'link': Url}
            

        else:
            result = {'error': 'Sentiment change failed, please try again.'}

    except Exception as e:
        print(str(e))

        result = {'error': 'Sentiment change failed, please try again.'}

    return jsonify(result)


@app.route('/bookrecom',  methods=["GET", "POST"])
def bookrecom():
    bookindex = request.form.get('book')

    booksingle = booksdata.filter(like=bookindex, axis=0)
    Title = booksingle['bookTitle'].values[0]
    Url = booksingle['imageUrlL'].values[0]
    Publisher = booksingle['publisher'].values[0]
    Author = booksingle['bookAuthor'].values[0]
    year =  booksingle['yearOfPublication'].values[0]
    ISBN = booksingle['ISBN'].values[0]
  
    random = booksdata.sample(n = 12)
    print(random.columns)
    imageurl =list(random["imageUrlL"])
    indexes = list(random["bookTitle"])
    time.sleep(10)
  
   
    return render_template('bookrecom.html', data=bookindex, title= Title, url=Url, publisher=Publisher, author=Author, year=year, ISBN=ISBN,  imageurl=imageurl, indexes=indexes )