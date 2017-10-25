"""Helpers for storing and loading results from Mongo Databases."""

from pymongo import MongoClient

def db_open(db_name, collection_name):
    """Opens a connection to the database and provides a collection."""
    client = MongoClient()
    database = client[db_name]
    return database[collection_name]

def db_insert_results(collection, post_name, output=None, data=None,
                      settings=None):
    """Inserts results as a post into a collection."""
    post = {'name': post_name,
            'output': output,
            'data': data,
            'settings': settings
           }
    post_id = collection.insert_one(post).inserted_id
    print("Saved results to database with id: " + str(post_id) + ".")
