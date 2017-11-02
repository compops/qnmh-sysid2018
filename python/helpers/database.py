"""Helpers for storing and loading results from Mongo Databases."""

from pymongo import MongoClient

def db_open(db_name, collection_name):
    """ Open a database connection.

        Opens a connection to the database and provides a collection.

        Args:
            db_name: name of the database.
            collection_name: name of the collection.

        Returns:
           pyMongo handle to the collection.

    """
    client = MongoClient()
    database = client[db_name]
    return database[collection_name]

def db_insert_results(collection, post_name, output=None, data=None,
                      settings=None):
    """ Inserts results as a post into a collection."

        Args:
            collection: name of the collection.
            post_name: name of the post to insert.
            output: MCMC output (dict) to store.
            data: data dict to store.
            settings: settings dict to store.

        Returns:
           Nothing.

    """
    post = {'name': post_name,
            'output': output,
            'data': data,
            'settings': settings
           }
    post_id = collection.insert_one(post).inserted_id
    print("Saved results to database with id: " + str(post_id) + ".")
