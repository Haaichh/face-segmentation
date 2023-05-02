import os

def create_directory(dir):
    """Creates directory"""

    try:
        os.mkdir(dir)
        print('Directory ' + dir + ' created')
    except FileExistsError:
        print('Directory ' + dir + ' already exists')