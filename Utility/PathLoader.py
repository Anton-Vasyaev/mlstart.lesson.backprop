from Utility.FileSystem import *

def load_keras_pathes(pathes):
    modelPathes = {}
    for path in pathes:
        modelName = file_name_without_extension(path)
        modelPathes[modelName] = path

    return modelPathes