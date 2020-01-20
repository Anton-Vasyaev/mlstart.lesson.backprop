import os

def get_all_files_from_deep(directory_name):
    files_list = []

    files = os.listdir(directory_name)
    for node_element in files:
        full_path_element = directory_name + os.sep + node_element
        if os.path.isdir(full_path_element):
            files_list += get_all_files_from_deep(full_path_element)
        else:
            files_list.append(full_path_element)

    return files_list


def file_name(path):
    return os.path.basename(path)


def file_name_without_extension(path):
    return os.path.splitext(os.path.basename(path))[0]


def file_extension(path):
    return os.path.splitext(os.path.basename(path))[1]


def directory_name(path):
    return os.path.dirname(path)


def createPath(path):
    directories = path.split(os.sep)
    createPath = ""

    for directory in directories:
        createPath = os.path.join(createPath, directory)
        if not os.path.exists(createPath):
            os.mkdir(createPath)
