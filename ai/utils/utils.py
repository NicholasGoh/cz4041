import os

def write_line(path, line):
    mode = 'a' if os.path.isfile(path) else 'w+'
    with open(path, mode) as f:
        f.write(f'{line}\n')
def read_file(path):
    assert os.path.isfile, 'file does not exist'
    mode = 'r'
    with open(path, mode) as f:
        return f.readlines()
