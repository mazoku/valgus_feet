from __future__ import division

import os

def get_names(dir):
    files = os.listdir(dir)
    names = [x[:-4] for x in files]
    return names


if __name__ == '__main__':
    dir = '/home/tomas/Data/Paty/zari/ply/'
    ext = '.ply'

    names = get_names(dir)
    print names