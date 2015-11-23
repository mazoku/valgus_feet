from __future__ import division

import pcl

if __name__ == '__main__':
    fname = '/home/tomas/Data/Paty/zari/ply/augustynova.ply'
    # fname = '/home/tomas/Data/Paty/zari/stl/augustynova.stl'
    cloud = pcl.load(fname)
    print(cloud.size)