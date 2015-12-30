from __future__ import division

import warnings

import numpy as np
from mayavi import mlab
import main
import skimage.exposure as skiexp
import matplotlib.pyplot as plt
import sklearn.cluster as sklclu
import tools

import os


def clustering_kmeans(features):
    X = np.vstack((x for x in features)).T
    n_clusters = 10
    kmeans = sklclu.KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    labels = kmeans.labels_
    # hist, bin_edges = np.histogram(labels, bins=n_clusters, density=True)
    hist, bins = skiexp.histogram(labels, nbins=n_clusters)
    max_lab = labels[np.argmax(hist)]
    max_labels = labels == max_lab

    return max_labels


def clustering_thresh(features, eps=4):
    peaks = []
    max_labels = np.ones(len(features[0]), dtype=np.bool)

    for i in features:
        hist, bins = skiexp.histogram(i, nbins=180)
        peak = bins[np.argmax(hist)]
        peaks.append(peak)

        max_labels *= (peak - eps < i) * (i < peak + eps)

    # print 'peaks = ', peaks

    return max_labels


def run(fname):
    vertices, faces, normals_v = main.read_ply(fname)
    if faces.shape[1] == 4:
        faces = faces[:, 1:]

    dirs = fname.split('/')
    name = dirs[-1][:-4]
    data_dir = os.path.join('/'.join(dirs[:-1]), 'npy')
    fig_dir = os.path.join('/'.join(dirs[:-1]), 'figs')

    # DIHEDRAL ANGLES ----------------------------------------------
    print 'Calculating dihedral angles ...',
    dih_xy, dih_xz, dih_yz = main.get_angles(normals_v, planes=['xy', 'xz'])
    print 'done'

    # plt.figure()
    # plt.hold(True)
    # plt.plot(bins_xy, hist_xy, 'r-')
    # plt.plot(bins_xz, hist_xz, 'g-')
    # plt.show()

    # KMEANS -------------------------------------------------------
    print 'Clustering ...',
    # max_labels = clustering_kmeans((dih_xy, dih_xz, dih_yz))
    #
    # mlab.clf()
    # mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], faces, scalars=max_labels.astype(np.int))
    # mlab.view(-90, 90)
    # mlab.show()

    max_labels = clustering_thresh((dih_xy, dih_xz))
    # mlab.clf()
    # mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], faces, scalars=max_labels.astype(np.int))
    # mlab.view(-90, 90)
    # mlab.show()

    print 'done'

    # FITTING PLANE ------------------------------------------------
    print 'Fitting plane ...',
    vertices = main.fitting_plane(vertices, faces, max_labels)
    print 'done'

    # SEGMENTING FEET ----------------------------------
    print 'Segmenting feet ...',
    foot_l, foot_l_mask, foot_r, foot_r_mask, feet_mask = main.feet_segmentation(vertices, faces)
    # mlab.clf()
    # mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], faces, scalars=feet_mask.astype(np.int))
    # mlab.view(-90, 90)
    # mlab.show()
    print 'done'

    # PT-1 ... LYTKO = CALF -------------------------------------------------------
    print 'Finding calf points ...',
    calf_l = main.calf_point(foot_l)
    calf_r = main.calf_point(foot_r)
    print 'done'

    # PT-2 ... ACHILOVKA -----------------------------------------------------------
    print 'Finding Achilleus\' points ...',
    achill_l = main.achill_point(foot_l, 'l')
    achill_r = main.achill_point(foot_r, 'r')
    print 'done'

    # PT-3 & 4 ... HEEL POINTS ------------------------------------------------------
    print 'Finding heel points ...',
    max_h = 100
    heel_cut_l, pts_l, mean_pt_l = main.cut_heel(foot_l, 'l', max_h)
    closest_pt_l, widest_pt_l = main.heel_points(foot_l, heel_cut_l, max_h, show=False)
    heel_cut_r, pts_r, mean_pt_r = main.cut_heel(foot_r, 'r', max_h)
    closest_pt_r, widest_pt_r = main.heel_points(foot_r, heel_cut_r, max_h, show=False)
    print 'done'

    left_points = [calf_l, achill_l, closest_pt_l, widest_pt_l]
    right_points = [calf_r, achill_r, closest_pt_r, widest_pt_r]

    # ANGLE CALCULATION -------------------------------------------------------------
    print 'Calculating angles ...'
    theta_l, poly1_l, poly2_l = main.angle(np.array(left_points))
    theta_r, poly1_r, poly2_r = main.angle(np.array(right_points))
    print '\t angle L = %.1f' % theta_l
    print '\t angle R = %.1f' % theta_r
    print 'done'

    # fig_l = main.draw_lines(foot_l, poly1_l, poly2_l, left_points)
    # # fig_l.savefig(os.path.join(fig_dir, name) + '_lines_L.png')
    # fig_r = main.draw_lines(foot_r, poly1_r, poly2_r, right_points)
    # # fig_r.savefig(os.path.join(fig_dir, name) + '_lines_R.png')
    # plt.show()


if __name__ == '__main__':
    warnings.filterwarnings('error')

    months = ['zari', 'rijen']
    month = months[0]
    dir_all = '/home/tomas/Data/Paty/' + month + '/ply/'
    dir_processed = '/home/tomas/Data/Paty/' + month + '/ply/npy/'

    all_names, proc_names, failed_names = main.process_filenames(dir_all, dir_processed)

    proc_files = tools.get_names('/home/tomas/Data/Paty/' + month + '/ply/figs/', ext='.png')
    proc_names = [x[:-8] for x in proc_files if 'lines' in x]
    failed_names = [x for x in all_names if x not in proc_names]

    # print 'all:', len(all_names)
    # print all_names
    #
    # print 'processed:', len(proc_names)
    # print proc_names
    #
    # print 'failed:', len(failed_names)
    # print failed_names

    # name = failed_names[0]
    # failes_names = [failed_names[0],]
    for (i, name) in enumerate(failed_names):
        fname = os.path.join(dir_all[:-1], name + '.ply')
        print '--  Processing  #%i/%i - %s  --' % (i + 1, len(failed_names), name)
        # run(fname)

        # ----------------------------------------------------------------------------------------------------
        # left_points = np.load(os.path.join(dir_processed, name + '_left_points.npy'))
        # right_points = np.load(os.path.join(dir_processed, name + '_right_points.npy'))
        # foot_l = np.load(os.path.join(dir_processed, name + '_foot_l.npy'))
        # foot_r = np.load(os.path.join(dir_processed, name + '_foot_r.npy'))
        #
        # theta_l, poly1_l, poly2_l = main.angle(np.array(left_points))
        # theta_r, poly1_r, poly2_r = main.angle(np.array(right_points))
        #
        # fig_l = main.draw_lines(foot_l, poly1_l, poly2_l, left_points)
        # # fig_l.savefig(os.path.join(fig_dir, name) + '_lines_L.png')
        #
        # fig_r = main.draw_lines(foot_r, poly1_r, poly2_r, right_points)
        # fig_size = plt.rcParams["figure.figsize"]
        # # fig_r.savefig(os.path.join(fig_dir, name) + '_lines_R.png')
        # plt.show()

        #
        basename = os.path.join(dir_processed, name)
        vertices = np.load(basename + '_vertices.npy')
        faces = np.load(basename + '_faces.npy')
        feet_mask = np.load(basename + '_feet_mask.npy')
        foot_l_mask = np.load(basename + '_foot_l_mask.npy')
        foot_r_mask = np.load(basename + '_foot_r_mask.npy')
        max_h = 100

        left_points = np.load(basename + '_left_points.npy')
        right_points = np.load(basename + '_right_points.npy')
        foot_l = np.load(basename + '_foot_l.npy')
        foot_r = np.load(basename + '_foot_r.npy')

        calf_l = main.calf_point(foot_l)
        calf_r = main.calf_point(foot_r)

        achill_l = main.achill_point(foot_l, 'l')
        achill_r = main.achill_point(foot_r, 'r')

        heel_cut_l, pts_l, mean_pt_l = main.cut_heel(foot_l, 'l', max_h)
        closest_pt_l, widest_pt_l = main.heel_points(foot_l, heel_cut_l, max_h, show=False)
        heel_cut_r, pts_r, mean_pt_r = main.cut_heel(foot_r, 'r', max_h)
        closest_pt_r, widest_pt_r = main.heel_points(foot_r, heel_cut_r, max_h, show=False)

        left_points = [calf_l, achill_l, closest_pt_l, widest_pt_l]
        right_points = [calf_r, achill_r, closest_pt_r, widest_pt_r]
        left_points = np.array(left_points)
        right_points = np.array(right_points)

        theta_l, poly1_l, poly2_l = main.angle(np.array(left_points))
        theta_r, poly1_r, poly2_r = main.angle(np.array(right_points))

        fig_dir = os.path.join((dir_processed[:-5]), 'figs_devel')

        print '\tSaving figure...',
        main.save_figures(fig_dir, name, vertices, faces, feet_mask, foot_l_mask, foot_r_mask, max_h,
                         heel_cut_l, heel_cut_r, pts_l, pts_r, mean_pt_l, mean_pt_r,
                         foot_l, foot_r, poly1_l, poly2_l, poly1_r, poly2_r, left_points, right_points)
        print 'done'
