import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

from edflow.util import pp2mkdtable
from edflow.data.util import adjust_support

from abc_pose.abcnet.heatmaps import kp2heat

from tqdm import tqdm, trange

from abc_pose.pose_sampling.keypoint_models import ALPHAPOSE_COCO_TUPLES
from abc_pose.pose_sampling.dist import kp2ang


def plot_seqs(dset, n=10, nframes=10):
    '''Given a :class:`MultiPersonDataset` show us a visualization of some
    videos and all person sequences in it.

    .. codeblock::

               t ->
               +---+---+-     -+---+
        frames | 1 | 2 |  ...  | N |
               +---+---+-     -+---+
                   +---+-     -+---+
        p1         | 1 |  ...  |   |
                   +---+-     -+---+
               +---+---+-     
        p2     | 1 | 2 |  ... 
               +---+---+-     

               ...


               +---+---+-     -+---+
        pn     | 1 | 2 |  ...  | N |
               +---+---+-     -+---+
    '''

    video_paths = dset.labels['video_path']
    video_names = sorted(np.unique(video_paths))[:n]

    for name in video_names:
        indices = np.where(video_paths == name)[0]

        n_frames = max(dset.labels['frame_idx'][indices])
        step = n_frames // nframes

        frame_paths = np.unique(dset.labels['frame_path'][indices])
        frames = sorted(frame_paths)

        person_boxes = {}
        for idx in indices:
            example = dset[idx]
            bbox = example['bbox']
            pid = example['person_id']
            sid = example['sequence_idx']
            fid = example['frame_idx']
            vid = example['video_path']
            crop_path= '{:0>5}-p{:0>3}-s{:0>3}-f{:0>3}.png'.format(
                    idx,
                    pid,
                    sid,
                    fid
                    )
            crop_path = os.path.join(vid + '_crops', crop_path)

            if pid not in person_boxes:
                person_boxes[pid] = []

            person_boxes[pid] += [[bbox, sid, fid, crop_path]]

        person_boxes = {k: sorted(v, key=lambda v: v[1]) for k, v in person_boxes.items()}

        fsize = plt.rcParams.get('figure.figsize')
        fsize[1] = (len(person_boxes) + 2) / 5 * fsize[1]
        f, ax = plt.subplots(1, 1, figsize=fsize)

        # All frames
        mean = half_range = n_frames / 2.
        ax.errorbar(mean, -1, xerr=half_range, ls='--', elinewidth=3, capsize=5)

        for i in np.concatenate([np.arange(n_frames, step=step), [n_frames]]):
            try:
                im1 = plt.imread(frames[i])
            except Exception as e:
                continue
            im1box = OffsetImage(im1, zoom=0.10)
            im1box.image.axes = ax

            ab = AnnotationBbox(im1box, [i, -1],
                        xybox=(i, -0.5),
                        xycoords='data',
                        boxcoords="data",
                        pad=0.0,
                        arrowprops=dict(
                            arrowstyle="-",
                            connectionstyle="angle,angleA=0,angleB=90,rad=3")
                        )

            ax.add_artist(ab)

        # All persons
        for pid, pdata in person_boxes.items():
            # mean = pdata[0][2] + (pdata[-1][2] - pdata[0][2]) / 2.
            # full_range = pdata[-1][2] - pdata[0][2]
            # half_range = full_range / 2.
            # ax.errorbar(mean, pid, xerr=half_range, ls='-', elinewidth=3, capsize=5)

            X = [pd[2] for pd in pdata]
            Y = [pid for i in range(len(pdata))]
            ax.scatter(X, Y, marker='s')

            for i in np.concatenate([np.arange(len(pdata), step=step), [len(pdata) - 1]]):
                try:
                    im1 = plt.imread(pdata[i][-1])
                except Exception as e:
                    print(e)
                    continue
                im1box = OffsetImage(im1, zoom=0.30)
                im1box.image.axes = ax

                frame_idx = pdata[i][2]
                ab = AnnotationBbox(im1box, [frame_idx, pid],
                            xybox=(frame_idx, pid+0.5),
                            xycoords='data',
                            boxcoords="data",
                            pad=0.0,
                            arrowprops=dict(
                                arrowstyle="-",
                                connectionstyle="angle,angleA=0,angleB=90,rad=3")
                            )

                ax.add_artist(ab)

        ax.set_xlabel('frame')
        ax.set_ylabel('person id (-1 is full frame)')
        name_parts = splitall(name)
        title_str = ''
        for npr in name_parts:
            if len(title_str + npr) >= 50:
                title_str += '\n'
            title_str += npr + '/'
        ax.set_title(title_str)

        savename = '{}-person_seq_lengths.png'.format(name)
        f.savefig(savename, dpi=600)
        print('Saved plot at {}'.format(savename))

        plt.close()


def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def plot_app_variance_at_similiar_pose(MP, n_app=10, n_pose=10, sp=None,
                                       filter_by_app=True, only_other_vid=True,
                                       dist_fn=None, dist_fn_kwargs={},
                                       only_one_ex_per_app=False):
    np.random.seed(42)

    ref_pose_idx = np.random.randint(len(MP))

    all_pids = np.array(MP.labels['pid'])

    all_poses = np.array(MP.labels['keypoints_rel'])
    ref_pose = all_poses[ref_pose_idx]
    print(ref_pose.shape)
    print(all_poses.shape)

    ref_pose_ = np.expand_dims(ref_pose, 0)
    # Distance of reference to all other poses to sort them accordingly
    diff = all_poses - ref_pose_
    print(diff.shape)
    dist2ref = np.mean(np.linalg.norm(diff, axis=-1), axis=-1)

    print('d2r', dist2ref.shape)

    # Get n_pose equally spaced poses
    sorted_indices = np.argsort(dist2ref)
    choice_of_choice = np.round(np.linspace(0, len(MP)-1, n_pose)).astype(int)
    choice_indices = sorted_indices[choice_of_choice]
    print(choice_indices.shape)

    examples = MP[choice_indices]
    examples_indices = choice_indices
    example_distances = dist2ref[choice_indices]
    var_on_examples = []
    var_indices = []

    apps = np.random.choice(np.unique(all_pids), size=n_app)
    print(apps.shape)

    for idx, ex in zip(choice_indices, examples):
        kps = ex['keypoints_rel']
        pid = ex['pid']

        # Distances between current pose and all others
        kps_ = np.expand_dims(kps, 0)
        if dist_fn is None:
            diff = all_poses - kps_
            dist2ref = np.linalg.norm(diff, axis=-1)
            dist2ref = np.mean(dist2ref, axis=-1)
        else:
            dist2ref = dist_fn(all_poses, kps_, **dist_fn_kwargs)

        dist2ref = np.argsort(dist2ref)
        pids_ = all_pids[dist2ref]

        var_ex = []
        var_idx = []

        if not filter_by_app:
            coc = np.round(np.linspace(0, n_app*25, n_app)).astype(int)
            if only_other_vid:
                choice_indices = dist2ref[pids_ != pid][coc]
            else:
                choice_indices = dist2ref[coc]

            var_ex = MP[choice_indices]
            var_idx = choice_indices

        else:
            for app in apps:
                dist2ref_ = dist2ref[pids_ == app]

                var_ex += [MP[dist2ref_[0]]]
                var_idx += [dist2ref_[0]]

        var_on_examples += [var_ex]
        var_indices += [var_idx]

        if only_one_ex_per_app:
            all_poses = all_poses[all_pids != pid]
            all_pids = all_pids[all_pids != pid]

    f, AX = plt.subplots(n_app+2, n_pose, figsize=(n_pose, n_app*2))

    for r_idx, Ax in enumerate(AX):
        for c_idx, ax in enumerate(Ax):
            if r_idx == 0:
                kps = examples[c_idx]['keypoints_rel'] * 256
                # ax.scatter(kps[..., 0], kps[..., 1], marker='.')
                heat = np.mean(kp2heat(kps, min_std=10), axis=-1)
                heat /= heat.max()
                print(heat.shape, heat.min(), heat.max())
                ax.imshow(heat)
                ax.axis('off')
                ax.set_aspect(1)
                # ax.set_ylim(1, 0)
                ax.set_title('$\Delta={:0.3}$'.format(example_distances[c_idx]), fontsize=6)
            elif r_idx == 1:
                crop_path = examples[c_idx]['crop_path']
                ax.imshow(plt.imread(crop_path))
                ax.axis('off')
            else:
                crop_path = var_on_examples[c_idx][r_idx - 2]['crop_path']
                ax.imshow(plt.imread(crop_path))
                ax.axis('off')

            if c_idx == 0 and r_idx > 1 and filter_by_app:
                ax.axis('on')
                ax.set_ylabel(apps[r_idx-2], rotation=0, horizontalalignment='right', fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])

    if sp is None:
        sp = 'app_vs_pose_var.png'

    if filter_by_app:
        sp = sp.replace('.png', '_no_app.png')
    if only_other_vid:
        sp =sp.replace('.png', '_other.png')

    f.savefig(sp, dpi=300)


def plot_dset_statistics(MP, n_pose=5):
    '''Plot histogram of kps dists and pid frames'''

    f, [ax1, ax2] = plt.subplots(2, 1)

    np.random.seed(42)

    ref_pose_idx = np.random.randint(len(MP))

    all_pids = np.array(MP.labels['pid'])
    uniq_pids = np.unique(all_pids)
    pid_vals = np.arange(len(uniq_pids))

    all_pids_as_nums = apn = np.zeros([len(all_pids)], dtype=int)
    for pid, val in zip(uniq_pids, pid_vals):
        all_pids_as_nums[all_pids == pid] = val

    binwidth = 1
    bins = np.arange(min(apn), max(apn) + binwidth, binwidth)
    print(bins.shape)
    print(len(uniq_pids))

    ax2.hist(all_pids_as_nums, bins=bins, ec='k', lw=0.5)
    ax2.set_xticks(bins + 0.5)
    ax2.set_xticklabels(uniq_pids, rotation=90, verticalalignment='top')
    ax2.set_ylabel('Count')
    ax2.set_xlabel('Person ID')

    all_poses = np.array(MP.labels['keypoints_rel'])
    ref_pose = all_poses[ref_pose_idx]

    # Distance of reference to all other poses to sort them accordingly
    diff = all_poses - np.expand_dims(ref_pose, 0)
    dist2ref = np.mean(np.linalg.norm(diff, axis=-1), axis=-1)
    # Get n_pose equally spaced poses
    sorted_indices = np.argsort(dist2ref)
    choice_of_choice = np.round(np.linspace(0, len(MP)-1, n_pose)).astype(int)
    choice_indices = sorted_indices[choice_of_choice]

    examples = MP[choice_indices]
    examples_indices = choice_indices
    example_distances = dist2ref[choice_indices]
    var_on_examples = []
    var_indices = []

    for idx, ex, dist in zip(choice_indices, examples, example_distances):
        kps = ex['keypoints_rel']
        pid = ex['pid']

        # Distances between current pose and all others
        diff = all_poses - np.expand_dims(kps, 0)
        dist2ref = np.linalg.norm(diff, axis=-1)
        dist2ref = np.mean(dist2ref, axis=-1)

        ax1.hist(dist2ref, alpha=0.5, label='$\Delta={:0.2}$'.format(dist))

    ax1.set_title('Distances to {} reference poses'.format(n_pose))
    ax1.set_ylabel('Count')
    ax1.set_xlabel('Distances')
    ax1.legend()

    f.savefig('dset_statistics.png')


def mean_plot(MP, n_pose=10, n_app=250, sp=None):
    np.random.seed(42)

    ref_pose_idx = np.random.randint(len(MP))

    all_pids = np.array(MP.labels['pid'])

    all_poses = np.array(MP.labels['keypoints_rel'])
    ref_pose = all_poses[ref_pose_idx]
    print(ref_pose.shape)
    print(all_poses.shape)

    # Distance of reference to all other poses to sort them accordingly
    diff = all_poses - np.expand_dims(ref_pose, 0)
    print(diff.shape)
    dist2ref = np.mean(np.linalg.norm(diff, axis=-1), axis=-1)
    print(dist2ref.shape)

    # Get n_pose equally spaced poses
    sorted_indices = np.argsort(dist2ref)
    choice_of_choice = np.round(np.linspace(0, len(MP)-1, n_pose)).astype(int)
    choice_indices = sorted_indices[choice_of_choice]
    print(choice_indices.shape)

    examples = MP[choice_indices]
    examples_indices = choice_indices
    example_distances = dist2ref[choice_indices]
    var_on_examples = []
    var_indices = []

    for idx, ex in zip(choice_indices, tqdm(examples)):
        kps = ex['keypoints_rel']
        pid = ex['pid']

        mean_im = None

        # Distances between current pose and all others
        diff = all_poses - np.expand_dims(kps, 0)
        dist2ref = np.linalg.norm(diff, axis=-1)
        dist2ref = np.mean(dist2ref, axis=-1)
        dist2ref = np.argsort(dist2ref)
        pids_ = all_pids[dist2ref]

        for i in trange(n_app):
            if mean_im is None:
                mean_im = np.array(MP[dist2ref[i]]['target'])
            else:
                mean_im += np.array(MP[dist2ref[i]]['target'])

        mean_im /= n_app
        var_ex = mean_im
        var_idx = list(range(n_app))

        var_on_examples += [var_ex]
        var_indices += [var_idx]

    f, AX = plt.subplots(1+2, n_pose)

    for r_idx, Ax in enumerate(AX):
        for c_idx, ax in enumerate(Ax):
            if r_idx == 0:
                kps = examples[c_idx]['keypoints_rel'] * 256
                # ax.scatter(kps[..., 0], kps[..., 1], marker='.')
                heat = np.mean(kp2heat(kps, min_std=10), axis=-1)
                heat /= heat.max()
                print(heat.shape, heat.min(), heat.max())
                ax.imshow(heat)
                ax.axis('off')
                ax.set_aspect(1)
                # ax.set_ylim(1, 0)
                ax.set_title('$\Delta={:0.3}$'.format(example_distances[c_idx]), fontsize=6)
            elif r_idx == 1:
                crop_path = examples[c_idx]['crop_path']
                ax.imshow(plt.imread(crop_path))
                ax.axis('off')
            else:
                mean_im = var_on_examples[c_idx]
                mean_im = adjust_support(mean_im, '0->1')
                print(mean_im.shape, mean_im.min(), mean_im.max())
                ax.imshow(mean_im)
                ax.axis('off')

    if sp is None:
        sp = 'mean_plot.png'
    f.savefig(sp, dpi=300)


def pose_statistics(keypoints, velocities, angles,
                    mean_pose=None, connections=ALPHAPOSE_COCO_TUPLES,
                    sp='pose_statistics.png'):
    """Plot info about the pose.

    Args:
        velocities (np.ndarray): ``[N, K, (2)]``.
        angles (list(np.ndarray, np.ndarray)): Triples + angles
    """

    triples, angles = angles

    f, ax = plt.subplots(1, 1)

    if mean_pose is None:
        mean_pose = np.mean(keypoints, axis=0)

    # ax.scatter(mean_pose[:, 0], mean_pose[:, 1])
    ax.set_aspect(1)
    ax.invert_yaxis()
    ax.axis('off')

    for a, b in connections:
        X = [mean_pose[a, 0], mean_pose[b, 0]]
        Y = [mean_pose[a, 1], mean_pose[b, 1]]
        ax.plot(X, Y)

    ang = kp2ang(mean_pose, triples)
    center_points = np.array([mean_pose[t[1]][:2] for t in triples])

    angles = np.arccos(angles) / np.pi * 360

    patches = []
    for i, [c, a_deg] in enumerate(zip(center_points, ang)):
        ax.scatter(c[0], c[1])
        
        ma_deg = np.mean(angles[..., i], axis=0)
        va_deg = np.std(angles[..., i], axis=0)

        t1, t2 = ma_deg - 2, ma_deg + 2
        w = mpatches.Wedge(c, 0.04, t1, t2, ec=None)
        patches += [w]

        t1, t2 = ma_deg - va_deg / 2., ma_deg + va_deg / 2.
        w = mpatches.Wedge(c, 0.04, t1, t2, ec=None)
        patches += [w]

    colors = np.linspace(0, 1, len(patches) // 2)
    colors = np.repeat(colors, 2)
    collection = PatchCollection(patches, cmap=plt.cm.hsv, alpha=0.3, clip_on=False)
    collection.set_array(np.array(colors))
    ax.add_collection(collection)

    f.savefig(sp, dpi=300)
    

if __name__ == '__main__':
    from multiperson_dataset import MultiPersonDataset

    # MP1 = MultiPersonDataset('/export/scratch/jhaux/Data/olympic_sports_new/basketball_layup/-9t7-hXcl4U_01168_01301.seq_track/')
    # MP2 = MultiPersonDataset('/export/scratch/jhaux/Data/olympic_sports_new/tennis_serve/2xQquQVOjXA_00588_00725.seq_track/')
    # MP3 = MultiPersonDataset('/export/scratch/jhaux/Data/olympic_sports_new/basketball_layup/ZQkt3S4WY5Y_01834_01986.seq_track')

    # MP = MP1 + MP2 + MP3
    # n = len(set(MP.labels['video_path']))

    # # plot_seqs(MP, 1)  # n)

    np.random.seed(42)

    # plot_app_variance_at_similiar_pose(MP)


    def listfiles(folder):
        ret_list = []
        for root, folders, files in os.walk(folder):
            for filename in folders + files:
                ret_list += [os.path.join(root, filename)]

        return  ret_list

    import re

    # https://regex101.com/r/C2NQu4/1
    regex = re.compile('.+seq$')

    root ='/export/scratch/jhaux/Data/olympic_sports_new/'
    # root ='/export/scratch/jhaux/Data/olympic_test/'
    videos = [v for v in listfiles(root) if regex.match(v) is not None]
    videos = np.random.choice(videos, size=min(len(videos), 100))

    MP = None
    for v in videos:
        v += '_track'
        try:
            if MP is None:
                MP = MultiPersonDataset(v)
            else:
                MP += MultiPersonDataset(v)
        except Exception as e:
            print(e)

    plot_app_variance_at_similiar_pose(MP)
