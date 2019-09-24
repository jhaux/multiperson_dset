try:
    import edflow
except ImportError as e:
    print('\033[91m{}\033[0m'
            .format('Please install edflow: https://github.com/pesser/edflow')
            )
    raise e

from edflow.custom_logging import get_logger
from edflow.data.dataset import DatasetMixin, ConcatenatedDataset, SubDataset
from edflow.data.util import adjust_support

from multiperson.multiperson_dataset import MultiPersonDataset

from PIL import Image
from tqdm import tqdm
import numpy as np
import os
import re
import yaml
from sklearn.model_selection import train_test_split


class AggregatedMultiPersonDataset(DatasetMixin):
    def __init__(self, config, root=None, ext=['mp4'], load_images=False,
                 force=False, debug=False, filter_fn=lambda vids: vids):
        '''Will return paths to crops, masks and keypoints of persons sorted by
        their respective frame index.
        Frames which are smaller than a third of config[`spatial_size`] will be
        dropped.

        Parameters
        ----------
        config : dict
            Containg a subset of the following keys:
            - spatial_size : int, tuple
                The size to which the person crops shall be resized.
            - mode : str
                Either train or eval, default train
            - data_split : list(str)
                Defining which parts of the data to use. Can be ``[train]``,
                ``[eval]``, ``[train, eval]`` or ``None``. If ``None`` mode
                is used (default).
            - random_seed : in
                Used for splitting the data into train and test split.
                Default: 42. Splitting is done on the list of videos, thus the
                person ids are disjunct between train and test split, assuming
                no person is in two videos.
            - test_size : float
                Defining, how the dataset is split into train and test set.
                Default 0.2.
        root : str
            Path to the directory containing the video data. If not specified
            (default), use the environment variable ``MULTIPERSON_ROOT``.
        ext : str or list(str)
            File extension of the videos. Must be supplied without the dot
            "``.``", e.g. ``mp4`` (default).
        load_image : bool
            Wether or not to load the images behind the keys `crop_path` and
            `mask_path`. If ``True`` adds the keys crop_im, mask, target.
        force : bool
            Wether or not to look for new videos. If ``True`` the dataset will
            scan root for all files with the extension :attr:`ext` and write
            the into a file called ``ready_videos.yaml``. Otherwise it will
            load this file and only use the videos specified in there.
            .. warning:: ``ready_videos.yaml`` will be overwritten on
                ``force=True``. Do not use it as to filter your videos without
                keeping this in mind.
        debug : bool
            Only use the first 2 found videos.
        filter_fn : Callable
            Function which takes the list of names of the detected videos and
            returns a possibly modified list of video names.
        '''

        self.config = config
        self.logger = get_logger(self)
        self.load_images = load_images

        size = self.config.setdefault('spatial_size', (256, 256))
        if isinstance(size, int):
            size = (size, size)
        elif not (isinstance(size, (tuple, list, np.ndarray))
                  and len(size) == 2):
            raise ValueError('Size must be either int or a tuple like of len 2, '
                             + 'but is of type {}, containing {}'
                             .format(type(size), size))
        self.crop_size = np.array(size)

        # Here are all the videos!
        if root is None:
            root = os.environ['MULTIPERSON_ROOT']

        self.logger.info('Lookin for videos at {}'.format(root))

        # To not always have to scan all directories store the usable videos in
        # list at root
        video_list = os.path.join(root, 'ready_videos.yaml')
        if not os.path.exists(video_list) or force:
            # Now let's find all the videos
            if not isinstance(ext, list) and isinstance(ext, str):
                ext = [ext]
            else:
                raise ValueError('ext must be either list or string, but is '
                                 + '{} > {}'.format(type(ext), ext))
            videos = find_videos(root, ext)
            # Videos are written out below
        else:
            with open(video_list, 'r') as vfile:
                videos = yaml.safe_load(vfile)

        if debug or config.setdefault('debug_mode', False) \
                or bool(os.environ.get('DEBUG', False)):
            videos = videos[:2]

        self.logger.debug('Videos:\n{}'
                         .format('\n'.join(videos)))

        # Now define which split of the data we want to use. There are two
        # splits, train and test. To use both specify in the config 
        # split: [train, eval]
        mode = self.config.setdefault('mode', 'train')
        split = self.config.setdefault('data_split', None)
        seed = self.config.setdefault('random_seed', 42)
        test_size = self.config.setdefault('test_size', 0.2)

        train_vids, test_vids = train_test_split(
                videos, test_size=test_size, random_state=seed, shuffle=False
                )

        splits = {'train': train_vids, 'eval': test_vids}

        if split is None:
            split = [mode]

        videos = []
        for s in split:
            videos += splits[s]

        videos = filter_fn(videos)

        split_s = ' and '.join(split)
        self.logger.info('Using {}-split: {} videos'
                         .format(split_s, len(videos)))

        # Given the videos, see if they have been processed yet and if so
        # concatenate those together.
        MPs = []
        misses = []
        hits = []
        dropped_frames = []
        dropped_vids = []
        lens = []
        total_frames = 0
        thresh = size[0] // 3
        self.logger.info('Loading video data...')
        min_seq_len = self.config.setdefault('min_seq_len', 200)
        for v in tqdm(videos, desc='video data'):
            v_t = v + '_track'
            try:
                MP = MultiPersonDataset(v_t)
                total_frames += len(MP)

                large_boxes = MP.labels['bbox'][:, 2] >= thresh
                large_boxes = np.argwhere(large_boxes)
                large_boxes = np.squeeze(large_boxes, 1)

                if large_boxes.shape[0] == 0:
                    # Do not add this dataset at all
                    dropped_vids += [v]
                    dropped_frames += [len(MP)]
                    continue

                SubMP = SubDataset(MP, large_boxes)

                seq_ids = np.array(SubMP.labels['sequence_idx'])
                long_seqs, seq_lens = _filter_seq_len(seq_ids,
                                                      min_seq_len,
                                                      True)
                long_seqs = np.argwhere(long_seqs)
                long_seqs = np.squeeze(long_seqs)

                lens += list(seq_lens)
                self.logger.debug('ls {}'.format(long_seqs))
                if len(long_seqs) == 0:
                    dropped_vids += [v]
                    dropped_frames += [len(MP)]
                    continue

                SubMP = SubDataset(SubMP, long_seqs)

                dropped_frames += [len(MP) - len(SubMP)]

                MPs += [SubMP]
                hits += [v]
            except FileNotFoundError as e:
                self.logger.debug(str(e))
                misses += [v]

        self.logger.info('Dropped {n} frames < {s}x{s} of {t} frames in total.'
                         .format(n=sum(dropped_frames), s=thresh, t=total_frames))
        self.logger.info('{}+-{} frames per video'
                         .format(np.mean(dropped_frames),
                                 np.std(dropped_frames)))
        n_dropped_vids = len(videos) - len(MPs)
        self.logger.info('Dropped {} videos entirely'
                         .format(len(dropped_vids)))
        self.logger.debug('Dropped Videos:\n{}'
                          .format('\n'.join(dropped_vids)))

        self.logger.info('Average sequense length: {}+-{} (before filtering)'
                         .format(np.mean(lens), np.std(lens)))
        lens = np.array(lens)
        self.logger.info('Average sequense length: {}+-{} (after filtering)'
                         .format(np.mean(lens[lens > min_seq_len]),
                                 np.std(lens[lens > min_seq_len])))

        fullMP = ConcatenatedDataset(*MPs)

        if not os.path.exists(video_list) or force:
            with open(video_list, 'w+') as vfile:
                yaml.dump(hits, vfile)

        self.MP = fullMP

        self.logger.info('Updating Labels...')
        # to be compatible with the Sequence dataset and make_abc_dataset, we
        # need to add a `fid` key and a `pid` key which is unique over all MPs
        self.MP.labels['fid'] = self.MP.labels['sequence_idx']

        vids = self.MP.labels['video_path']

        pid_path = os.path.join(root, 'pids.npy')
        if not os.path.exists(pid_path) or force:
            pids = np.array(self.MP.labels['person_id']).astype(str)
            pid_labels = np.stack([vids, pids], axis=-1)
            unique_pids = np.char.join('', list(pid_labels) + [''])[:-1]

            np.save(pid_path, unique_pids)
            self.logger.info('saved pid labels to {}'.format(pid_path))
        else:
            unique_pids = np.load(pid_path)
            self.logger.info('loaded pid labels from {}'.format(pid_path))
            assert len(unique_pids) == len(self.MP)
        self.MP.labels['pid'] = unique_pids

        video_fid_path = os.path.join(root, 'video_fids.npy')
        if not os.path.exists(video_fid_path) or force:
            video_fids = np.array(self.MP.labels['frame_idx']).astype(str)
            video_fid_labels = np.stack([vids, video_fids], axis=-1)
            unique_video_fids = np.char.join('', list(video_fid_labels) + [''])[:-1]

            np.save(video_fid_path, unique_video_fids)
            self.logger.info('saved video_fid labels to {}'.format(video_fid_path))
        else:
            unique_video_fids = np.load(video_fid_path)
            self.logger.info('loaded video_fid labels from {}'.format(video_fid_path))
            assert len(unique_video_fids) == len(self.MP)

        self.MP.labels['video_fid'] = unique_video_fids

        kps = np.copy(self.MP.labels['keypoints_rel'])
        kps[..., :2] = kps[..., :2] * self.crop_size
        self.MP.labels['keypoints'] = kps

        self.labels = self.MP.labels
        self.logger.info('Created Labels')

        self.logger.info('Constructed MultiPersonFrames containing '
                         + '{} videos and a total of {} frames'
                         .format(len(hits), len(self.MP)))
        self.logger.debug('Missed the following videos:\n{}'
                          .format('\n'.join(misses)))

    def get_example(self, idx):
        example = self.MP[idx]

        example['fid'] = self.labels['fid'][idx]
        example['pid'] = self.labels['pid'][idx]
        example['vid'] = example['video_path']

        example['box'] = example['bbox']
        example['orig_path'] = example['frame_path']

        example['keypoints'] = self.labels['keypoints'][idx]

        if self.load_images:
            crop_im = Image.open(example['crop_path'])

            crop_im = crop_im.resize(self.crop_size)
            crop_im = np.array(crop_im)
            crop_im = adjust_support(crop_im, '-1->1', '0->255')
            example['crop_im'] = crop_im

            m_path = example['mask_path']
            if os.path.exists(m_path):
                mask_im = Image.open(m_path)
                mask_im = mask_im.resize(self.crop_size)
                mask_im = np.array(mask_im)
                mask_im = adjust_support(mask_im, '-1->1')
            else:
                mask_im = np.ones(crop_im.shape[:2])

            if len(mask_im.shape) == 2:
                mask_im = np.expand_dims(mask_im, -1)

            example['mask'] = mask_im

            example['target'] = np.concatenate([crop_im, mask_im], axis=-1)
            example['flow'] = None

        return example

    def __len__(self):
        return len(self.MP)


def test_ending(string, tests=[], mode='or'):
    if mode == 'or':
        for test in tests:
            if string[-len(test):] == test:
                return True
        return False

    elif mode == 'and':
        for test in tests:
            if string[-len(test):] != test:
                return False
        return True

    else:
        raise ValueError('Unrecognized mode. Must be one of `or`, '
                         + '`and` but is {}'.format(mode))


def find_videos(root, extensions=['mp4']):
    '''Returns all videos we might be interested in, ignoring known subfolders
    created by the data aquisition pipeline.
    '''

    regexs = ['.+{}$'.format(e) for e in extensions]
    # https://regex101.com/r/h0my31/1
    regex = re.compile('|'.join(regexs))

    videos = [v for v in listfiles(root) if regex.match(v) is not None]
    videos = [v for v in videos if '/vis/' not in v]

    return sorted(videos)


def listfiles(folder):
    ret_list = []
    for root, folders, files in os.walk(folder):
        new_folders = []
        for f in folders:
            if not test_ending(f, ['_frames', '_masks', '_crops', '_track']):
                new_folders += [f]
        folders[:] = new_folders
        for filename in folders + files:
            ret_list += [os.path.join(root, filename)]

    return  ret_list


def filter_seq_len(dset, seq_key='sequence_id', min_len=4):
    labels = dset.labels[seq_key]

    index_mask = _filter_seq_len(labels, min_len)
    sub_indices = np.arange(len(labels))[index_mask]

    return SubDataset(dset, sub_indices)

def _filter_seq_len(labels, min_len=4, debug=False):
    diff = labels[:-1] - labels[1:]
    diff = np.concatenate([[-1], diff])
    stops = np.where(diff != -1)[0]
    stops = np.concatenate([stops, [len(labels)]])

    starts = np.clip(stops - 1, 0, len(labels) - 1)
    starts = np.concatenate([[0], starts])

    seq_lens = labels[stops - 1]
    mask_vals = seq_lens >= min_len

    mask = np.zeros([len(labels)], dtype=bool)
    last_start = starts[0]
    for start, stop, val in zip(starts, stops, mask_vals):
        mask[last_start:stop+1] = val
        last_start = stop

    if debug:
        return mask, seq_lens
    return mask


if __name__ == '__main__':
    from edflow.util import pp2mkdtable
    from edflow.data.util import plot_datum

    debug = False

    D = AggregatedMultiPersonDataset({'spatial_size': 256},
                                     root='/export/scratch/jhaux/Data/trickyoga',
                                     ext='mp4',
                                     force=True,  # See if new videos are ready!
                                     debug=debug)
    D2 = AggregatedMultiPersonDataset({'spatial_size': 256},
                                      root='/export/scratch/jhaux/Data/olympic_sports_new',
                                      ext='seq',
                                      force=False,  # See if new videos are ready!
                                      debug=debug)

    d = D[10]
    tab = pp2mkdtable(d)
    plot_datum(d, 'ty_10.png')
    print(tab)

    d = D2[10]
    tab = pp2mkdtable(d)
    plot_datum(d, 'oly_10.png')
    print(tab)

    print("D1: {}\nD2: {}".format(len(D), len(D2)))

    with open('multiperson.md', 'w+') as df:
        df.write(tab)
