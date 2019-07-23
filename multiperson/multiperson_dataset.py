import ast
import json
import re
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

try:
    from edflow.data.dataset import CsvDataset
except ImportError as e:
    print('\033[91m{}\033[0m'
            .format('Please install edflow: https://github.com/pesser/edflow')
            )
    raise e


class MultiPersonDataset(CsvDataset):
    '''Expects the following Datastructure:
    .. codeblock::
        data_root/
            |- nested folders/
            |   |- VIDEO1
            |   |- VIDEO1_frames/
            |   |    |- frame_0001.png
            |   |    |- frame_0002.png
            |   |    |- frame_0003.png
            |   |    ...
            |   |- VIDEO1_masks/
            |   |    |- mask_0001.png
            |   |    |- mask_0002.png
            |   |    |- mask_0003.png
            |   |    ...
            |   |- VIDEO1_crops/
            |   |    |- crop_0001.png
            |   |    |- crop_0002.png
            |   |    |- crop_0003.png
            |   |    ...
            |   |- VIDEO1_track/
            |   |    |- alphapose-forvis-tracked.json
            |   |    ...
            |   |- VIDEO2
            |   |- VIDEO2_frames/
            |   |- VIDEO2_masks/
            |   |- VIDEO2_crops/
            |   |- VIDEO2_track/
            |   ...
            ...

    ``VIDEO1`` etc, are the videos, from which frames ``frame_XX.png`` are
    extracted. The file ``alphapose-forvis-tracked.json`` should following the
    datastructure from AlphaPose's PoseFlow tracker
    (``https://github.com/MVIG-SJTU/AlphaPose/tree/pytorch``). It contains a
    ``dict`` with keys for each frame at which a list of dicts resides, one for
    each person in that frame. Each person in each frame is describe by its
    ``id`` and ``keypoints``.

    The Dataset will take all inputs and sort them by person id, frame id and
    video name, s.t. at each index of the Dataset one will get the keypoints
    of one person with information about the frame id and video name.

    Example output:
    +-----------------+-----------+----------------------------------------------------------------------------------------------------------+
    |            Name |      Type |                                                                                                  Content |
    +=================+===========+==========================================================================================================+
    |      frame_path |       str |   /export/scratch/jhaux/Data/olympic sports/pole_vault/IWGyzPNXI_U_08144_08435.seq_frames/frame_0005.png |
    +-----------------+-----------+----------------------------------------------------------------------------------------------------------+
    |      video_path |       str |                         /export/scratch/jhaux/Data/olympic sports/pole_vault/IWGyzPNXI_U_08144_08435.seq |
    +-----------------+-----------+----------------------------------------------------------------------------------------------------------+
    |       frame_idx |     int64 |                                                                                                        4 |
    +-----------------+-----------+----------------------------------------------------------------------------------------------------------+
    |    sequence_idx |     int64 |                                                                                                        0 |
    +-----------------+-----------+----------------------------------------------------------------------------------------------------------+
    |       person_id |     int64 |                                                                                                        1 |
    +-----------------+-----------+----------------------------------------------------------------------------------------------------------+
    |   keypoints_abs |   ndarray |                                                                                                  (17, 3) |
    +-----------------+-----------+----------------------------------------------------------------------------------------------------------+
    |            bbox |   ndarray |                                                                                                     (4,) |
    +-----------------+-----------+----------------------------------------------------------------------------------------------------------+
    |   keypoints_rel |   ndarray |                                                                                                  (17, 3) |
    +-----------------+-----------+----------------------------------------------------------------------------------------------------------+
    |          index_ |       int |                                                                                                       10 |
    +-----------------+-----------+----------------------------------------------------------------------------------------------------------+
    '''

    def __init__(self, data_root, force=False, **pandas_kwargs):
        '''
        As crawling the data can be time consuming, the dataset will store its
        contents as csv file, and then reuse it at the next construction.
        Use the :attr:`force` argument to rewrite the file.
        
        Args:
            data_root (str): Path to the data.
            force (bool): Parse the folder structure again and write out a csv
                file.
            pandas_kwargs (kwargs): Keyword Arguments passed directly to pandas
                when constructing the underlying :class:`CsvDataset`.
        '''

        csv_name = os.path.join(data_root, 'per_person_content.csv')
        labels_name = os.path.join(data_root, 'per_person_labels.npz')

        super().__init__(csv_name, 
                         sep=';',
                         memory_map=True)

        self.labels.update(np.load(labels_name))

    def get_example(self, idx):
        example = {k: self.labels[k][idx] for k in self.labels}
        pid = example['person_id']
        sid = example['sequence_idx']
        fid = example['frame_idx']
        vid = example['video_path']

        crop_path_ = '{:0>5}-p{:0>3}-s{:0>3}-f{:0>3}.png'.format(
                idx,
                pid,
                sid,
                fid
                )

        crop_path = os.path.join(vid + '_crops', crop_path_)
        example['crop_path'] = crop_path

        mask_path = os.path.join(vid + '_masks', crop_path_)
        example['mask_path'] = mask_path

        return example


def csv2np(csv_root):

    arr_converter = Str2Np()

    data = pd.read_csv(csv_root,
                       sep=';',
                       converters={
                           'bbox': arr_converter,
                           'keypoints_abs':  arr_converter,
                           'keypoints_rel':  arr_converter
                           },
                       memory_map=True
                       )

    lpath = os.path.join(os.path.dirname(csv_root), 'per_person_labels.npz')
    invalid = ['frame_path', 'video_path']
    labels = {k: np.stack(data[k].values) for k in data if k not in invalid}
    np.savez(lpath, **labels)

    dropkeys = [k for k in data if k not in invalid]
    print(dropkeys)

    os.system('cp {} {}.old'.format(csv_root, csv_root))
    dropped_name = os.path.join(os.path.dirname(csv_root), 'per_person_content.csv')
    dropped = data.drop(labels=dropkeys, axis=1)
    dropped.to_csv(dropped_name, sep=';', index=False)

    for k, v in labels.items():
        print(k, v.shape)


class Str2Np(object):
    '''To understand what's going on here: https://regex101.com/r/sW3qN8/17
    The string representation of numpy arrays does not contain commas,
    which are needed to convert them to list using ast.literal eval.
    '''
    def __call__(self, arr_as_str):
        if not hasattr(self, 'regex'):
            self.regex = re.compile(r'(?<=[-\d\].])\s+(?=[-\d\[])')

        try:
            better_string = self.regex.sub(', ', arr_as_str)
            arr = ast.literal_eval(better_string)
            arr = np.array(arr)
        except Exception as e:
            print(arr_as_str)
            print(better_string)
            raise e
        return arr


def extract_lines(tracking_data):
    ''' Converts dict of list of persons to list of persons with frame
    annotation.
    
    Args:
        tracking_data (dict): ``frame: [{idx: 1, ...}, {...}, ...]``
    '''

    linear = []
    for i, k in enumerate(sorted(tracking_data.keys())):
        for data in tracking_data[k]:
            example = {'orig': k, 'fid': i}
            example.update(data)
            linear += [example]

    sorted_linear = sorted(linear, key=lambda e: [e['idx'], e['fid']])

    last_id_change = 0
    last_id = None
    last_fid = -1
    for example in sorted_linear:
        ex_id = example['idx']
        if last_id != ex_id or last_fid != example['fid'] - 1:
            last_id_change = example['fid']

        seq_idx = example['fid'] - last_id_change
        example['sid'] = seq_idx
        last_id = ex_id
        last_fid = example['fid']

    return sorted_linear


def prepare_keypoints(kps_raw):
    '''Converts kps of form ``[x, y, c, x, y, c, ...]`` to 
    ``[[x, y, c], [x, y, c], ...]``'''

    x = kps_raw[::3]
    y = kps_raw[1::3]
    c = kps_raw[2::3]

    return np.stack([x, y, c], axis=-1)

def square_bbox(prepared_kps, pad=0.35, kind='percent'):
    if not kind in ['percent', 'abs']:
        raise ValueError('`kind` must be one of [`percent`, `abs`], but is {}'
                         .format(kind))

    x = prepared_kps[:, 0]
    y = prepared_kps[:, 1]

    minx, maxx = x.min(), x.max()
    miny, maxy = y.min(), y.max()

    wx = maxx - minx
    wy = maxy - miny
    w = max(wx, wy)

    centerx = minx + wx / 2.
    centery = miny + wy / 2.

    if pad is not None and pad != 0:
        if kind == 'percent':
            w = (1 + pad) * w
        else:
            w += pad

    bbox = np.array([centerx - w/2., centery - w/2., w, w])

    return bbox


def get_kps_rel(kps_abs, bbox):
    kps_rel = np.copy(kps_abs)
    kps_rel[:, :2] = kps_rel[:, :2] - bbox[:2]

    kps_rel[:, :2] = kps_rel[:, :2] / bbox[2:]

    return kps_rel


def add_lines_to_csv(data_frame, track_dir, frame_dir, root, kp_in_csv=True):
    json_name = os.path.join(root,
                             track_dir,
                             'alphapose-forvis-tracked.json')

    with open(json_name, 'r') as json_file:
        tracking_data = json.load(json_file)

    all_kps_abs = []
    all_kps_rel = []
    all_boxes = []

    raw_lines = extract_lines(tracking_data)
    for j, line in enumerate(tqdm(raw_lines, 'L')):
        kps_abs = prepare_keypoints(line['keypoints'])
        bbox = square_bbox(kps_abs)
        kps_rel = get_kps_rel(kps_abs, bbox)

        frame_root = os.path.join(root, frame_dir, line['orig'])

        vid = os.path.join(root, frame_dir[:-7])
        pid = line['idx']
        fid = line['fid']
        sid = line['sid']

        if kp_in_csv:
            data_frame = data_frame.append(
                    {
                        'frame_path': frame_root,
                        'video_path': vid,
                        'frame_idx': fid,
                        'sequence_idx': sid,
                        'person_id': pid,
                        'keypoints_abs': kps_abs,
                        'bbox': bbox,
                        'keypoints_rel': kps_rel
                        },
                    ignore_index=True  # append with incremental index
                    )
        else:
            all_kps_abs += [kps_abs]
            all_kps_rel += [kps_rel]
            all_boxes += [bbox]

            data_frame = data_frame.append(
                    {
                        'frame_path': frame_root,
                        'video_path': vid,
                        'frame_idx': fid,
                        'sequence_idx': sid,
                        'person_id': pid,
                        },
                    ignore_index=True  # append with incremental index
                    )

    if not kp_in_csv:
        return data_frame, np.stack(all_kps_abs), np.stack(all_kps_rel), np.stack(all_boxes)
    else:
        return data_frame


def parse_data(data_root, csv_name):
    print('Creating csv')
    data_frame = pd.DataFrame(columns=
            [
                'frame_path',
                'video_path',
                'frame_idx',
                'sequence_idx',
                'person_id',
                'keypoints_abs',
                'bbox',
                'keypoints_rel'
                ]
            )

    for i, [root, dirs, files] in enumerate(tqdm(os.walk(data_root), desc='W')):
        track_dirs = [d for d in dirs if d[-6:] == '_track']
        frame_dirs = [d for d in dirs if d[-7:] == '_frames']

        if len(track_dirs) != 0 and len(track_dirs) == len(frame_dirs):
            # This folder contains the relevant video data.

            for td, fd in zip(tqdm(track_dirs, desc='V'), frame_dirs):
                data_frame = add_lines_to_csv(data_frame, tf, fd, root)

    data_frame.to_csv(csv_name, sep=';', index=False)
    print(data_frame)
    print(data_frame['keypoints_rel'][0])
    print(data_frame['keypoints_abs'][0])
    print(data_frame['bbox'][0])


if __name__ == '__main__':
    # l_re = re.compile(r'(?<=[\d, \]])\s+(?=[\d, \[])')

    # a = np.array([[[1.1, 2.2, 3.3], [4, 5, 6]]]*3)

    # aa = '"'+str(a)+'"'

    # print('aa', aa)

    # asub = l_re.sub(', ', aa)
    # print('as', asub)

    # lit = ast.literal_eval(asub)
    # print('l', lit)

    # nlit = np.array(lit)
    # print('n', nlit)

    # csv2np('/export/scratch/jhaux/Data/olympic sports/per_person_content_with_kps.csv')

    # exit()

    # MP = MultiPersonDataset('/export/scratch/jhaux/Data/olympic sports/', True)
    # MP = MultiPersonDataset('/export/scratch/jhaux/Data/olympic sports/')
    MP = MultiPersonDataset('/export/scratch/jhaux/Data/olympic_test/', True)

    from edflow.util import pprint, pp2mkdtable

    d = MP[10]
    print(pp2mkdtable(d))

    print(len(MP))
    print(list(MP.labels.keys()))
    print(MP.labels['person_id'].shape)
    print(MP.labels['bbox'].shape)
    print(MP.labels['keypoints_rel'].shape)

    for i in range(min(1000, len(MP))):
        d = MP[i]
        print(d['sequence_idx'], d['frame_idx'], os.path.basename(d['video_path']))
