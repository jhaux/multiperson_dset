'''Crop out all persons using the bounding box information.'''


from PIL import Image
import numpy as np

from edflow.data.util import get_support, adjust_support
from edflow.data.dataset import DatasetMixin


def crop(image, box):
    '''Arguments:
        image (np.ndarray or PIL.Image): Image to crop.
        box (list): Box specifying ``[x, y, width, height]``
        points (np.ndarray): Optional set of points in image coordinate, which
        are translated to box coordinates. Shape: ``[(*), 2]``.

    Returns:
        np.ndarray: Cropped image with shape ``[W, H, C]`` and same support
            as :attr:`image`.

        If points is not None:
            np.ndarray: The translated point coordinates.
    '''

    is_image = True
    if not isinstance(image, Image.Image):
        in_support = get_support(image)
        image = adjust_support(image, '0->255')
        image = Image.fromarray(image)
        is_image = False

    box[2:] = box[:2] + box[2:]

    image = image.crop(box)

    if not is_image:
        image = adjust_support(np.array(image), in_support)

    return image


def resize(image, size, points=None, prev_size=None):
    '''Arguments:
        image (np.ndarray or PIL.Image): Image to crop.
        size (int or list): Size of the image after resize.
        points (np.ndarray): Optional set of points in image coordinate, which
            are translated to box coordinates. Shape: ``[(*), 2]``.
        prev_size (int or list): Used to calculate the scaling for the points.
            If not given, this is estimated from the shape of the image.

    Returns:
        np.ndarray: Resized image with shape ``[W, H, C]`` and same support
            as :attr:`image`.

        If points is not None:
            np.ndarray: The translated point coordinates.
    '''

    in_support = get_support(image)
    if prev_size is None:
        prev_size = image.shape[:2]
    else:
        if isinstance(prev_size, int):
            prev_size = [prev_size] * 2
    prev_size = np.array(prev_size)

    if not isinstance(image, Image.Image):
        image = adjust_support(image, '0->255')
        image = Image.fromarray(image)

    if isinstance(size, int):
        size = [size] * 2
    image = image.resize(size)
    size = np.array(size)

    image = adjust_support(np.array(image), in_support)

    if points is not None:
        points[..., :2] = points[..., :2] * size / prev_size

        return image, points

    return image


def pc2box(point_cloud):
    '''Finds the bounding box of a given pointcloud with shape ``[(*), 2]``.
    '''

    if not isinstance(point_cloud, np.ndarray):
        point_cloud = np.array(point_cloud)

    xs = point_cloud[..., 0]
    ys = point_cloud[..., 1]

    minx = xs.min()
    maxx = xs.max()
    miny = ys.min()
    maxy = ys.max()

    mins = np.array([minx, miny])
    maxs = np.array([maxx, maxy])

    widths = maxs - mins

    box = np.concatenate([mins, widths])

    return box


def make_quad_box(box, pad_pix=None, pad_percent=None):
    assert pad_pix is not None or pad_percent is not None

    max_len = max(box[2:])
    if pad_pix is not None:
        max_len += pad_pix
    else:
        max_len *= 1 + pad_percent

    centers = box[:2] + 0.5*box[2:]

    starts = centers - 0.5*max_len
    widths = np.array([max_len, max_len])

    return np.concatenate([starts, widths])


class CropDataset(DatasetMixin):
    '''Crops an image at a given key to a quadratic box estimated from a
    set of 2D points at a second given key.
    '''

    def __init__(self, base_dset, image_key='frame_path', box_key='bbox'):
        self.data = base_dset

        self.imkey = image_key
        self.bkey = box_key

        self.resize = resize

    def __len__(self):
        return len(self.data)

    def get_example(self, idx):
        ex = self.data[idx]

        image = np.array(Image.open(ex[self.imkey]))
        box = ex[self.bkey]

        im_crop = crop(image, box)

        ex['im_crop'] = im_crop
        ex['image'] = image

        return ex


def plot_kps(datum, idx=-1):
    import matplotlib.pyplot as plt
    im = datum['image']
    kps = datum['keypoints']

    f, ax = plt.subplots(1, 1)

    ax.imshow(adjust_support(im, '0->1', clip=True))
    ax.scatter(kps[:, 0], kps[:, 1])

    f.savefig('kps-{}.png'.format(idx))


def test_new_crop(d, idx):
    from multiperson_dataset import square_bbox, get_kps_rel

    kps = d['keypoints_abs']

    box = square_bbox(kps)
    kps_rel = get_kps_rel(kps, box)


    image = np.array(Image.open(d['frame_path']))
    im_crop = crop(image, box)

    im_crop = adjust_support(im_crop, '0->255')
    im_crop = Image.fromarray(im_crop)
    im_crop.save('croptest_{}.png'.format(idx), 'PNG')


    d['keypoints'] = kps
    d['im_crop'] = im_crop

    plot_kps(d, idx)


if __name__ == '__main__':
    from multiperson_dataset import MultiPersonDataset

    # MP = MultiPersonDataset('/export/scratch/jhaux/Data/olympic sports/')
    MP = MultiPersonDataset('/export/scratch/jhaux/Data/olympic_test/')
    CMP = CropDataset(MP)

    from edflow.util import pprint, pp2mkdtable
    from edflow.data.util import plot_datum

    idx = 10

    for idx in range(25):
        d = CMP[idx]
        print(pp2mkdtable(d))
        plot_datum(d, 'crop_{}.png'.format(idx))

        test_new_crop(d, idx)

    for idx in np.random.randint(len(CMP), size=10):
        d = CMP[idx]
        print(pp2mkdtable(d))
        plot_datum(d, 'crop_{}.png'.format(idx))

        test_new_crop(d, idx)
