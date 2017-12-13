from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect, re
import numpy as np
import os
import collections
from skimage import measure

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (image_numpy + 1) / 2.0 * 255.0
    if image_numpy.shape[0] == 1:
        image_numpy = image_numpy.repeat(3, 0)
    elif image_numpy.shape[0] == 2:
        image_numpy = np.concatenate((image_numpy, np.zeros([1, image_numpy.shape[1], image_numpy.shape[2]],
                                                            dtype=image_numpy.dtype)), axis=0)
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]) )


def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def compute_Rand_F_scores(S, T, do_thin=False):
    # input s, t
    if S.ndim == 2:
        S = S.reshape([1, 1] + S.shape)
        T = T.reshape([1, 1] + T.shape)
    numImages = T.shape[0]
    scores = np.zeros(numImages)

    for k in range(numImages):
        t = T[k].squeeze(axis=0)
        s = S[k].squeeze(axis=0)
        t = t > 0.5
        s = s > 0.5
        if do_thin:
            from skimage.morphology import thin
            s = thin(s)
        t_label = measure.label(t, background=1)
        s_label = measure.label(s, background=1)
        t_max = t_label.max()
        s_max = s_label.max()

        # joint distribution
        p = np.zeros([t_max + 1, s_max + 1])
        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
                p[t_label[i, j], s_label[i, j]] += 1
        p_ = p[1:, :]
        n = p.sum()
        p_ /= n
        p__ = p_[:, 1:]
        pi0 = p_[:, 0]
        aux = pi0.sum()
        ai = np.sum(p_, axis=1)
        bj = np.sum(p__, axis=0)
        sumA2 = np.power(ai, 2).sum()
        sumB2 = np.power(bj, 2).sum() + aux / n
        sumAB2 = np.power(p__, 2).sum() + aux / n
        prec = sumAB2 / sumB2
        rec = sumAB2 / sumA2
        # F-score
        index = 2 / (1 / prec + 1 / rec)
        scores[k] = index
    return scores


def mul(in1, in2):
    # mult in1 with in2 (target)
    out = None
    size1 = in1.data.size()
    size2 = in2.data.size()
    if size1 == size2:
        out = torch.mul(in1, in2)
    elif size1 < size2:
        # pad in1
        pad_l = int((size2[3] - size1[3]) / 2)
        pad_b = int((size2[2] - size1[2]) / 2)
        pad_r = int(size2[3] - size1[3] - pad_l)
        pad_t = int(size2[2] - size1[2] - pad_b)
        out = torch.nn.ReflectionPad2d((pad_l, pad_r, pad_t, pad_b))(in1) * in2
    return out
