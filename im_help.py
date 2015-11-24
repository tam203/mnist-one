import Image
import numpy as np

def image_from_2d(mat,path):
    assert mat.ndim == 2
    max = np.amax(mat)
    min = np.amin(mat)
    absmax = max if max >= -min else -min
    norm = 255/absmax
    print("max is %s so norm is %s "%(absmax, norm))
    im = Image.new("RGB", mat.shape, "white")
    im_data = im.load()
    for i in range(im.width):
        for j in range(im.height):
            r = int(mat[i,j] * norm) if mat[i,j] > 0 else 0
            b = -int(mat[i,j] * norm) if mat[i,j] < 0 else 0
            im_data[i,j] = (r, 0, b)
    im.save(path)


def reshape(mat):
    assert mat.ndim == 1
    assert mat.shape[0] == 784
    dup = np.copy(mat)
    dup.shape = (28, 28)
    return dup





