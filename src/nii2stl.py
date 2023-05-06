from pathlib import Path
import nibabel as nib
import numpy as np
from stl import mesh
from skimage import measure


def load_nifti(nifti_file_path):
    img = nib.load(nifti_file_path)
    return img.get_fdata()


def convert_nii2stl(img):
    verts, faces, normals, values = measure.marching_cubes(img, 0)
    obj_3d = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        obj_3d.vectors[i] = verts[f]
    return obj_3d


def save_stl(img, nifti_file_path):
    img.save(nifti_file_path)


if __name__ == '__main__':
    # data path
    dir_path = Path('../data/')
    image_path = dir_path / 'raw/RibFrac420-image.nii.gz'
    label_path = dir_path / 'raw/RibFrac420-label.nii.gz'
    save_path = dir_path / 'interium/RibFrac420-image_3d.stl'

    # format conversion
    img = load_nifti(image_path)
    obj = convert_nii2stl(img)
    save_stl(obj, save_path)
