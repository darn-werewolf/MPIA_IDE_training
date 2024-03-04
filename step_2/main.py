import os
import pickle
import numpy as np

from astropy.io import fits
from img2vec_pytorch import Img2Vec
from PIL import Image
from tqdm import tqdm


def fits_open(filename):
    # Function to open FITS files asynchronously
    with fits.open(filename) as hdul:
        return hdul[0].data.copy()


def im_stacker(cutout_id):
    base_filename = "/data/beegfs/astro-storage/groups/banados/jwolf/euclid_ero/Perseus_out/fits_cutouts/cutout_{BAND}_{ID}.fits"

    # Asynchronously read FITS data
    Y = fits_open(base_filename.format(BAND="Y", ID=cutout_id))
    J = fits_open(base_filename.format(BAND="J", ID=cutout_id))
    H = fits_open(base_filename.format(BAND="H", ID=cutout_id))

    # Identify indices of dead pixels
    dead_pix = [np.where(np.isnan(Y)), np.where(np.isnan(J)), np.where(np.isnan(H))]

    # Fix dead pixels with the median of the image
    Y_median = np.nanmedian(Y)
    Y[np.isnan(Y)] = Y_median

    J_median = np.nanmedian(J)
    J[np.isnan(J)] = J_median

    H_median = np.nanmedian(H)
    H[np.isnan(H)] = H_median

    stack_out = np.stack((Y, J, H), axis=-1)

    return stack_out, dead_pix


def im_rescaler(rgb_image):
    rgb_image_normalized = (rgb_image - np.min(rgb_image)) / (
        np.max(rgb_image) - np.min(rgb_image)
    )
    rgb_image_rescaled = (rgb_image_normalized * 255).astype(np.uint8)
    return rgb_image_rescaled


def im_vectorizer(rgb_array):
    img2vec = Img2Vec()
    im = Image.fromarray(im_rescaler(rgb_array))
    vec = img2vec.get_vec(im)
    return im, vec


def process_image(im_idx):
    rgb_stack, curr_dead = im_stacker(im_idx)
    rgb_scaled = im_rescaler(rgb_stack)
    curr_im, curr_vec = im_vectorizer(rgb_scaled)
    curr_im.save(
        f"/data/beegfs/astro-storage/groups/jwst/fowler/data/images/{im_idx}.png"
    )
    np.savetxt(
        f"/data/beegfs/astro-storage/groups/jwst/fowler/data/text/{im_idx}.txt",
        curr_vec,
    )
    with open(
        f"/data/beegfs/astro-storage/groups/jwst/fowler/data/text/{im_idx}_deadpix.pkl",
        "wb",
    ) as file:
        pickle.dump(curr_dead, file)


if __name__ == "__main__":
    all_files = os.listdir(
        "/data/beegfs/astro-storage/groups/banados/jwolf/euclid_ero/Perseus_out/fits_cutouts/"
    )
    file_names = [
        f
        for f in all_files
        if os.path.isfile(
            os.path.join(
                "/data/beegfs/astro-storage/groups/banados/jwolf/euclid_ero/Perseus_out/fits_cutouts/",
                f,
            )
        )
    ]
    clean_filenames = [x.replace(".fits", "") for x in file_names]
    clean_ids = set([x.split("_")[2] for x in clean_filenames])
    for xx in tqdm(clean_ids, desc="Processing images."):
        process_image(xx)
