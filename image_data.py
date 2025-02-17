# This script carries out the imaging. It makes a dirty image, reads
# and aligns the user-supplied clean mask, carries out multiscale
# imaging down to S/N ~ 4, builds a mask of bright emission, carries
# out a single scale clean down to low S/N within this mask, and then
# exports the output to FITS.

# Right now it should work well for CO21 and C18O21. Imaging of the
# two-d products is still TBD.

# Edit the "Control Flow" section to use the script.

# WARNING! Right now a bug prevents the script from stopping and
# starting effectively, so it needs to be run end-to-end to work. This
# shouldn't be a big problem for the 7m data.

import os
import phangsPipeline as pp
import analysisUtils as au
import glob

casa_log_origin = "image_data"
casalog.showconsole(onconsole=False)

orig_log_path = casalog.logfile()
log_dir = os.path.dirname(orig_log_path)
orig_log_filename = os.path.basename(orig_log_path)
orig_log_date = orig_log_filename[5:-4]
staging_log_path = "{:}/{:}-image_data.log".format(log_dir, orig_log_date)
casalog.setlogfile(staging_log_path)
casalog.version()

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Control Flow
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# ... a text list. The script will process only these galaxies.
only = ["ngc_4038_4039"]

# ... skip these galaxies
skip = []

# ... start with this galaxy
first = ""
last = ""

# A list of strings specifing the array combination(s) you want to image.
# Acceptable options are:
#  - '7m'
#  - '12m_com'
#  - '12m_ext'
#  - '12m_com+7m'
#  - '12m_ext+12m_com'
#  - '12m_ext+12m_com+7m'
#  - empty list (all of the above)
just_array = [
]

# ... set as the products to be handled. Valid choices for the basic
# PHANGS data are 'co21', 'c18o21', 'cont', 'co21_chan0', and
# 'c18o21_chan0'. Note that right now cont and chan0 are not tested.
# Choices added by cdw are '13co21', 'co10', 'cn10high', and 'cn10low'.

just_product = ['co21']
#lines = ['co21', 'c18o21','13co21','co10','cn10high','cn10low']

# ... set these variables to indicate what steps of the script should
# be performed. The steps do:

# make_dirty_image - make a niter=0 image cube. Useful for checking
# mosaic parameters and that sort of thing. Used as a template for the
# clean mask, so this needs to be done first.

# revert_to_dirty - (DOES NOT WORK RIGHT NOW) reset the whole process
# so that the cube is now the dirty cube. Unfortunately there's some
# 'memory' related to the clean call in CASA that is causing a bug so
# that things don't cleanly resume from here.

# read_in_clean_mask - if a clean mask is found in the ../clean_masks/
# directory, read it in and align it to the astrometric grid of the
# dirty image. This is now the .mask file and will be used in future
# imaging.

# run_multiscale_clean - run a multiscale clean using a set of scales
# selected for that array combination. By default, clean down to a
# signal to noise of 4 in the residuals or stop when successive
# iterations of clean change the flux in the model by less than 2%.

# revert_to_multiscale - (DOES NOT WORK RIGHT NOW) reset the process
# to just after the multiscale clean. See revert_to_dirty above.

# make_singlescale_mask - construct a signal-to-noise based mask for
# use in single scale clean. Uses the .image to do this, so that
# bright regions in the image so far get more cleaning. This maks is
# joined with the clean mask, so no regions are included outside the
# clean mask.

# run_singlescale_clean - run a single scale hogbom clean inside the
# bright regions defined above. Clean with a very low (S/N ~ 1)
# threshold and stop based on convergence in flux between successive
# model images.

# export_to_fits - export the dirt, multiscale, and final images and
# associated products to FITS files for subsequent processing.
 
# Right now I recommend to not change the flow, and to rerun imaging
# from beginning to end when using the script.

make_dirty_image=True
forceSquare=False
revert_to_dirty=False
read_in_clean_mask=False
run_multiscale_clean=False
revert_to_multiscale=False
make_singlescale_mask=False
run_singlescale_clean=False
export_to_fits= False

do_only_new = False

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Loop
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

gals = pp.list_gal_names()

array_list = [
    '7m',
    '12m_com',
    '12m_ext',
    '12m_com+7m',
    '12m_ext+12m_com',
    '12m_ext+12m_com+7m',
]

product_list = [
    'co10',
    'cn10high',
    'cn10low',
    'co21',
    'c18o21',
    '13co21',
    'co10_chan0',
    'cn10high_chan0',
    'cn10low_chan0',
    'co21_chan0',
    'c18o21_chan0',
    '13co21_chan0',
    'cont',
]

before_first = True
after_last = False

for gal in gals:
    
    if len(only) > 0:
        if only.count(gal) == 0:
            casalog.origin(casa_log_origin)
            casalog.post("Skipping "+gal, "INFO")
            continue

    if len(skip) > 0:
        if skip.count(gal) > 0:
            casalog.origin(casa_log_origin)
            casalog.post("Skipping "+gal, "INFO")
            continue

    if first != "":
        if gal == first:
            before_first = False
        if before_first:
            continue
    
    if last != "":
        if after_last == True:
            continue
        if gal == last:
            after_last = True
 
    for array in array_list:

        if len(just_array) > 0:
            if array not in just_array:
                casalog.origin(casa_log_origin)
                casalog.post("Skipping "+array, "INFO")
                continue

        for product in product_list:

            if len(just_product) > 0:
                if just_product.count(product) == 0:
                    casalog.origin(casa_log_origin)
                    casalog.post("Skipping "+product, "INFO")
                    continue

            casalog.origin(casa_log_origin)
            casalog.post(gal + " " + array + " " + product, "INFO")

            this_dir = pp.dir_for_gal(gal)
            out_image_name = this_dir+gal+'_'+array+'_'+product+'.image'
            has_image = len(glob.glob(out_image_name)) > 0            

            if do_only_new:
                this_dir = pp.dir_for_gal(gal)
                out_image_name = this_dir+gal+'_'+array+'_'+product+'.image'
                has_image = len(glob.glob(out_image_name)) > 0
                if has_image:
                    casalog.origin(casa_log_origin)
                    casalog.post("", "WARN")
                    casalog.post("... You requested to only image new data.", "WARN")
                    casalog.post("... I found an existing image named "+out_image_name+" .", "WARN")
                    casalog.post("... I will skip this combination of galaxy, array, and product.", "WARN")
                    casalog.post("", "WARN")
                    continue

            clean_call = \
                pp.buildPhangsCleanCall(
                gal=gal,
                array=array,
                product=product,
                tag='',
                forceSquare=forceSquare)

            if clean_call == None:
                continue

            pp.phangsImagingRecipe(
                clean_call=clean_call,
                make_dirty_image=make_dirty_image,
                revert_to_dirty=revert_to_dirty,
                read_in_clean_mask=read_in_clean_mask,
                run_multiscale_clean=run_multiscale_clean,
                revert_to_multiscale=revert_to_multiscale,
                make_singlescale_mask=make_singlescale_mask,
                run_singlescale_clean=run_singlescale_clean,
                do_export_to_fits=export_to_fits,
                )

casalog.setlogfile(orig_log_path)
