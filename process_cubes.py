# This script post-processes the imaging into science-ready cubes.

import os
import phangsPipeline as pp
import phangsCubePipeline as pcp
import analysisUtils as au
import glob

casa_log_origin = "process_cubes"
casalog.showconsole(onconsole=False)

orig_log_path = casalog.logfile()
log_dir = os.path.dirname(orig_log_path)
orig_log_filename = os.path.basename(orig_log_path)
orig_log_date = orig_log_filename[5:-4]
staging_log_path = "{:}/{:}-process_cubes.log".format(log_dir, orig_log_date)
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

# A list of strings specifing the array combination(s) you want to process.
# Acceptable options are:
#  - '7m'
#  - '12m_com'
#  - '12m_ext'
#  - '12m_com+7m'
#  - '12m_ext+12m_com'
#  - '12m_ext+12m_com+7m'
#  - '7m+tp'
#  - '12m_com+7m+tp'
#  - '12m_ext+12m_com+7m+tp'
#  - empty list (all of the above)
just_array = [
    '7m',
    '12m_com',
    '12m_com+7m',
    '12m_com+7m+tp',
]

# ... set as the products to be handled. Valid choices for the basic
# PHANGS data are 'co21', 'c18o21', 'cont', 'co21_chan0', and
# 'c18o21_chan0'. Note that right now cont and chan0 are not tested.
# Choices added by cdw are '13co21', 'co10', 'cn10high', and 'cn10low'.

just_product = ['co21']
#just_product = ['co21','13co21','c18o21','co10','cn10high','cn10low']

# ... set these variables to indicate what steps of the script should
# be performed.

rebuild_directories = True

stage_cubes = True
stage_singledish = True

primary_beam_correct = True
convolve_to_round_beam = True

prep_for_feather = True
# only use Chris's new feather order
feather_data = False
reverse_feather_data = True

cleanup_cubes = True

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Directories and definitions
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

interferometric_array_list = [
    '7m',
    '12m_com',
    '12m_ext',
    '12m_com+7m',
    '12m_ext+12m_com',
    '12m_ext_12m_com+7m',
]
full_array_list = [
    '7m',
    '12m_com',
    '12m_ext',
    '12m_com+7m',
    '12m_ext+12m_com',
    '12m_ext+12m_com+7m',
    '7m+tp',
    '12m_com+7m+tp',
    '12m_ext+12m_com+7m+tp',
]
full_product_list = ['co21','c18o21','13co21','co10','cn10high','cn10low']
gal_part_list = pp.list_gal_names()
dir_key = pp.read_dir_key()

inroot_dir = '../'
vstring = 'v0'
outroot_dir = '../release/'+vstring+'/'

cutoff = 0.25

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Wipe and rebuild if requested
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
if rebuild_directories:
    pcp.rebuild_directories(outroot_dir=outroot_dir)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Loop over all galaxies to stage, process, mosaic, and cleanup
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

for this_loop in ['stage', 'process', 'feather', 'cleanup']:

    casalog.origin(casa_log_origin)
    casalog.post("", "INFO")
    casalog.post("Looping over all galaxies and products.", "INFO")
    casalog.post("... this loop is to: "+this_loop, "INFO")
    casalog.post("", "INFO")

    before_first = True
    after_last = False

    for gal_part in gal_part_list:

        if gal_part in dir_key.keys():
            whole_gal = dir_key[gal_part]
            is_mosaic = True
        else:
            whole_gal = gal_part
            is_mosaic = False

        if len(only) > 0:
            not_in_list = (only.count(gal_part) == 0) and \
                (only.count(whole_gal) == 0)
            if not_in_list:
                casalog.origin(casa_log_origin)
                casalog.post("Skipping galaxy part "+gal_part, "INFO")
                continue

        if len(skip) > 0:
            in_list = (skip.count(gal_part) > 0) or \
                (skip.count(whole_gal) > 0)
            if in_list:
                casalog.origin(casa_log_origin)
                casalog.post("Skipping galaxy part "+gal_part, "INFO")
                continue

        if first != "":
            if gal_part == first:
                before_first = False
                if before_first:
                    continue
                
        if last != "":
            if after_last == True:
                continue
            if gal_part == last:
                after_last = True

        casalog.origin(casa_log_origin)
        casalog.post("Processing galaxy part: "+gal_part, "INFO")
        casalog.post("... part of whole galaxy: "+whole_gal, "INFO")

        for product in full_product_list:
            
            if len(just_product) > 0:
                if just_product.count(product) == 0:
                    casalog.origin(casa_log_origin)
                    casalog.post("... skipping line/product: "+product, "INFO")
                    continue

            casalog.origin(casa_log_origin)
            casalog.post("... processing line/product: "+product, "INFO")

            if this_loop == 'stage' and stage_singledish:
                pcp.phangs_stage_singledish(
                    gal=gal_part, product=product, 
                    root_dir = outroot_dir, 
                    overwrite=True
                    )

            for array in full_array_list:

                if len(just_array) > 0:
                    if just_array.count(array) == 0:
                        casalog.origin(casa_log_origin)
                        casalog.post("... ... skipping array "+array, "INFO")
                        continue

                casalog.origin(casa_log_origin)
                casalog.post("... ... processing array: "+array, "INFO")

                if this_loop == 'stage' and stage_cubes:
                    if (array == '7m+tp'
                        or array == '12m_com+7m+tp'
                        or array == '12m_ext+12m_com+7m+tp'):
                        continue
                    pcp.phangs_stage_cubes(
                        gal=gal_part, array=array, product=product, 
                        root_dir = outroot_dir, 
                        overwrite=True
                        )
                    
                if this_loop == 'process' and primary_beam_correct:
                    if (array == '7m+tp'
                        or array == '12m_com+7m+tp'
                        or array == '12m_ext+12m_com+7m+tp'):
                        continue
                    pcp.phangs_primary_beam_correct(
                        gal=gal_part, array=array, product=product, 
                        root_dir=outroot_dir,
                        overwrite=True, cutoff=cutoff,
                        )

                if this_loop == 'process' and convolve_to_round_beam:
                    if (array == '7m+tp'
                        or array == '12m_com+7m+tp'
                        or array == '12m_ext+12m_com+7m+tp'):
                        continue
                    pcp.phangs_convolve_to_round_beam(
                        gal=gal_part, array=array, product=product, 
                        root_dir=outroot_dir,
                        overwrite=True
                        )

                if this_loop == 'feather' and prep_for_feather:
                    if (array == '12m_ext'
                        or array == '12m_com'
                        or array == '12m_ext+12m_com'
                        or array == '7m+tp'
                        or array == '12m_com+7m+tp'
                        or array == '12m_ext+12m_com+7m+tp'):
                        continue
                    pcp.prep_for_feather(
                        gal=gal_part, array=array, product=product,
                        root_dir=outroot_dir,
                        overwrite=True
                        )

                if this_loop == 'feather' and feather_data:
                    if (array == '12m_ext'
                        or array == '12m_com'
                        or array == '12m_ext+12m_com'
                        or array == '7m+tp'
                        or array == '12m_com+7m+tp'
                        or array == '12m_ext+12m_com+7m+tp'):
                        continue
                    pcp.phangs_feather_data(
                        gal=gal_part, array=array, product=product,
                        root_dir=outroot_dir,
                        overwrite=True, cutoff=cutoff,
                        )

                if this_loop == 'feather' and reverse_feather_data:
                    if (array == '12m_ext'
                        or array == '12m_com'
                        or array == '12m_ext+12m_com'
                        or array == '7m+tp'
                        or array == '12m_com+7m+tp'
                        or array == '12m_ext+12m_com+7m+tp'):
                        continue
                    pcp.chris_feather_data(
                        gal=gal_part, array=array, product=product,
                        root_dir=outroot_dir,
                        overwrite=True, cutoff=cutoff,
                        )
                
                if this_loop == 'cleanup' and cleanup_cubes:
                    pcp.phangs_cleanup_cubes(
                        gal=gal_part, array=array, product=product,
                        root_dir=outroot_dir,
                        overwrite=True
                        )

casalog.setlogfile(orig_log_path)
