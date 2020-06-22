# This script stages the imaging. It copies the calibrated data into
# the imaging directories, regrids to the desired velocity axis, and
# then extracts line and continuum data sets ready for imaging.

# Edit the "Control Flow" section to use the script.

import os
import phangsPipeline as pp
import analysisUtils as au
import glob
import sys
import datetime

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Control Flow
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# text file path to redirect terminal output
utc_now_str = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
log_path = "/home/brunettn/antennae/imaging/logs/stage_imaging_{:}.log".format(utc_now_str)
sys.stdout = open(log_path, "w", 1)
sys.stderr = sys.stdout

# ... a list of directories
data_dirs = [
    '/home/brunettn/antennae/2018.1.00272.S/'
    ]

# ... a text list. The script will process only these galaxies.

only = [
    "ngc_4038_4039"
    ]

# ... skip these galaxies.

skip = []

# ... start with this galaxy

first = ""
last = ""

# ... set this to '12m' or '7m' to stage data only for those
# arrays. Leave it as None to process all data. If both 12m and 7m
# data are processed, then the script will also create 12m+7m data. So
# you need to rerun the staging when both data sets arrive.
#
# This looks like it should be the "array code tag" from the ms_file_key.txt
# file. -brunettn

just_array = None

# List of lines to process. There's not a lot of error catching
# here. It needs to be a list and it only knows about co21 and c18o21
# right now. It's inflexible because the spectral regridding
# parameters remain hard-coded until we come up with a more general
# solution to the rebin-and-regrid problem.

lines = ['co21']
#lines = ['co21', 'c18o21', '13co21']
#lines = ['co21', 'c18o21','13co21']
#lines = ['13co21','c18o21']
#lines = ['c18o21']
#lines = ['co21', 'c18o21','13co21','co10','cn10high','cn10low']

# ... set these variables to indicate what steps of the script should
# carry out. The steps do:

# do_copy - copy the data from the calibrated data directory to the
# working directory. Will create the working directory for this galaxy
# if it doesn't exist already. Uses "ms_file_key.txt" and maps
# multi-part galaxies to directories using "dir_key.txt"

# do_custom_scripts - Will run custom processing for each galaxy. For
# example, additional flagging and uv continuum subtraction.

# do_extract_lines - extract lines from the measurement set and regrid
# them onto our working grid. The line set is assumed to be 12co21 and
# c18o21 for PHANGS. The velocity grid is defined in
# "mosaic_definitions.txt" and this step should be rerun after
# changing that grid. Also makes "channel 0" (line integrated) data
# sets.

# do_extract_cont - extract a single-channel continuum data set from
# the measurement set, first flagging lines. The velocity windows used
# for flagging lines is set in "mosaic_definitions.txt"

do_copy = False
do_custom_scripts = False
do_extract_lines = True
do_concat_lines = True
do_extract_cont = False
do_concat_cont = False
do_only_new = False
do_cleanup = False

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Loop
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

gals = pp.list_gal_names()

before_first = True
after_last = False

for gal in gals:

    if len(only) > 0:
        if only.count(gal) == 0:
            continue
    if len(skip) > 0:
        if skip.count(gal) > 0:
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
    
    if do_only_new:        
        this_dir = pp.dir_for_gal(gal)
        has_12m = len(glob.glob(this_dir+gal+'*12m_co21.ms')) > 0
        has_7m = len(glob.glob(this_dir+gal+'*7m_co21.ms')) > 0
        if has_12m or has_7m:
            print ""
            print "... You requested to only stage new data."
            print "... I found an existing file for "+gal+" ."
            print "... I will skip that galaxy."
            print ""
            continue

    # Copy the calibrated data to the working directory. Split out the
    # source observations. The copies will be deleted late, but ensure
    # that nothing we do should damage the original calibrated MSes.

    if do_copy:
        pp.copy_data(
            gal=gal,
            just_array=just_array,
            do_split=True,
            do_statwt=False,
            quiet=False,
            data_dirs=data_dirs)

    # Optionally, run custom scripts at this stage. This could, for
    # example, flag data or carry out uv continuum subtraction. The
    # line and continuum extensions defined here point the subsequent
    # programs at the processed data.

    line_ext = ''
    cont_ext = ''
    if do_custom_scripts:
        scripts_for_this_gal = glob.glob('../scripts/custom_staging_scripts/'+gal+'_staging_script.py')
        for this_script in scripts_for_this_gal:
            execfile(this_script)

    # Extract lines, includes regridding and rebinning to the velocity
    # grid specified in the text file keys. Runs statwt afterwards,
    # the result is a bunch of line-only data sets but still
    # execution-by-execution.

    if do_extract_lines:
        pp.extract_phangs_lines(
            gal=gal,
            just_array=just_array,
            quiet=False,
            append_ext=line_ext,
            lines=lines)

    # Concatenate the extracted lines into the measurement sets that
    # we will use for imaging. This step also makes a "channel 0"
    # measurement for each line.

    if do_concat_lines:
        pp.concat_phangs_lines(
            gal=gal,
            just_array=just_array,
            quiet=False,
            lines=lines)

    # Extract the continuum, avoiding lines and averaging all
    # frequencies in each SPW together. This step also uses statwt to
    # empirically weight the data.

    if do_extract_cont:
        pp.extract_phangs_continuum(
            gal=gal,
            just_array=just_array,
            quiet=False,
            do_statwt=True,
            append_ext=cont_ext)

    if do_concat_cont:
        pp.concat_phangs_continuum(
            gal=gal,
            just_array=just_array,
            quiet=False)

    # Remove intermediate files. The big space-savers here are the
    # initial copies of the data. The data after frequency averaging
    # are smaller by a large factor (~10). For reference, re-copying
    # all of the PHANGS-ALMA LP takes less than a day on the OSU
    # system. Full line and continuum exraction takes longer.

    if do_cleanup:
        pp.cleanup_phangs_staging(
            gal=gal,
            just_array=just_array)

sys.stdout.close()
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
