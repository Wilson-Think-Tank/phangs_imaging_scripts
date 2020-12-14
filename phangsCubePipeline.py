# NB: CASA doesn't always include the pwd in the module search path. I
# had to modify my init.py file to get this to import. See the README.

import os
import numpy as np
import pyfits
import scipy.ndimage as ndimage
import glob

# Other PHANGS scripts
from phangsPipelinePython import *
import line_list

# Analysis utilities
import analysisUtils as au

# CASA imports
from taskinit import *

# Import specific CASA tasks
from exportfits_cli import exportfits_cli as exportfits
from feather_cli import feather_cli as feather
from imhead_cli import imhead_cli as imhead
from immath_cli import immath_cli as immath
from impbcor_cli import impbcor_cli as impbcor
from importfits_cli import importfits_cli as importfits
from imrebin_cli import imrebin_cli as imrebin
from imregrid_cli import imregrid_cli as imregrid
from imsmooth_cli import imsmooth_cli as imsmooth
from imsubimage_cli import imsubimage_cli as imsubimage
from rmtables_cli import rmtables_cli as rmtables
from pipelineVersion import version as pipeVer
# Physical constants
sol_kms = 2.9979246e5

casa_log_origin = "phangsCubePipeline"
casalog.showconsole(onconsole=False)

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# DIRECTORY STRUCTURE AND FILE MANAGEMENT
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

def rebuild_directories(outroot_dir=None):
    """
    Reset and rebuild the directory structure.
    """

    log_file = casalog.logfile()
    
    if outroot_dir is None:
        casalog.origin(casa_log_origin)
        casalog.post("Specify a root directory.", "SEVERE", "rebuild_directories")
        return

    check_string = raw_input("Reseting directory structure in "+\
                                 outroot_dir+". Type 'Yes' to confirm.")
    if check_string != "Yes":
        casalog.origin(casa_log_origin)
        casalog.post("Aborting", "SEVERE", "rebuild_directories")
        return

    if os.path.isdir(outroot_dir) == False:
        casalog.origin(casa_log_origin)
        casalog.post("Got make "+outroot_dir+" manually. Then I will make the subdirectories.", "INFO", "rebuild_directories")

    for subdir in ['raw/','process/','products/']:
        casalog.origin(casa_log_origin)
        os.system('rm -rf '+outroot_dir+subdir+" >> "+log_file+" 2>&1")
        os.system('mkdir --parents '+outroot_dir+subdir+" >> "+log_file+" 2>&1")

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# SET UP THE CUBES
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

def phangs_stage_cubes(
    gal=None, array=None, product=None,
    root_dir=None, 
    overwrite=False,
    ):
    """
    Stage cubes for further processing in CASA. This imports them back
    into CASA as .image files.
    """

    if gal is None or array is None or product is None or \
            root_dir is None:
        casalog.origin(casa_log_origin)
        casalog.post("Missing required input.", "SEVERE", "phangs_stage_cubes")
        return
    
    input_dir = dir_for_gal(gal)
    in_cube_name = input_dir+gal+'_'+array+'_'+product+'.fits'
    in_pb_name = input_dir+gal+'_'+array+'_'+product+'_pb.fits'

    out_dir = root_dir+'raw/'
    out_cube_name = out_dir+gal+'_'+array+'_'+product+'.image'
    out_pb_name = out_dir+gal+'_'+array+'_'+product+'.pb'

    casalog.origin(casa_log_origin)
    casalog.post("... importing data for "+in_cube_name, "INFO", "phangs_stage_cubes")

    if os.path.isfile(in_cube_name):
        importfits(fitsimage=in_cube_name, imagename=out_cube_name,
                   zeroblanks=True, overwrite=overwrite)
    else:
        casalog.origin(casa_log_origin)
        casalog.post("File not found "+in_cube_name, "SEVERE", "phangs_stage_cubes")

    if os.path.isfile(in_pb_name):
        importfits(fitsimage=in_pb_name, imagename=out_pb_name,
                   zeroblanks=True, overwrite=overwrite)
    else:
        casalog.origin(casa_log_origin)
        casalog.post("Directory not found "+in_pb_name, "SEVERE", "phangs_stage_cubes")

def phangs_stage_singledish(
    gal=None, product=None, root_dir=None, 
    overwrite=False):
    """
    Copy the single dish data for further processing
    """

    log_file = casalog.logfile()

    if gal is None or product is None or \
            root_dir is None:
        casalog.origin(casa_log_origin)
        casalog.post("Missing required input.", "SEVERE", "phangs_stage_singledish")
        return
    
    sdk = read_singledish_key()
    if (gal in sdk.keys()) == False:
        casalog.origin(casa_log_origin)
        casalog.post(gal+" not found in single dish key.", "SEVERE", "phangs_stage_singledish")
        return
    
    this_key = sdk[gal]
    if (product in this_key.keys()) == False:
        casalog.origin(casa_log_origin)
        casalog.post(product+" not found in single dish key for "+gal, "SEVERE", "phangs_stage_singledish")
        return
    
    sdfile_in = this_key[product]
    
    sdfile_out = root_dir+'raw/'+gal+'_tp_'+product+'.image'    

    casalog.origin(casa_log_origin)
    casalog.post("... importing single dish data for "+sdfile_in, "INFO", "phangs_stage_singledish")

    importfits(fitsimage=sdfile_in, imagename=sdfile_out+'.temp',
               zeroblanks=True, overwrite=overwrite)

    if overwrite:
        casalog.origin(casa_log_origin)
        os.system('rm -rf '+sdfile_out+" >> "+log_file+" 2>&1")
    imsubimage(imagename=sdfile_out+'.temp', outfile=sdfile_out,
               dropdeg=True)
    casalog.origin(casa_log_origin)
    os.system('rm -rf '+sdfile_out+'.temp'+" >> "+log_file+" 2>&1")

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# BASIC IMAGE PROCESSING STEPS
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

def phangs_primary_beam_correct(
    gal=None, array=None, product=None, root_dir=None, 
    cutoff=0.25, overwrite=False):
    """
    Construct primary-beam corrected images using PHANGS naming conventions.
    """

    input_dir = root_dir+'raw/'
    input_cube_name = input_dir+gal+'_'+array+'_'+product+'.image'
    input_pb_name = input_dir+gal+'_'+array+'_'+product+'.pb'
    output_dir = root_dir+'process/'
    output_cube_name = output_dir+gal+'_'+array+'_'+product+'_pbcorr.image'

    casalog.origin(casa_log_origin)
    casalog.post("", "INFO", "phangs_primary_beam_correct")
    casalog.post("... producing a primary beam corrected image for "+input_cube_name, "INFO", "phangs_primary_beam_correct")
    casalog.post("", "INFO", "phangs_primary_beam_correct")
    
    primary_beam_correct(
        infile=input_cube_name, 
        pbfile=input_pb_name,
        outfile=output_cube_name,
        cutoff=cutoff, overwrite=overwrite)

def primary_beam_correct(
    infile=None, pbfile=None, outfile=None, 
    cutoff=0.25, overwrite=False):
    """
    Construct a primary-beam corrected image.
    """

    log_file = casalog.logfile()

    if infile is None or pbfile is None or outfile is None:
        casalog.origin(casa_log_origin)
        casalog.post("Missing required input.", "SEVERE", "primary_beam_correct")
        return

    if os.path.isdir(infile) == False:
        casalog.origin(casa_log_origin)
        casalog.post("Input file missing - "+infile, "SEVERE", "primary_beam_correct")
        return

    if os.path.isdir(pbfile) == False:
        casalog.origin(casa_log_origin)
        casalog.post("Primary beam file missing - "+pbfile, "SEVERE", "primary_beam_correct")
        return

    if overwrite:
        casalog.origin(casa_log_origin)
        os.system('rm -rf '+outfile+" >> "+log_file+" 2>&1")

    impbcor(imagename=infile, pbimage=pbfile, outfile=outfile, cutoff=cutoff)

def phangs_convolve_to_round_beam(
    gal=None, array=None, product=None, root_dir=None, 
    force_beam=None, overwrite=False):
    """
    Construct primary-beam corrected images using PHANGS naming
    conventions. Runs on both primary beam corrected cube and flat
    cube, forcing the same beam for both.
    """

    if gal is None or array is None or product is None or \
            root_dir is None:
        casalog.origin(casa_log_origin)
        casalog.post("Missing required input.", "SEVERE", "phangs_convolve_to_round_beam")
        return
    
    input_dir = root_dir+'process/'
    input_cube_name = input_dir+gal+'_'+array+'_'+product+'_pbcorr.image'
    output_dir = root_dir+'process/'
    output_cube_name = output_dir+gal+'_'+array+'_'+product+'_pbcorr_round.image'

    casalog.origin(casa_log_origin)
    casalog.post("", "INFO", "phangs_convolve_to_round_beam")
    casalog.post("... convolving to a round beam for "+input_cube_name, "INFO", "phangs_convolve_to_round_beam")
    casalog.post("", "INFO", "phangs_convolve_to_round_beam")

    round_beam = \
        convolve_to_round_beam(
        infile=input_cube_name,
        outfile=output_cube_name,
        force_beam=force_beam,
        overwrite=overwrite)

    casalog.origin(casa_log_origin)
    casalog.post("", "INFO", "phangs_convolve_to_round_beam")
    casalog.post("... found beam of "+str(round_beam)+" arcsec. Forcing flat cube to this.", "INFO", "phangs_convolve_to_round_beam")
    casalog.post("", "INFO", "phangs_convolve_to_round_beam")

    input_dir = root_dir+'raw/'
    input_cube_name = input_dir+gal+'_'+array+'_'+product+'.image'
    output_dir = root_dir+'process/'
    output_cube_name = output_dir+gal+'_'+array+'_'+product+'_flat_round.image'    

    convolve_to_round_beam(
        infile=input_cube_name,
        outfile=output_cube_name,
        force_beam=round_beam,
        overwrite=overwrite)
    
def convolve_to_round_beam(
    infile=None, outfile=None, force_beam=None, overwrite=False):
    """
    Convolve supplied image to have a round beam.
    """

    if infile is None or outfile is None:
        casalog.origin(casa_log_origin)
        casalog.post("Missing required input.", "SEVERE", "convolve_to_round_beam")
        return

    if os.path.isdir(infile) == False:
        casalog.origin(casa_log_origin)
        casalog.post("Input file missing - "+infile, "SEVERE", "convolve_to_round_beam")
        return    

    if force_beam is None:
        hdr = imhead(infile)

        if (hdr['axisunits'][0] != 'rad'):
            casalog.origin(casa_log_origin)
            casalog.post("Based on CASA experience. I expected units of radians.", "SEVERE", "convolve_to_round_beam")
            casalog.post("I did not find this. Returning. Adjust code or investigate file "+infile, "SEVERE", "convolve_to_round_beam")
            return
        pixel_as = abs(hdr['incr'][0]/np.pi*180.0*3600.)

        if (hdr['restoringbeam']['major']['unit'] != 'arcsec'):
            casalog.origin(casa_log_origin)
            casalog.post("Based on CASA experience. I expected units of arcseconds for the beam.", "SEVERE", "convolve_to_round_beam")
            casalog.post("I did not find this. Returning. Adjust code or investigate file "+infile, "SEVERE", "convolve_to_round_beam")
            return    
        bmaj = hdr['restoringbeam']['major']['value']    
        target_bmaj = np.sqrt((bmaj)**2+(2.0*pixel_as)**2)
    else:
        target_bmaj = force_beam

    imsmooth(imagename=infile,
             outfile=outfile,
             targetres=True,
             major=str(target_bmaj)+'arcsec',
             minor=str(target_bmaj)+'arcsec',
             pa='0.0deg',
             overwrite=overwrite
             )

    return target_bmaj

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# ROUTINES FOR FEATHERING THE DATA
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

def get_cube_angular_extent(cube):
    """Return the RA and Dec ranges of the given cube.

    Parameters
    ----------
    cube : str
        Path to the cube you want the angular extent for.

    Returns
    -------
    extent : tuple
        Tuple of tuples where the first row (i.e. extent[0]) contains the sorted
        RA angular extents, and the second row (i.e. extent[1]) contains the
        sorted Dec. extents.
    """
    hdr = imhead(imagename=cube)

    ra_edge_1 = hdr["refval"][0] - (hdr["refpix"][0] * hdr["incr"][0])
    dec_edge_1 = hdr["refval"][1] - (hdr["refpix"][1] * hdr["incr"][1])
    ra_edge_2 = (hdr["refval"][0]
                 + ((hdr["shape"][0] - hdr["refpix"][0] - 1) * hdr["incr"][0]))
    dec_edge_2 = (hdr["refval"][1]
                  + ((hdr["shape"][1] - hdr["refpix"][1] - 1) * hdr["incr"][1]))

    ra_edges_sorted = np.sort([ra_edge_1, ra_edge_2])
    dec_edges_sorted = np.sort([dec_edge_1, dec_edge_2])

    return (
        (ra_edges_sorted[0], ra_edges_sorted[1]),
        (dec_edges_sorted[0], dec_edges_sorted[1]),
    )

def check_angular_extent_encompasses_other(extent_1, extent_2):
    """Return whether the first given extent completely encompasses the second.

    Parameters
    ----------
    extent_1 : tuple
        Output of get_cube_angular_extent function run on cube_1.
    extent_2 : tuple
        Output of get_cube_angular_extent function run on cube_2.

    Returns
    -------
    encompasses_other : bool
        Whether the first extent completely encompasses the second.

    Notes
    -----
    Extents should come from the get_cube_angular_extent function.
    """
    encompassed = list()
    for i in range(2): # 0 should be RA, 1 should be Dec
        if (extent_2[i][0] >= extent_1[i][0]
            and extent_2[i][1] >= extent_1[i][0]
            and extent_2[i][0] <= extent_1[i][1]
            and extent_2[i][1] <= extent_1[i][1]):
            encompassed.append(True)
        else:
            encompassed.append(False)

    return np.all(encompassed)

def get_n_pad_pixels_to_encompass(cube_1, cube_2, extent_1=None, extent_2=None):
    """Return number of pixels needed to pad angular edges of first cube to encompass the second.

    Parameters
    ----------
    cube_1 : str
        Path to the cube you want to encompass the second.
    cube_2 : str
        Path to the cube you want encompassed by the first.
    extent_1 : tuple, optional
        Output of get_cube_angular_extent function run on cube_1. Optional
        because get_cube_angular_extent will just be run here if not provided.
    extent_2 : tuple, optional
        Output of get_cube_angular_extent function run on cube_2. Optional
        because get_cube_angular_extent will just be run here if not provided.

    Returns
    -------
    n_pad_pixels : int
        Number of pixels required to pad on each angular side of cube_1 to
        completely encompass cube_2.

    Notes
    -----
    This assumes the first cube does not already encompass the second in both
    angular dimensions. If the cube_1 is larger than cube_2 in a dimension and
    that is the largest difference between all dimensions then it will probably
    return that as the padding instead of ignoring that case (i.e. not perferct
    as-is).
    """
    if extent_1 == None:
        extent_1 = get_cube_angular_extent(cube_1)
    if extent_2 == None:
        extent_2 = get_cube_angular_extent(cube_2)

    hdr_1 = imhead(imagename=cube_1)

    delta_angular = np.abs(np.array(extent_1) - np.array(extent_2))

    delta_pix = [
        delta_angular[0] / hdr_1["incr"][0],
        delta_angular[1] / hdr_1["incr"][1],
    ]

    return 10 + int(np.ceil(np.max(np.abs(delta_pix))))

def prep_for_feather(
    gal=None, array=None, product=None, root_dir=None, 
    overwrite=False):
    """
    Prepare the single dish data for feathering
    """

    log_file = casalog.logfile()
    
    if gal is None or array is None or product is None or \
            root_dir is None:
        casalog.origin(casa_log_origin)
        casalog.post("Missing required input.", "SEVERE", "prep_for_feather")
        return    

    sdfile_in = root_dir+'raw/'+gal+'_tp_'+product+'.image'
    sdfile_out = root_dir+'process/'+gal+'_tp_'+product+'_align_'+array+'.image'
    interf_in = root_dir+'process/'+gal+'_'+array+'_'+product+'_pbcorr_round.image'
    pbfile_name = root_dir+'raw/'+gal+'_'+array+'_'+product+'.pb'    

    if (os.path.isdir(sdfile_in) == False):
        casalog.origin(casa_log_origin)
        casalog.post("Single dish file not found: "+sdfile_in, "SEVERE", "prep_for_feather")
        return

    if (os.path.isdir(interf_in) == False):
        casalog.origin(casa_log_origin)
        casalog.post("Interferometric file not found: "+interf_in, "SEVERE", "prep_for_feather")
        return

    if (os.path.isdir(pbfile_name) == False):
        casalog.origin(casa_log_origin)
        casalog.post("Primary beam file not found: "+pbfile_name, "SEREVE", "prep_for_feather")
        return

    # pad the interferometric cube in RA and Dec. if it does not fully cover the
    # TP FoV
    interf_extent = get_cube_angular_extent(interf_in)
    sd_extent = get_cube_angular_extent(sdfile_in)
    interf_covers_sd = check_angular_extent_encompasses_other(
        interf_extent,
        sd_extent,
    )
    if not interf_covers_sd:
        casalog.origin(casa_log_origin)
        casalog.post("Interferometric FoV does not fully cover TP FoV.", "INFO", "prep_for_feather")
        casalog.post("Padding the interferometric intensity and primary-beam cubes.", "INFO", "prep_for_feather")

        n_pad_pixels = get_n_pad_pixels_to_encompass(
            interf_in,
            sdfile_in,
            extent_1=interf_extent,
            extent_2=sd_extent,
        )

        myia = au.createCasaTool(iatool)
        myia.open(interf_in)
        myia.pad(
            outfile=interf_in + ".pad_tmp",
            npixels=n_pad_pixels,
            wantreturn=False,
        )
        myia.done()
        myia.open(pbfile_name)
        myia.pad(outfile=pbfile_name + ".pad_tmp",
            npixels=n_pad_pixels,
            wantreturn=False,)
        myia.done()

        rmtables([interf_in, pbfile_name])
        os.rename(interf_in + ".pad_tmp", interf_in)
        os.rename(pbfile_name + ".pad_tmp", pbfile_name)

    # Align the relevant TP data to the product.
    imregrid(imagename=sdfile_in,
             template=interf_in,
             output=sdfile_out,
             asvelocity=True,
             axes=[-1],
             interpolation='cubic',
             overwrite=overwrite)


def phangs_feather_data(
    gal=None, array=None, product=None, root_dir=None, 
    overwrite=False):
    """
    Feather the pbcorr'd interferometric and aligned total power data
    """

    log_file = casalog.logfile()

    if gal is None or array is None or product is None or \
            root_dir is None:
        casalog.origin(casa_log_origin)
        casalog.post("Missing required input.", "SEVERE", "phangs_feather_data")
        return    

    sdfile_in = root_dir+'process/'+gal+'_tp_'+product+'_align_'+array+'.image'
    interf_in = root_dir+'process/'+gal+'_'+array+'_'+product+'_pbcorr_round.image'
    pbfile_name = root_dir+'raw/'+gal+'_'+array+'_'+product+'.pb' 

    if (os.path.isdir(sdfile_in) == False):
        casalog.origin(casa_log_origin)
        casalog.post("Single dish file not found: "+sdfile_in, "SEVERE", "phangs_feather_data")
        return
        
    if (os.path.isdir(interf_in) == False):
        casalog.origin(casa_log_origin)
        casalog.post("Interferometric file not found: "+interf_in, "SEVERE", "phangs_feather_data")
        return

    if (os.path.isdir(pbfile_name) == False):
        casalog.origin(casa_log_origin)
        casalog.post("Primary beam file not found: "+pbfile_name, "SEVERE", "phangs_feather_data")
        return

    # Feather the "pbcorr"d inteferometric and "align"d TP data.
    outfile_name = root_dir+'process/'+gal+'_'+array+'+tp_'+product+ \
        '_pbcorr_round.image'

    if overwrite:
        casalog.origin(casa_log_origin)
        os.system('rm -rf '+outfile_name+" >> "+log_file+" 2>&1")
    os.system('rm -rf '+outfile_name+'.temp'+" >> "+log_file+" 2>&1")
    feather(imagename=outfile_name+'.temp',
            highres=interf_in,
            lowres=sdfile_in,
            sdfactor=1.0,
            lowpassfiltersd=False)
    imsubimage(imagename=outfile_name+'.temp', outfile=outfile_name,
               dropdeg=True)
    casalog.origin(casa_log_origin)
    os.system('rm -rf '+outfile_name+'.temp'+" >> "+log_file+" 2>&1")
    infile_name = outfile_name

# apply primary beam to flatten the feathered data.
    outfile_name = root_dir+'process/'+gal+'_'+array+'+tp_'+product+ \
        '_flat_round.image'
    
    if overwrite:
        casalog.origin(casa_log_origin)
        os.system('rm -rf '+outfile_name+" >> "+log_file+" 2>&1")

    casalog.origin(casa_log_origin)
    casalog.post(infile_name, "INFO", "phangs_feather_data")
    casalog.post(pbfile_name, "INFO", "phangs_feather_data")
    impbcor(imagename=infile_name,
            pbimage=pbfile_name, 
            outfile=outfile_name, 
            mode='multiply')

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# CLEANUP
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

def convert_jytok(
    infile=None, outfile=None, overwrite=False, inplace=False):
    """
    Convert a cube from Jy/beam to K.
    """

    log_file = casalog.logfile()

    c = 2.99792458e10
    h = 6.6260755e-27
    kb = 1.380658e-16

    if infile is None or (outfile is None and inplace==False):
        casalog.origin(casa_log_origin)
        casalog.post("Missing required input.", "SEVERE", "convert_jytok")
        return
    
    if os.path.isdir(infile) == False:
        casalog.origin(casa_log_origin)
        csalog.post("Input file not found: "+infile, "SEVERE", "convert_jytok")
        return
    
    if inplace == False:
        if overwrite:
            casalog.origin(casa_log_origin)
            os.system('rm -rf '+outfile+" >> "+log_file+" 2>&1")
        
        if os.path.isdir(outfile):
            casalog.origin(casa_log_origin)
            casalog.post("Output file already present: "+outfile, "SEVERE", "convert_jytok")
            return

        casalog.origin(casa_log_origin)
        os.system('cp -r '+infile+' '+outfile+" >> "+log_file+" 2>&1")
        target_file = outfile
    else:
        target_file = infile

    hdr = imhead(target_file, mode='list')
    unit = hdr['bunit']
    if unit != 'Jy/beam':
        casalog.origin(casa_log_origin)
        casalog.post("Unit is not Jy/beam. Returning.", "SEVERE", "convert_jytok")
        return

    #restfreq_hz = hdr['restfreq'][0]

    if hdr['cunit3'] != 'Hz':
        casalog.origin(casa_log_origin)
        casalog.post("I expected frequency as the third axis but did not find it.", "SEVERE", "convert_jytok")
        casalog.post("Returning.", "SEVERE", "convert_jytok")
        return
    
    crpix3 = hdr['crpix3']
    cdelt3 = hdr['cdelt3']
    crval3 = hdr['crval3']
    naxis3 = hdr['shape'][2]
    faxis_hz = (np.arange(naxis3)+1.-crpix3)*cdelt3+crval3
    freq_hz = np.mean(faxis_hz)
    
    bmaj_unit = hdr['beammajor']['unit']
    if bmaj_unit != 'arcsec':
        casalog.origin(casa_log_origin)
        casalog.post("Beam unit is not arcsec, which I expected. Returning.", "SEVERE", "convert_jytok")
        casalog.post("Unit instead is "+bmaj_unit, "SEVERE", "convert_jytok")
        return    
    bmaj_as = hdr['beammajor']['value']
    bmin_as = hdr['beamminor']['value']
    bmaj_sr = bmaj_as/3600.*np.pi/180.
    bmin_sr = bmin_as/3600.*np.pi/180.
    beam_in_sr = np.pi*(bmaj_sr/2.0*bmin_sr/2.0)/np.log(2)
    
    jtok = c**2 / beam_in_sr / 1e23 / (2*kb*freq_hz**2)

    myia = au.createCasaTool(iatool)
    myia.open(target_file)
    vals = myia.getchunk()
    vals *= jtok
    myia.putchunk(vals)
    myia.setbrightnessunit("K")
    myia.close()

    imhead(target_file, mode='put', hdkey='JTOK', hdvalue=jtok)

    return

def trim_cube(    
    infile=None, outfile=None, overwrite=False, inplace=False, min_pixperbeam=3):
    """
    Trim and rebin a cube to smaller size.
    """

    log_file = casalog.logfile()
    
    if infile is None or outfile is None:
        casalog.origin(casa_log_origin)
        casalog.post("Missing required input.", "SEVERE", "trim_cube")
        return
    
    if os.path.isdir(infile) == False:
        casalog.origin(casa_log_origin)
        casalog.post("Input file not found: "+infile, "SEVERE", "trim_cube")
        return

    # First, rebin if needed
    hdr = imhead(infile)
    if (hdr['axisunits'][0] != 'rad'):
        casalog.origin(casa_log_origin)
        casalog.post("Based on CASA experience. I expected units of radians.", "SEVERE", "trim_cube")
        casalog.post("I did not find this. Returning. Adjust code or investigate file "+infile, "SEVERE", "trim_cube")
        return

    pixel_as = abs(hdr['incr'][0]/np.pi*180.0*3600.)

    if (hdr['restoringbeam']['major']['unit'] != 'arcsec'):
        casalog.origin(casa_log_origin)
        casalog.post("Based on CASA experience. I expected units of arcseconds for the beam.", "SEVERE", "trim_cube")
        casalog.post("I did not find this. Returning. Adjust code or investigate file "+infile, "SEVERE", "trim_cube")
        return    
    bmaj = hdr['restoringbeam']['major']['value']    
    
    pix_per_beam = bmaj*1.0 / pixel_as*1.0
    
    if pix_per_beam > 6:
        imrebin(
            imagename=infile,
            outfile=outfile+'.temp',
            factor=[2,2,1],
            crop=True,
            dropdeg=True,
            overwrite=overwrite,
            )
    else:
        casalog.origin(casa_log_origin)
        os.system('cp -r '+infile+' '+outfile+'.temp'+" >> "+log_file+" 2>&1")

    # Figure out the extent of the image inside the cube
    myia = au.createCasaTool(iatool)
    myia.open(outfile+'.temp')
    mask = myia.getchunk(getmask=True)    
    myia.close()

    this_shape = mask.shape

    mask_spec_x = np.sum(np.sum(mask*1.0,axis=2),axis=1) > 0
    pad = 0
    xmin = np.max([0,np.min(np.where(mask_spec_x))-pad])
    xmax = np.min([np.max(np.where(mask_spec_x))+pad,mask.shape[0]-1])

    mask_spec_y = np.sum(np.sum(mask*1.0,axis=2),axis=0) > 0
    ymin = np.max([0,np.min(np.where(mask_spec_y))-pad])
    ymax = np.min([np.max(np.where(mask_spec_y))+pad,mask.shape[1]-1])

    mask_spec_z = np.sum(np.sum(mask*1.0,axis=0),axis=0) > 0
    zmin = np.max([0,np.min(np.where(mask_spec_z))-pad])
    zmax = np.min([np.max(np.where(mask_spec_z))+pad,mask.shape[2]-1])
    
    box_string = ''+str(xmin)+','+str(ymin)+','+str(xmax)+','+str(ymax)
    chan_string = ''+str(zmin)+'~'+str(zmax)

    casalog.origin(casa_log_origin)
    casalog.post("... ... ... box selection: "+box_string, "INFO", "trim_cube")
    casalog.post("... ... ... channel selection: "+chan_string, "INFO", "trim_cube")

    if overwrite:
        casalog.origin(casa_log_origin)
        os.system('rm -rf '+outfile+" >> "+log_file+" 2>&1")
        imsubimage(
        imagename=outfile+'.temp',
        outfile=outfile,
        box=box_string,
        chans=chan_string,
        )

    casalog.origin(casa_log_origin)
    os.system('rm -rf '+outfile+'.temp'+" >> "+log_file+" 2>&1")
    
def phangs_cleanup_cubes(
        gal=None, array=None, product=None, root_dir=None, 
        overwrite=False, min_pixeperbeam=3, roundbeam_tol=0.01, 
        vstring=''):
    """
    Clean up cubes.
    """

    log_file = casalog.logfile()

    if gal is None or array is None or product is None or \
            root_dir is None:
        casalog.origin(casa_log_origin)
        casalog.post("Missing required input.", "SEVERE", "phangs_cleanup_cubes")
        return

#    If necessary, create the galaxy subdirectory in 'products/'  : cdw
    
    if os.path.isdir(root_dir+'products/'+gal+'/') == False:
        casalog.origin(casa_log_origin)
        casalog.post("Making product sudirectory for  "+gal, "INFO", "phangs_cleanup_cubes")
        os.system('mkdir '+root_dir+'products/'+gal+'/'+" >> "+log_file+" 2>&1")

    for this_ext in ['flat', 'pbcorr']:

        root = root_dir+'process/'+gal+'_'+array+'_'+product+'_'+this_ext
        rootfits = root_dir+'products/'+gal+'/'+gal+'_'+array+'_'+product+'_'+this_ext
        infile = root+'_round.image'
        outfile = root+'_round_k.image'
        outfile_fits = rootfits+'_round_k.fits'
    
        if os.path.isdir(infile) == False:
            casalog.origin(casa_log_origin)
            casalog.post("File does not exist: "+infile, "SEVERE", "phangs_cleanup_cubes")
            casalog.post("Returning.", "SEVERE", "phangs_cleanup_cubes")
            return

        # Trim the cube to a smaller size and rebin as needed

        trim_cube(infile=infile, outfile=outfile, 
                  overwrite=overwrite, inplace=False,
                  min_pixperbeam=min_pixeperbeam)

        # Convert to Kelvin

        convert_jytok(infile=outfile, inplace=True)

        # Export to FITS
    
        exportfits(imagename=outfile, fitsimage=outfile_fits,
                   velocity=True, overwrite=True, dropstokes=True, 
                   dropdeg=True, bitpix=-32)
    
        # Clean up headers

        hdu = pyfits.open(outfile_fits)

        hdr = hdu[0].header
        data = hdu[0].data

#        for card in ['BLANK','DATE-OBS','OBSERVER','O_BLANK','O_BSCALE',
#                     'O_BZERO','OBSRA','OBSDEC','OBSGEO-X','OBSGEO-Y','OBSGEO-Z',
#                     'DISTANCE']:
#            if card in hdr.keys():
#                hdr.remove(card)
            
        while 'HISTORY' in hdr.keys():
            hdr.remove('HISTORY')

        hdr.add_history('This cube was produced by the PHANGS-ALMA pipeline.')
        hdr.add_history('PHANGS-ALMA Pipeline version ' + pipeVer)
        if vstring != '':
            hdr.add_history('This is part of data release '+vstring)

        hdr['OBJECT'] = dir_for_gal(gal)

        if vstring == '':
            hdr['ORIGIN'] = 'PHANGS-ALMA'
        else:
            hdr['ORIGIN'] = 'PHANGS-ALMA '+vstring

        datamax = np.nanmax(data)
        datamin = np.nanmin(data)
        hdr['DATAMAX'] = datamax
        hdr['DATAMIN'] = datamin

        # round the beam if it lies within the specified tolerance

        bmaj = hdr['BMAJ']
        bmin = hdr['BMIN']
        if bmaj != bmin:
            frac_dev = np.abs(bmaj-bmin)/bmaj
            if frac_dev <= roundbeam_tol:
                casalog.origin(casa_log_origin)
                casalog.post("Rounding beam.", "INFO", "phangs_cleanup_cubes")
                hdr['BMAJ'] = bmaj
                hdr['BMIN'] = bmaj
                hdr['BPA'] = 0.0
            else:
                casalog.origin(casa_log_origin)
                casalog.post("Beam too asymmetric to round.", "SEVERE", "phangs_cleanup_cubes")
                casalog.post("... fractional deviation: "+str(frac_dev), "SEVERE", "phangs_cleanup_cubes")
                
        hdu.writeto(outfile_fits, clobber=True)

#        return         # cdw: this was why it wasn't writing all to fits

# new: for rebinning and trimming pb files cdw Jan 9, 2020
#       added writing aligned TP file to fits Jan 22, 2020

def phangs_cleanup_pbcubes(
        gal=None, array=None, product=None, root_dir=None, 
        overwrite=False, min_pixeperbeam=3, roundbeam_tol=0.01, 
        vstring=''):
    """
    Clean up cubes.
    """

    log_file = casalog.logfile()

    if gal is None or array is None or product is None or \
            root_dir is None:
        casalog.origin(casa_log_origin)
        casalog.post("Missing required input.", "SEVERE", "phangs_cleanup_pbcubes")
        return

    for this_ext in ['pb']:   # edited cdw

        inroot = root_dir+'raw/'+gal+'_'+array+'_'+product
        root = root_dir+'process/'+gal+'_'+array+'_'+product
        rootfits = root_dir+'products/'+gal+'/'+gal+'_'+array+'_'+product
        infile = inroot+'.pb'
        outfile = root+'.pb.rebin'
        outfile_fits = rootfits+'_pb_rebin.fits'
    
        # Trim the cube to a smaller size and rebin as needed

# need to use the image cube to see check rebinning is needed for pb cube

        infile = root+'_flat_round.image'

        if os.path.isdir(infile) == False:
            casalog.origin(casa_log_origin)
            casalog.post("File does not exist: "+infile, "SEVERE", "phangs_cleanup_pbcubes")
            casalog.post("Returning.", "SEVERE", "phangs_cleanup_pbcubes")
            return

    # First, rebin if needed
        hdr = imhead(infile)
        if (hdr['axisunits'][0] != 'rad'):
            casalog.origin(casa_log_origin)
            casalog.post("Based on CASA experience. I expected units of radians.", "SEVERE", "phangs_cleanup_pbcubes")
            casalog.post("I did not find this. Returning. Adjust code or investigate file "+infile, "SEVERE", "phangs_cleanup_pbcubes")
            return

        pixel_as = abs(hdr['incr'][0]/np.pi*180.0*3600.)

        if (hdr['restoringbeam']['major']['unit'] != 'arcsec'):
            casalog.origin(casa_log_origin)
            casalog.post("Based on CASA experience. I expected units of arcseconds for the beam.", "SEVERE", "phangs_cleanup_pbcubes")
            casalog.post("I did not find this. Returning. Adjust code or investigate file "+infile, "SEVERE", "phangs_cleanup_pbcubes")
            return    
        bmaj = hdr['restoringbeam']['major']['value']    
    
        pix_per_beam = bmaj*1.0 / pixel_as*1.0
    
# may need a new version of this too? cdw
#        trim_cube(infile=infile, outfile=outfile, 
#                  overwrite=overwrite, inplace=False,
#                  min_pixperbeam=min_pixeperbeam)

        infile = inroot+'.pb'

        if os.path.isdir(infile) == False:
            casalog.origin(casa_log_origin)
            casalog.post("File does not exist: "+infile, "SEVERE", "phangs_cleanup_pbcubes")
            casalog.post("Returning.", "SEVERE", "phangs_cleanup_pbcubes")
            return

        if pix_per_beam > 6:
            imrebin(
                imagename=infile,
                outfile=outfile+'.temp',
                factor=[2,2,1],
                crop=True,
                dropdeg=True,
                overwrite=overwrite,
                )
        else:
            casalog.origin(casa_log_origin)
            os.system('cp -r '+infile+' '+outfile+'.temp'+" >> "+log_file+" 2>&1")

    # Figure out the extent of the image inside the cube
        myia = au.createCasaTool(iatool)
        myia.open(outfile+'.temp')
        mask = myia.getchunk(getmask=True)    
        myia.close()

        this_shape = mask.shape
    
        mask_spec_x = np.sum(np.sum(mask*1.0,axis=2),axis=1) > 0
        pad = 0
        xmin = np.max([0,np.min(np.where(mask_spec_x))-pad])
        xmax = np.min([np.max(np.where(mask_spec_x))+pad,mask.shape[0]-1])

        mask_spec_y = np.sum(np.sum(mask*1.0,axis=2),axis=0) > 0
        ymin = np.max([0,np.min(np.where(mask_spec_y))-pad])
        ymax = np.min([np.max(np.where(mask_spec_y))+pad,mask.shape[1]-1])

        mask_spec_z = np.sum(np.sum(mask*1.0,axis=0),axis=0) > 0
        zmin = np.max([0,np.min(np.where(mask_spec_z))-pad])
        zmax = np.min([np.max(np.where(mask_spec_z))+pad,mask.shape[2]-1])
    
        box_string = ''+str(xmin)+','+str(ymin)+','+str(xmax)+','+str(ymax)
        chan_string = ''+str(zmin)+'~'+str(zmax)

        casalog.origin(casa_log_origin)
        casalog.post("... ... ... box selection: "+box_string, "INFO", "phangs_cleanup_pbcubes")
        casalog.post("... ... ... channel selection: "+chan_string, "INFO", "phangs_cleanup_pbcubes")

        if overwrite:
            casalog.origin(casa_log_origin)
            os.system('rm -rf '+outfile+" >> "+log_file+" 2>&1")
            imsubimage(
                imagename=outfile+'.temp',
                outfile=outfile,
                box=box_string,
                chans=chan_string,
                )

        casalog.origin(casa_log_origin)
        os.system('rm -rf '+outfile+'.temp'+" >> "+log_file+" 2>&1")
    
        # Export to FITS
    
        exportfits(imagename=outfile, fitsimage=outfile_fits,
                   velocity=True, overwrite=True, dropstokes=True, 
                   dropdeg=True, bitpix=-32)
    
        # Clean up headers

        hdu = pyfits.open(outfile_fits)

        hdr = hdu[0].header
        data = hdu[0].data

#        for card in ['BLANK','DATE-OBS','OBSERVER','O_BLANK','O_BSCALE',
#                     'O_BZERO','OBSRA','OBSDEC','OBSGEO-X','OBSGEO-Y','OBSGEO-Z',
#                     'DISTANCE']:
#            if card in hdr.keys():
#                hdr.remove(card)
            
        while 'HISTORY' in hdr.keys():
            hdr.remove('HISTORY')

        hdr.add_history('This cube was produced by the PHANGS-ALMA pipeline.')
        hdr.add_history('PHANGS-ALMA Pipeline version ' + pipeVer)
        if vstring != '':
            hdr.add_history('This is part of data release '+vstring)

        hdr['OBJECT'] = dir_for_gal(gal)

        if vstring == '':
            hdr['ORIGIN'] = 'PHANGS-ALMA'
        else:
            hdr['ORIGIN'] = 'PHANGS-ALMA '+vstring

        datamax = np.nanmax(data)
        datamin = np.nanmin(data)
        hdr['DATAMAX'] = datamax
        hdr['DATAMIN'] = datamin

        hdu.writeto(outfile_fits, clobber=True)

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# LINEAR MOSAICKING ROUTINES
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

def phangs_common_res_for_mosaic(
    gal=None, array=None, product=None, root_dir=None, 
    overwrite=False, target_res=None):
    """
    Convolve multi-part cubes to a common res for mosaicking.
    """

    if gal is None or array is None or product is None or \
            root_dir is None:
        casalog.origin(casa_log_origin)
        casalog.post("Missing required input.", "SEVERE", "phangs_common_res_for_mosaic")
        return    

    # Look up parts
    this_mosaic_key = mosaic_key()
    if (gal in this_mosaic_key.keys()) == False:
        casalog.origin(casa_log_origin)
        casalog.post("Galaxy "+gal+" not in mosaic key.", "SEVERE", "phangs_common_res_for_mosaic")
        return
    parts = this_mosaic_key[gal]

    for this_ext in ['flat_round', 'pbcorr_round']:           

        infile_list = []
        outfile_list = []
        input_dir = root_dir+'process/'
        for this_part in parts:
            infile = input_dir+this_part+'_'+array+'_'+product+'_'+this_ext+'.image'
            infile_list.append(infile)
            outfile = input_dir+this_part+'_'+array+'_'+product+'_'+this_ext+'_tomerge.image'
            outfile_list.append(outfile)

        common_res_for_mosaic(
            infile_list=infile_list,
            outfile_list=outfile_list,
            overwrite=overwrite, target_res=target_res)

def common_res_for_mosaic(
    infile_list = None, outfile_list = None,
    overwrite=False, target_res=None):
    """
    Convolve multi-part cubes to a common res for mosaicking.
    """

    if (infile_list is None) or \
            (outfile_list is None):
        casalog.origin(casa_log_origin)
        casalog.post("Missing required input.", "SEVERE", "common_res_for_mosaic")
        return    
    
    if len(infile_list) != len(outfile_list):
        casalog.origin(casa_log_origin)
        casalog.post("Mismatch in input lists.", "SEVERE", "common_res_for_mosaic")
        return    

    for this_file in infile_list:
        if os.path.isdir(this_file) == False:
            casalog.origin(casa_log_origin)
            casalog.post("File not found "+this_file, "SEVERE", "common_res_for_mosaic")
            return
    
    # Figure out target resolution if it is not supplied

    if target_res is None:
        casalog.origin(casa_log_origin)
        casalog.post("Calculating target resolution ... ", "INFO", "common_res_for_mosaic")

        bmaj_list = []
        pix_list = []

        for infile in infile_list:
            casalog.origin(casa_log_origin)
            casalog.post("Checking "+infile, "INFO", "common_res_for_mosaic")

            hdr = imhead(infile)

            if (hdr['axisunits'][0] != 'rad'):
                casalog.origin(casa_log_origin)
                casalog.post("Based on CASA experience. I expected units of radians.", "SEVERE", "common_res_for_mosaic")
                casalog.post("I did not find this. Returning. Adjust code or investigate file "+infile, "SEVERE", "common_res_for_mosaic")
                return
            this_pixel = abs(hdr['incr'][0]/np.pi*180.0*3600.)

            if (hdr['restoringbeam']['major']['unit'] != 'arcsec'):
                casalog.origin(casa_log_origin)
                casalog.post("Based on CASA experience. I expected units of arcseconds for the beam.", "SEVERE", "common_res_for_mosaic")
                casalog.post("I did not find this. Returning. Adjust code or investigate file "+infile, "SEVERE", "common_res_for_mosaic")
                return
            this_bmaj = hdr['restoringbeam']['major']['value']

            bmaj_list.append(this_bmaj)
            pix_list.append(this_pixel)
        
        max_bmaj = np.max(bmaj_list)
        max_pix = np.max(pix_list)
        target_bmaj = np.sqrt((max_bmaj)**2+(2.0*max_pix)**2)
    else:
        target_bmaj = force_beam

    for ii in range(len(infile_list)):
        this_infile = infile_list[ii]
        this_outfile = outfile_list[ii]
        casalog.origin(casa_log_origin)
        casalog.post("Convolving "+this_infile+' to '+this_outfile, "INFO", "common_res_for_mosaic")
        
        imsmooth(imagename=this_infile,
             outfile=this_outfile,
             targetres=True,
             major=str(target_bmaj)+'arcsec',
             minor=str(target_bmaj)+'arcsec',
             pa='0.0deg',
             overwrite=overwrite
             )

    return target_bmaj

def build_common_header(
    infile_list = None, 
    ra_ctr = None, dec_ctr = None,
    delta_ra = None, delta_dec = None):
    """
    Build a target header to be used as a template by imregrid.
    """

    if infile_list is None:
        casalog.origin(casa_log_origin)
        casalog.post("Missing required input.", "SEVERE", "build_common_header")
        return    

    # Logic to determine tuning parameters here if they aren't passed.

    if os.path.isdir(infile_list[0]) == False:
        casalog.origin(casa_log_origin)
        casalog.post("File not found "+infile_list[0], "SEVERE", "build_common_header")
        casalog.post("Returning.", "SEVERE", "build_common_header")
        return None
    target_hdr = imregrid(infile_list[0], template='get')
    
    # N.B. Could put a lot of general logic here, but we are usually
    # working in a pretty specific case.

    if (target_hdr['csys']['direction0']['units'][0] != 'rad') or \
            (target_hdr['csys']['direction0']['units'][1] != 'rad'):
        casalog.origin(casa_log_origin)
        casalog.post("Based on CASA experience. I expected pixel units of radians.", "SEVERE", "build_common_header")
        casalog.post("I did not find this. Returning. Adjust code or investigate file "+infile_list[0], "SEVERE", "build_common_header")
        return

    # Put in our target values for the center after converting to radians
    ra_ctr_in_rad = ra_ctr * np.pi / 180.
    dec_ctr_in_rad = dec_ctr * np.pi / 180.

    target_hdr['csys']['direction0']['crval'][0] = ra_ctr_in_rad
    target_hdr['csys']['direction0']['crval'][1] = dec_ctr_in_rad

    # Adjust the size and central pixel
    
    ra_pix_in_as = np.abs(target_hdr['csys']['direction0']['cdelt'][0]*180./np.pi*3600.)
    dec_pix_in_as = np.abs(target_hdr['csys']['direction0']['cdelt'][1]*180./np.pi*3600.)
    ra_axis_size = np.ceil(delta_ra / ra_pix_in_as)
    new_ra_ctr_pix = ra_axis_size/2.0
    dec_axis_size = np.ceil(delta_dec / dec_pix_in_as)
    new_dec_ctr_pix = dec_axis_size/2.0
    
    target_hdr['csys']['direction0']['crpix'][0] = new_ra_ctr_pix
    target_hdr['csys']['direction0']['crpix'][1] = new_dec_ctr_pix
    
    if ra_axis_size > 1e4 or dec_axis_size > 1e4:
        casalog.origin(casa_log_origin)
        casalog.post("This is a very big image you plan to create.", "WARN", "build_common_header")
        casalog.post(ra_axis_size, " x ", dec_axis_size, "WARN", "build_common_header")
        test = raw_input("Continue? Hit [y] if so.")
        if test != 'y':
            return

    target_hdr['shap'][0] = int(ra_axis_size)
    target_hdr['shap'][1] = int(dec_axis_size)
    
    return(target_hdr)

def align_for_mosaic(
    infile_list = None, outfile_list = None,
    overwrite=False, target_hdr=None):
    """
    Align a list of files to a target coordinate system.
    """

    if infile_list is None or outfile_list is None or \
            target_hdr is None:
        casalog.origin(casa_log_origin)
        casalog.post("Missing required input.", "SEVERE", "build_common_header")
        return    

    for ii in range(len(infile_list)):
        this_infile = infile_list[ii]
        this_outfile = outfile_list[ii]        

        if os.path.isdir(this_infile) == False:
            casalog.origin(casa_log_origin)
            casalog.post("File "+this_infile+" not found. Continuing.", "WARN", "build_common_header")
            continue

        imregrid(imagename=this_infile,
                 template=target_hdr,
                 output=this_outfile,
                 asvelocity=True,
                 axes=[-1],
                 interpolation='cubic',
                 overwrite=overwrite)

    return

def phangs_align_for_mosaic(
    gal=None, array=None, product=None, root_dir=None, 
    overwrite=False, target_hdr=None):
    """
    Convolve multi-part cubes to a common res for mosaicking.
    """

    if gal is None or array is None or product is None or \
            root_dir is None:
        casalog.origin(casa_log_origin)
        casalog.post("Missing required input.", "SEVERE", "phangs_align_for_mosaic")
        return    
    
    # Look up parts

    this_mosaic_key = mosaic_key()
    if (gal in this_mosaic_key.keys()) == False:
        casalog.origin(casa_log_origin)
        casalog.post("Galaxy "+gal+" not in mosaic key.", "SEVERE", "phangs_align_for_mosaic")
        return
    parts = this_mosaic_key[gal]

    # Read the key that defines the extent and center of the mosaic
    # manually. We will use this to figure out the target header.

    multipart_key = read_multipart_key()
    if (gal in multipart_key.keys()) == False:
        casalog.origin(casa_log_origin)
        casalog.post("Galaxy "+gal+" not in multipart key.", "SEVERE", "phangs_align_for_mosaic")
        casalog.post("... working on a general header construction algorithm.", "SEVERE", "phangs_align_for_mosaic")
        casalog.post("... for now, go enter a center and size into the multipart key:", "SEVERE", "phangs_align_for_mosaic")
        casalog.post("... multipart_fields.txt ", "SEVERE", "phangs_align_for_mosaic")
        return
    this_ra_ctr = multipart_key[gal]['ra_ctr_deg']
    this_dec_ctr = multipart_key[gal]['dec_ctr_deg']
    this_delta_ra = multipart_key[gal]['delta_ra_as']
    this_delta_dec = multipart_key[gal]['delta_dec_as']

    for this_ext in ['flat_round', 'pbcorr_round']:           

        # Align data

        infile_list = []
        outfile_list = []
        input_dir = root_dir+'process/'
        output_dir = root_dir+'process/'
        for this_part in parts:
            infile = input_dir+this_part+'_'+array+'_'+product+'_'+this_ext+'_tomerge.image'
            infile_list.append(infile)
            outfile = output_dir+this_part+'_'+array+'_'+product+'_'+this_ext+'_onmergegrid.image'
            outfile_list.append(outfile)

        # Work out the target header if it does not exist yet.

        if target_hdr is None:
            target_hdr = \
                build_common_header(
                infile_list = infile_list, 
                ra_ctr = this_ra_ctr, dec_ctr = this_dec_ctr,
                delta_ra = this_delta_ra, delta_dec = this_delta_dec)
            
        align_for_mosaic(
            infile_list = infile_list, 
            outfile_list = outfile_list,
            overwrite=overwrite, target_hdr=target_hdr)

        # Align primary beam images, too, to use as weight.

        infile_list = []
        outfile_list = []
        input_dir = root_dir+'raw/'
        output_dir = root_dir+'process/'
        for this_part in parts:
            input_array = array
            if array == '7m+tp':
                input_array = '7m'
            if array == '12m+7m+tp':
                input_array = '12m+7m'
            infile = input_dir+this_part+'_'+input_array+'_'+product+'.pb'
            infile_list.append(infile)
            outfile = output_dir+this_part+'_'+array+'_'+product+'_'+this_ext+'_mergeweight.image'
            outfile_list.append(outfile)

        align_for_mosaic(
            infile_list = infile_list, 
            outfile_list = outfile_list,
            overwrite=overwrite, target_hdr=target_hdr)

def mosaic_aligned_data(
    infile_list = None, weightfile_list = None,
    outfile = None, overwrite=False):
    """
    Combine a list of aligned data with primary-beam (i.e., inverse
    noise) weights using simple linear mosaicking.
    """

    log_file = casalog.logfile()

    if infile_list is None or weightfile_list is None or \
            outfile is None:
        casalog.origin(casa_log_origin)
        casalog.post("Missing required input.", "SEVERE", "mosaic_aligned_data")
        return    

    sum_file = outfile+'.sum'
    weight_file = outfile+'.weight'

    if (os.path.isdir(outfile) or \
            os.path.isdir(sum_file) or \
            os.path.isdir(weight_file)) and \
            (overwrite == False):
        casalog.origin(casa_log_origin)
        casalog.post("Output file present and overwrite off.", "SEVERE", "mosaic_aligned_data")
        casalog.post("Returning.", "SEVERE", "mosaic_aligned_data")
        return

    if overwrite:
        casalog.origin(casa_log_origin)
        os.system('rm -rf '+outfile+'.temp'+" >> "+log_file+" 2>&1")
        os.system('rm -rf '+outfile+" >> "+log_file+" 2>&1")
        os.system('rm -rf '+sum_file+" >> "+log_file+" 2>&1")
        os.system('rm -rf '+weight_file+" >> "+log_file+" 2>&1")
        os.system('rm -rf '+outfile+'.mask'+" >> "+log_file+" 2>&1")

    imlist = infile_list[:]
    imlist.extend(weightfile_list)
    n_image = len(infile_list)
    lel_exp_sum = ''
    lel_exp_weight = ''
    first = True
    for ii in range(n_image):
        this_im = 'IM'+str(ii)
        this_wt = 'IM'+str(ii+n_image)
        this_lel_sum = '('+this_im+'*'+this_wt+'*'+this_wt+')'
        this_lel_weight = '('+this_wt+'*'+this_wt+')'
        if first:
            lel_exp_sum += this_lel_sum
            lel_exp_weight += this_lel_weight
            first=False
        else:
            lel_exp_sum += '+'+this_lel_sum
            lel_exp_weight += '+'+this_lel_weight

    immath(imagename = imlist, mode='evalexpr',
           expr=lel_exp_sum, outfile=sum_file,
           imagemd = imlist[0])
    
    myia = au.createCasaTool(iatool)
    myia.open(sum_file)
    myia.set(pixelmask=1)
    myia.close()

    immath(imagename = imlist, mode='evalexpr',
           expr=lel_exp_weight, outfile=weight_file)
    myia.open(weight_file)
    myia.set(pixelmask=1)
    myia.close()

    immath(imagename = [sum_file, weight_file], mode='evalexpr',
           expr='iif(IM1 > 0.0, IM0/IM1, 0.0)', outfile=outfile+'.temp',
           imagemd = sum_file)

    immath(imagename = weight_file, mode='evalexpr',
           expr='iif(IM0 > 0.0, 1.0, 0.0)', outfile=outfile+'.mask')

    imsubimage(imagename=outfile+'.temp', outfile=outfile,
               mask='"'+outfile+'.mask"', dropdeg=True)



    return

def phangs_mosaic_data(
    gal=None, array=None, product=None, root_dir=None, 
    overwrite=False):
    """
    Linearly mosaic multipart cubes.
    """

    if gal is None or array is None or product is None or \
            root_dir is None:
        casalog.origin(casa_log_origin)
        casalog.post("Missing required input.", "SEVERE", "phangs_mosaic_data")
        return    
    
    # Look up parts

    this_mosaic_key = mosaic_key()
    if (gal in this_mosaic_key.keys()) == False:
        casalog.origin(casa_log_origin)
        casalog.post("Galaxy "+gal+" not in mosaic key.", "SEVERE", "phangs_mosaic_data")
        return
    parts = this_mosaic_key[gal]

    for this_ext in ['flat_round', 'pbcorr_round']:           

        infile_list = []
        wtfile_list = []
        input_dir = root_dir+'process/'
        output_dir = root_dir+'process/'
        for this_part in parts:
            infile = input_dir+this_part+'_'+array+'_'+product+'_'+this_ext+'_onmergegrid.image'
            wtfile = output_dir+this_part+'_'+array+'_'+product+'_'+this_ext+'_mergeweight.image'
            if (os.path.isdir(infile) == False):
                casalog.origin(casa_log_origin)
                casalog.post("Missing "+infile, "WARN", "phangs_mosaic_data")
                casalog.post("Skipping.", "WARN", "phangs_mosaic_data")
                continue
            if (os.path.isdir(wtfile) == False):
                casalog.origin(casa_log_origin)
                casalog.post("Missing "+wtfile, "WARN", "phangs_mosaic_data")
                casalog.post("Skipping.", "WARN", "phangs_mosaic_data")
                continue
            infile_list.append(infile)
            wtfile_list.append(wtfile)

        if len(infile_list) == 0:
            casalog.origin(casa_log_origin)
            casalog.post("No files to include in the mosaic. Returning.", "SEVERE", "phangs_mosaic_data")
            return

        outfile = output_dir+gal+'_'+array+'_'+product+'_'+this_ext+'.image'
    
        mosaic_aligned_data(
            infile_list = infile_list, weightfile_list = wtfile_list,
            outfile = outfile, overwrite=overwrite)
