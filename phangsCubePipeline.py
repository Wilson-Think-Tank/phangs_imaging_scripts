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
from concat import concat
from exportfits import exportfits
from feather import feather
from flagdata import flagdata
from imhead import imhead
from immath import immath
from impbcor import impbcor
from importfits import importfits
from imrebin import imrebin
from imregrid import imregrid
from imsmooth import imsmooth
from imstat import imstat
from imsubimage import imsubimage
from makemask import makemask
from mstransform import mstransform
from split import split
from statwt import statwt
from tclean import tclean
from uvcontsub import uvcontsub
from visstat import visstat
from pipelineVersion import version as pipeVer
# Physical constants
sol_kms = 2.9979246e5

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# DIRECTORY STRUCTURE AND FILE MANAGEMENT
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

def rebuild_directories(outroot_dir=None):
    """
    Reset and rebuild the directory structure.
    """

    log_file = casalog.logfile()
    
    if outroot_dir is None:
        casalog.post("Specify a root directory.", "SEVERE", "")
        return

    check_string = raw_input("Reseting directory structure in "+\
                                 outroot_dir+". Type 'Yes' to confirm.")
    if check_string != "Yes":
        casalog.post("Aborting", "SEVERE", "")
        return

    if os.path.isdir(outroot_dir) == False:
        casalog.post("Got make "+outroot_dir+" manually. Then I will make the subdirectories.", "INFO", "")

    for subdir in ['raw/','process/','products/','feather/','delivery/']:
        os.system('rm -rf '+outroot_dir+'/'+subdir+" > "+log_file+" 2>&1")
        os.system('mkdir '+outroot_dir+'/'+subdir+" > "+log_file+" 2>&1")

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
        casalog.post("Missing required input.", "SEVERE", "")
        return
    
    input_dir = dir_for_gal(gal)
    in_cube_name = input_dir+gal+'_'+array+'_'+product+'.fits'
    in_pb_name = input_dir+gal+'_'+array+'_'+product+'_pb.fits'

    out_dir = root_dir+'raw/'
    out_cube_name = out_dir+gal+'_'+array+'_'+product+'.image'
    out_pb_name = out_dir+gal+'_'+array+'_'+product+'.pb'
    
    casalog.post("... importing data for "+in_cube_name, "INFO", "")

    if os.path.isfile(in_cube_name):
        importfits(fitsimage=in_cube_name, imagename=out_cube_name,
                   zeroblanks=True, overwrite=overwrite)
    else:
        casalog.post("File not found "+in_cube_name, "SEVERE", "")

    if os.path.isfile(in_pb_name):
        importfits(fitsimage=in_pb_name, imagename=out_pb_name,
                   zeroblanks=True, overwrite=overwrite)
    else:
        casalog.post("Directory not found "+in_pb_name, "SEVERE", "")

def phangs_stage_singledish(
    gal=None, product=None, root_dir=None, 
    overwrite=False):
    """
    Copy the single dish data for further processing
    """

    log_file = casalog.logfile()

    if gal is None or product is None or \
            root_dir is None:
        casalog.post("Missing required input.", "SEVERE", "")
        return
    
    sdk = read_singledish_key()
    if (gal in sdk.keys()) == False:
        casalog.post(gal+" not found in single dish key.", "SEVERE", "")
        return
    
    this_key = sdk[gal]
    if (product in this_key.keys()) == False:
        casalog.post(product+" not found in single dish key for "+gal, "SEVERE", "")
        return
    
    sdfile_in = this_key[product]
    
    sdfile_out = root_dir+'raw/'+gal+'_tp_'+product+'.image'    

    casalog.post("... importing single dish data for "+sdfile_in, "INFO", "")

    importfits(fitsimage=sdfile_in, imagename=sdfile_out+'.temp',
               zeroblanks=True, overwrite=overwrite)

    if overwrite:
        os.system('rm -rf '+sdfile_out+" > "+log_file+" 2>&1")
    imsubimage(imagename=sdfile_out+'.temp', outfile=sdfile_out,
               dropdeg=True)
    os.system('rm -rf '+sdfile_out+'.temp'+" > "+log_file+" 2>&1")

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

    casalog.post("", "INFO", "")
    casalog.post("... producing a primary beam corrected image for "+input_cube_name, "INFO", "")
    casalog.post("", "INFO", "")
    
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
        casalog.post("Missing required input.", "SEVERE", "")
        return

    if os.path.isdir(infile) == False:
        casalog.post("Input file missing - "+infile, "SEVERE", "")
        return

    if os.path.isdir(pbfile) == False:
        casalog.post("Primary beam file missing - "+pbfile, "SEVERE", "")
        return

    if overwrite:
        os.system('rm -rf '+outfile+" > "+log_file+" 2>&1")

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
        casalog.post("Missing required input.", "SEVERE", "")
        return
    
    input_dir = root_dir+'process/'
    input_cube_name = input_dir+gal+'_'+array+'_'+product+'_pbcorr.image'
    output_dir = root_dir+'process/'
    output_cube_name = output_dir+gal+'_'+array+'_'+product+'_pbcorr_round.image'

    casalog.post("", "INFO", "")
    casalog.post("... convolving to a round beam for "+input_cube_name, "INFO", "")
    casalog.post("", "INFO", "")

    round_beam = \
        convolve_to_round_beam(
        infile=input_cube_name,
        outfile=output_cube_name,
        force_beam=force_beam,
        overwrite=overwrite)

    casalog.post("", "INFO", "")
    casalog.post("... found beam of "+str(round_beam)+" arcsec. Forcing flat cube to this.", "INFO", "")
    casalog.post("", "INFO", "")

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
        casalog.post("Missing required input.", "SEVERE", "")
        return

    if os.path.isdir(infile) == False:
        casalog.post("Input file missing - "+infile, "SEVERE", "")
        return    

    if force_beam is None:
        hdr = imhead(infile)

        if (hdr['axisunits'][0] != 'rad'):
            casalog.post("Based on CASA experience. I expected units of radians.", "SEVERE", "")
            casalog.post("I did not find this. Returning. Adjust code or investigate file "+infile, "SEVERE", "")
            return
        pixel_as = abs(hdr['incr'][0]/np.pi*180.0*3600.)

        if (hdr['restoringbeam']['major']['unit'] != 'arcsec'):
            casalog.post("Based on CASA experience. I expected units of arcseconds for the beam.", "SEVERE", "")
            casalog.post("I did not find this. Returning. Adjust code or investigate file "+infile, "SEVERE", "")
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

def prep_for_feather(
    gal=None, array=None, product=None, root_dir=None, 
    overwrite=False):
    """
    Prepare the single dish data for feathering
    """

    log_file = casalog.logfile()
    
    if gal is None or array is None or product is None or \
            root_dir is None:
        casalog.post("Missing required input.", "SEVERE", "")
        return    

    sdfile_in = root_dir+'raw/'+gal+'_tp_'+product+'.image'
    interf_in = root_dir+'process/'+gal+'_'+array+'_'+product+'_flat_round.image'
    pbfile_name = root_dir+'raw/'+gal+'_'+array+'_'+product+'.pb'    

    if (os.path.isdir(sdfile_in) == False):
        casalog.post("Single dish file not found: "+sdfile_in, "SEVERE", "")
        return

    if (os.path.isdir(interf_in) == False):
        casalog.post("Interferometric file not found: "+interf_in, "SEVERE", "")
        return

    if (os.path.isdir(pbfile_name) == False):
        casalog.post("Primary beam file not found: "+pbfile_name, "SEREVE", "")
        return

    # Align the relevant TP data to the product.
    sdfile_out = root_dir+'process/'+gal+'_tp_'+product+'_align_'+array+'.image'
    imregrid(imagename=sdfile_in,
             template=interf_in,
             output=sdfile_out,
             asvelocity=True,
             axes=[-1],
             interpolation='cubic',
             overwrite=overwrite)

    # Taper the TP data by the primary beam.
    taperfile_out = root_dir+'process/'+gal+'_tp_'+product+'_taper_'+array+'.image'
    if overwrite:
        os.system('rm -rf '+taperfile_out+" > "+log_file+" 2>&1")

    impbcor(imagename=sdfile_out, 
            pbimage=pbfile_name, 
            outfile=taperfile_out, 
            mode='multiply',
            stokes='I')

    return

def phangs_feather_data(
    gal=None, array=None, product=None, root_dir=None, 
    cutoff=-1,overwrite=False):
    """
    Feather the interferometric and total power data.
    """

    log_file = casalog.logfile()

    if gal is None or array is None or product is None or \
            root_dir is None:
        casalog.post("Missing required input.", "SEVERE", "")
        return    

    sdfile_in = root_dir+'process/'+gal+'_tp_'+product+'_taper_'+array+'.image'
    interf_in = root_dir+'process/'+gal+'_'+array+'_'+product+'_flat_round.image'
    pbfile_name = root_dir+'raw/'+gal+'_'+array+'_'+product+'.pb' 

    if (os.path.isdir(sdfile_in) == False):
        casalog.post("Single dish file not found: "+sdfile_in, "SEVERE", "")
        return
        
    if (os.path.isdir(interf_in) == False):
        casalog.post("Interferometric file not found: "+interf_in, "SEVERE", "")
        return

    if (os.path.isdir(pbfile_name) == False):
        casalog.post("Primary beam file not found: "+pbfile_name, "SEVERE", "")
        return

    # Feather the inteferometric and "flat" TP data.
    outfile_name = root_dir+'process/'+gal+'_'+array+'+tp_'+product+ \
        '_flat_round.image'

    if overwrite:        
        os.system('rm -rf '+outfile_name+" > "+log_file+" 2>&1")
    os.system('rm -rf '+outfile_name+'.temp'+" > "+log_file+" 2>&1")
    feather(imagename=outfile_name+'.temp',
            highres=interf_in,
            lowres=sdfile_in,
            sdfactor=1.0,
            lowpassfiltersd=False)
    imsubimage(imagename=outfile_name+'.temp', outfile=outfile_name,
               dropdeg=True)
    os.system('rm -rf '+outfile_name+'.temp'+" > "+log_file+" 2>&1")
    infile_name = outfile_name

    # Primary beam correct the feathered data.
    outfile_name = root_dir+'process/'+gal+'_'+array+'+tp_'+product+ \
        '_pbcorr_round.image'
    
    if overwrite:        
        os.system('rm -rf '+outfile_name+" > "+log_file+" 2>&1")

    casalog.post(infile_name, "INFO", "")
    casalog.post(pbfile_name "INFO", "")
    impbcor(imagename=infile_name,
            pbimage=pbfile_name, 
            outfile=outfile_name, 
            mode='divide', cutoff=cutoff)

def chris_feather_data(
    gal=None, array=None, product=None, root_dir=None, 
    cutoff=-1,overwrite=False):
    """
    Feather the pbcorr'd interferometric and aligned total power data
    """

    log_file = casalog.logfile()

    if gal is None or array is None or product is None or \
            root_dir is None:
        casalog.post("Missing required input.", "SEVERE", "")
        return    

    sdfile_in = root_dir+'process/'+gal+'_tp_'+product+'_align_'+array+'.image'
    interf_in = root_dir+'process/'+gal+'_'+array+'_'+product+'_pbcorr_round.image'
    pbfile_name = root_dir+'raw/'+gal+'_'+array+'_'+product+'.pb' 

    if (os.path.isdir(sdfile_in) == False):
        casalog.post("Single dish file not found: "+sdfile_in, "SEVERE", "")
        return
        
    if (os.path.isdir(interf_in) == False):
        casalog.post("Interferometric file not found: "+interf_in, "SEVERE", "")
        return

    if (os.path.isdir(pbfile_name) == False):
        casalog.post("Primary beam file not found: "+pbfile_name, "SEVERE", "")
        return

    # Feather the "pbcorr"d inteferometric and "align"d TP data.
    outfile_name = root_dir+'process/'+gal+'_'+array+'+tp_'+product+ \
        '_pbcorr_round.image'

    if overwrite:        
        os.system('rm -rf '+outfile_name+" > "+log_file+" 2>&1")
    os.system('rm -rf '+outfile_name+'.temp'+" > "+log_file+" 2>&1")
    feather(imagename=outfile_name+'.temp',
            highres=interf_in,
            lowres=sdfile_in,
            sdfactor=1.0,
            lowpassfiltersd=False)
    imsubimage(imagename=outfile_name+'.temp', outfile=outfile_name,
               dropdeg=True)
    os.system('rm -rf '+outfile_name+'.temp'+" > "+log_file+" 2>&1")
    infile_name = outfile_name

# apply primary beam to flatten the feathered data.
    outfile_name = root_dir+'process/'+gal+'_'+array+'+tp_'+product+ \
        '_flat_round.image'
    
    if overwrite:        
        os.system('rm -rf '+outfile_name+" > "+log_file+" 2>&1")

    casalog.post(infile_name, "INFO", "")
    casalog.post(pbfile_name, "INFO", "")
    impbcor(imagename=infile_name,
            pbimage=pbfile_name, 
            outfile=outfile_name, 
            mode='multiply', cutoff=-1.0)
#            mode='divide', cutoff=cutoff)

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
        casalog.post("Missing required input.", "SEVERE", "")
        return
    
    if os.path.isdir(infile) == False:
        csalog.post("Input file not found: "+infile, "SEVERE", "")
        return
    
    if inplace == False:
        if overwrite:
            os.system('rm -rf '+outfile+" > "+log_file+" 2>&1")
        
        if os.path.isdir(outfile):
            casalog.post("Output file already present: "+outfile, "SEVERE", "")
            return

        os.system('cp -r '+infile+' '+outfile+" > "+log_file+" 2>&1")
        target_file = outfile
    else:
        target_file = infile

    hdr = imhead(target_file, mode='list')
    unit = hdr['bunit']
    if unit != 'Jy/beam':
        casalog.post("Unit is not Jy/beam. Returning.", "SEVERE", "")
        return

    #restfreq_hz = hdr['restfreq'][0]

    if hdr['cunit3'] != 'Hz':
        casalog.post("I expected frequency as the third axis but did not find it.", "SEVERE", "")
        casalog.post("Returning.", "SEVERE", "")
        return
    
    crpix3 = hdr['crpix3']
    cdelt3 = hdr['cdelt3']
    crval3 = hdr['crval3']
    naxis3 = hdr['shape'][2]
    faxis_hz = (np.arange(naxis3)+1.-crpix3)*cdelt3+crval3
    freq_hz = np.mean(faxis_hz)
    
    bmaj_unit = hdr['beammajor']['unit']
    if bmaj_unit != 'arcsec':
        casalog.post("Beam unit is not arcsec, which I expected. Returning.", "SEVERE", "")
        casalog.post("Unit instead is "+bmaj_unit, "SEVERE", "")
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
        casalog.post("Missing required input.", "SEVERE", "")
        return
    
    if os.path.isdir(infile) == False:
        casalog.post("Input file not found: "+infile, "SEVERE", "")
        return

    # First, rebin if needed
    hdr = imhead(infile)
    if (hdr['axisunits'][0] != 'rad'):
        casalog.post("Based on CASA experience. I expected units of radians.", "SEVERE", "")
        casalog.post("I did not find this. Returning. Adjust code or investigate file "+infile, "SEVERE", "")
        return

    pixel_as = abs(hdr['incr'][0]/np.pi*180.0*3600.)

    if (hdr['restoringbeam']['major']['unit'] != 'arcsec'):
        casalog.post("Based on CASA experience. I expected units of arcseconds for the beam.", "SEVERE", "")
        casalog.post("I did not find this. Returning. Adjust code or investigate file "+infile, "SEVERE", "")
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
        os.system('cp -r '+infile+' '+outfile+'.temp'+" > "+log_file+" 2>&1")

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

    casalog.post("... ... ... box selection: "+box_string, "INFO", "")
    casalog.post("... ... ... channel selection: "+chan_string, "INFO", "")

    if overwrite:
        os.system('rm -rf '+outfile+" > "+log_file+" 2>&1")
        imsubimage(
        imagename=outfile+'.temp',
        outfile=outfile,
        box=box_string,
        chans=chan_string,
        )
    
    os.system('rm -rf '+outfile+'.temp'+" > "+log_file+" 2>&1")
    
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
        casalog.post("Missing required input.", "SEVERE", "")
        return

#    If necessary, create the galaxy subdirectory in 'products/'  : cdw
    
    if os.path.isdir(root_dir+'products/'+gal+'/') == False:
        casalog.post("Making product sudirectory for  "+gal, "INFO", "")
        os.system('mkdir '+root_dir+'products/'+gal+'/'+" > "+log_file+" 2>&1")

    for this_ext in ['flat', 'pbcorr']:

        root = root_dir+'process/'+gal+'_'+array+'_'+product+'_'+this_ext
        rootfits = root_dir+'products/'+gal+'/'+gal+'_'+array+'_'+product+'_'+this_ext
        infile = root+'_round.image'
        outfile = root+'_round_k.image'
        outfile_fits = rootfits+'_round_k.fits'
    
        if os.path.isdir(infile) == False:
            casalog.post("File does not exist: "+infile, "SEVERE", "")
            casalog.post("Returning.", "SEVERE", "")
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
                casalog.post("Rounding beam.", "INFO", "")
                hdr['BMAJ'] = bmaj
                hdr['BMIN'] = bmaj
                hdr['BPA'] = 0.0
            else:
                casalog.post("Beam too asymmetric to round.", "SEVERE", "")
                casalog.post("... fractional deviation: "+str(frac_dev), "SEVERE", "")
                
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
        casalog.post("Missing required input.", "SEVERE", "")
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
            casalog.post("File does not exist: "+infile, "SEVERE", "")
            casalog.post("Returning.", "SEVERE", "")
            return

    # First, rebin if needed
        hdr = imhead(infile)
        if (hdr['axisunits'][0] != 'rad'):
            casalog.post("Based on CASA experience. I expected units of radians.", "SEVERE", "")
            casalog.post("I did not find this. Returning. Adjust code or investigate file "+infile, "SEVERE", "")
            return

        pixel_as = abs(hdr['incr'][0]/np.pi*180.0*3600.)

        if (hdr['restoringbeam']['major']['unit'] != 'arcsec'):
            casalog.post("Based on CASA experience. I expected units of arcseconds for the beam.", "SEVERE", "")
            casalog.post("I did not find this. Returning. Adjust code or investigate file "+infile, "SEVERE", "")
            return    
        bmaj = hdr['restoringbeam']['major']['value']    
    
        pix_per_beam = bmaj*1.0 / pixel_as*1.0
    
# may need a new version of this too? cdw
#        trim_cube(infile=infile, outfile=outfile, 
#                  overwrite=overwrite, inplace=False,
#                  min_pixperbeam=min_pixeperbeam)

        infile = inroot+'.pb'

        if os.path.isdir(infile) == False:
            casalog.post("File does not exist: "+infile, "SEVERE", "")
            casalog.post("Returning.", "SEVERE", "")
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
            os.system('cp -r '+infile+' '+outfile+'.temp'+" > "+log_file+" 2>&1")

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

        casalog.post("... ... ... box selection: "+box_string, "INFO", "")
        casalog.post("... ... ... channel selection: "+chan_string, "INFO", "")

        if overwrite:
            os.system('rm -rf '+outfile+" > "+log_file+" 2>&1")
            imsubimage(
                imagename=outfile+'.temp',
                outfile=outfile,
                box=box_string,
                chans=chan_string,
                )
    
        os.system('rm -rf '+outfile+'.temp'+" > "+log_file+" 2>&1")
    
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
        casalog.post("Missing required input.", "SEVERE", "")
        return    

    # Look up parts
    this_mosaic_key = mosaic_key()
    if (gal in this_mosaic_key.keys()) == False:
        casalog.post("Galaxy "+gal+" not in mosaic key.", "SEVERE", "")
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
        casalog.post("Missing required input.", "SEVERE", "")
        return    
    
    if len(infile_list) != len(outfile_list):
        casalog.post("Mismatch in input lists.", "SEVERE", "")
        return    

    for this_file in infile_list:
        if os.path.isdir(this_file) == False:
            casalog.post("File not found "+this_file, "SEVERE", "")
            return
    
    # Figure out target resolution if it is not supplied

    if target_res is None:
        casalog.post("Calculating target resolution ... ", "INFO", "")

        bmaj_list = []
        pix_list = []

        for infile in infile_list:
            casalog.post("Checking "+infile, "INFO", "")

            hdr = imhead(infile)

            if (hdr['axisunits'][0] != 'rad'):
                casalog.post("Based on CASA experience. I expected units of radians.", "SEVERE", "")
                casalog.post("I did not find this. Returning. Adjust code or investigate file "+infile, "SEVERE", "")
                return
            this_pixel = abs(hdr['incr'][0]/np.pi*180.0*3600.)

            if (hdr['restoringbeam']['major']['unit'] != 'arcsec'):
                casalog.post("Based on CASA experience. I expected units of arcseconds for the beam.", "SEVERE", "")
                casalog.post("I did not find this. Returning. Adjust code or investigate file "+infile, "SEVERE", "")
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
        casalog.post("Convolving "+this_infile+' to '+this_outfile, "INFO", "")
        
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
        casalog.post("Missing required input.", "SEVERE", "")
        return    

    # Logic to determine tuning parameters here if they aren't passed.

    if os.path.isdir(infile_list[0]) == False:
        casalog.post("File not found "+infile_list[0], "SEVERE", "")
        casalog.post("Returning.", "SEVERE", "")
        return None
    target_hdr = imregrid(infile_list[0], template='get')
    
    # N.B. Could put a lot of general logic here, but we are usually
    # working in a pretty specific case.

    if (target_hdr['csys']['direction0']['units'][0] != 'rad') or \
            (target_hdr['csys']['direction0']['units'][1] != 'rad'):
        casalog.post("Based on CASA experience. I expected pixel units of radians.", "SEVERE", "")
        casalog.post("I did not find this. Returning. Adjust code or investigate file "+infile_list[0], "SEVERE", "")
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
        casalog.post("This is a very big image you plan to create.", "WARN", "")
        casalog.post(ra_axis_size, " x ", dec_axis_size, "WARN", "")
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
        casalog.post("Missing required input.", "SEVERE", "")
        return    

    for ii in range(len(infile_list)):
        this_infile = infile_list[ii]
        this_outfile = outfile_list[ii]        

        if os.path.isdir(this_infile) == False:
            casalog.post("File "+this_infile+" not found. Continuing.", "WARN", "")
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
        casalog.post("Missing required input.", "SEVERE", "")
        return    
    
    # Look up parts

    this_mosaic_key = mosaic_key()
    if (gal in this_mosaic_key.keys()) == False:
        casalog.post("Galaxy "+gal+" not in mosaic key.", "SEVERE", "")
        return
    parts = this_mosaic_key[gal]

    # Read the key that defines the extent and center of the mosaic
    # manually. We will use this to figure out the target header.

    multipart_key = read_multipart_key()
    if (gal in multipart_key.keys()) == False:
        casalog.post("Galaxy "+gal+" not in multipart key.", "SEVERE", "")
        casalog.post("... working on a general header construction algorithm.", "SEVERE", "")
        casalog.post("... for now, go enter a center and size into the multipart key:", "SEVERE", "")
        casalog.post("... multipart_fields.txt ", "SEVERE", "")
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
        casalog.post("Missing required input.", "SEVERE", "")
        return    

    sum_file = outfile+'.sum'
    weight_file = outfile+'.weight'

    if (os.path.isdir(outfile) or \
            os.path.isdir(sum_file) or \
            os.path.isdir(weight_file)) and \
            (overwrite == False):
        casalog.post("Output file present and overwrite off.", "SEVERE", "")
        casalog.post("Returning.", "SEVERE", "")
        return

    if overwrite:
        os.system('rm -rf '+outfile+'.temp'+" > "+log_file+" 2>&1")
        os.system('rm -rf '+outfile+" > "+log_file+" 2>&1")
        os.system('rm -rf '+sum_file+" > "+log_file+" 2>&1")
        os.system('rm -rf '+weight_file+" > "+log_file+" 2>&1")
        os.system('rm -rf '+outfile+'.mask'+" > "+log_file+" 2>&1")

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
        casalog.post("Missing required input.", "SEVERE", "")
        return    
    
    # Look up parts

    this_mosaic_key = mosaic_key()
    if (gal in this_mosaic_key.keys()) == False:
        casalog.post("Galaxy "+gal+" not in mosaic key.", "SEVERE", "")
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
                casalog.post("Missing "+infile, "WARN", "")
                casalog.post("Skipping.", "WARN", "")
                continue
            if (os.path.isdir(wtfile) == False):
                casalog.post("Missing "+wtfile, "WARN", "")
                casalog.post("Skipping.", "WARN", "")
                continue
            infile_list.append(infile)
            wtfile_list.append(wtfile)

        if len(infile_list) == 0:
            casalog.post("No files to include in the mosaic. Returning.", "SEVERE", "")
            return

        outfile = output_dir+gal+'_'+array+'_'+product+'_'+this_ext+'.image'
    
        mosaic_aligned_data(
            infile_list = infile_list, weightfile_list = wtfile_list,
            outfile = outfile, overwrite=overwrite)
