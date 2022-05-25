# Copyright 2020, General Electric Company. All rights reserved. See https://github.com/xcist/code/blob/master/LICENSE

#import numpy.matlib as nm
from scipy import interpolate
import matplotlib.pyplot as plt
from catsim.pyfiles.CommonTools import *

def GetDefaultWidthLength(shape):
    shape = shape.lower()
    if shape=="uniform":
        width = 1.
        length = 1.
    elif shape=="gaussian":
        width = 1.
        length = 1.
    elif shape=="performix":
        width = 1.
        length = 1.
    elif shape=="pharos_small":
        width = 1.
        length = 1.
    elif shape=="pharos_large":
        width = 1.
        length = 1.
    elif shape=="gemini":
        width = 1.
        length = 1.

    return width, length

def GetIntensity(cfg):
    # suppose in experiments we have 30x30 samples, which sample take pixelX*pixelZ mm
    nx = 30
    ny = 30
    x_grid, y_grid = np.mgrid[0:nx, 0:ny]
    if cfg.scanner.focalspotShape.lower() == "gaussian":
        # one sigma in units of pixel number(or should we define sigma in mm?)
        # question: how do we know the size of pixel?
        if hasattr(cfg.scanner, "focalspotSigmaX"): sx = cfg.scanner.focalspotSigmaX
        else: sx = 10
        if hasattr(cfg.scanner, "focalspotSigmaZ"): sz = cfg.scanner.focalspotSigmaZ
        else: sz = 10
        weights = np.exp(-((x_grid-(nx-1)/2)**2/sx**2+(y_grid-(ny-1)/2)**2/sz**2)/2)
    elif cfg.scanner.focalspotShape.lower() == 'uniform':
        weights = np.ones((nx, ny))
    weights /= np.sum(weights)

    return weights

def ParseFocalspotData(path):
    alldata = np.load(path)
    data = alldata['data']
    xstart = alldata['xstart']
    ystart = alldata['ystart']
    pixsize_x = alldata['pixsize_x']
    pixsize_z = alldata['pixsize_z']

    return data, pixsize_x, pixsize_z, xstart, ystart

def SetFocalspot(cfg):
    # if shape and data is not defined, defaults to Uniform; 
    if (not hasattr(cfg.scanner, "focalspotShape")) and (not hasattr(cfg.scanner, "focalspotData")):
        cfg.scanner.focalspotShape = "Uniform"
    elif hasattr(cfg.scanner, "focalspotShape") and hasattr(cfg.scanner, "focalspotData"):
        print("Warning: Both shape and data are provided")
    # load default width and length
    if not all([hasattr(cfg.scanner, "focalspotWidth"), hasattr(cfg.scanner, "focalspotLength")]):
        cfg.scanner.focalspotWidth, cfg.scanner.focalspotLength = GetDefaultWidthLength(cfg.scanner.focalspotShape)

    # load npz focus spot image, the measured intensity will always be in the xz plane, but we need to add y axis
    if hasattr(cfg.scanner, "focalspotData"):
        I, pixsize_x, pixsize_z, xstart, zstart = ParseFocalspotData(cfg.scanner.focalspotData)
    else:
        I = GetIntensity(cfg)
        xstart, zstart = 0, 0
        pixsize_x = cfg.scanner.focalspotPixSizeX
        pixsize_z = cfg.scanner.focalspotPixSizeZ

    nx, nz = I.shape
    ny = nz
    if hasattr(cfg.scanner, 'focalspotShape') and  cfg.scanner.focalspotShape.lower() == 'uniform':
        dx = cfg.scanner.focalspotWidth/nx
        dy = -cfg.scanner.focalspotLength/np.tan(cfg.scanner.targetAngle*np.pi/180.)/ny
        dz = cfg.scanner.focalspotLength/nz
    else:
        dx = pixsize_x
        dy = -pixsize_z/np.tan(cfg.scanner.targetAngle*np.pi/180)
        dz = pixsize_z

    # remove too small values
    I /= np.max(I)
    I[I<0.02] = 0
    valid_idx_x = np.sum(I, axis=1)>0.
    valid_idx_z = np.sum(I, axis=0)>0.
    I = I[np.ix_(valid_idx_x, valid_idx_z)]
    fs_pos_x = (xstart + dx*(np.arange(nx)-0.5))[valid_idx_x]
    fs_pos_z = (zstart + dz*(np.arange(nz)-0.5))[valid_idx_z]
    nx, nz = I.shape

    # recenter
    fs_pos_x -= (fs_pos_x[0] + fs_pos_x[-1])*0.5
    fs_pos_z -= (fs_pos_z[0] + fs_pos_z[-1])*0.5
    
    # rescale
    if not hasattr(cfg.scanner, 'focalspotShape') or cfg.scanner.focalspotShape.lower() != 'uniform':
        Ix = np.sum(I, axis=1)
        Ix /= np.max(Ix)
        max_idx = np.argmax(Ix)
        finterp_x_half1 = interpolate.interp1d(Ix[0:max_idx], fs_pos_x[0:max_idx])
        pos1 = finterp_x_half1(cfg.scanner.focalspotWidthThreshold)
        finterp_x_half2 = interpolate.interp1d(Ix[max_idx:], fs_pos_x[max_idx:])
        pos2 = finterp_x_half2(cfg.scanner.focalspotWidthThreshold)
        W0 = (pos2 - pos1) # in units of mm

        Iz = np.sum(I, axis=0)
        Iz /= np.max(Iz)
        max_idx = np.argmax(Iz)
        finterp_z_half1 = interpolate.interp1d(Iz[0:max_idx], fs_pos_z[0:max_idx])
        pos1 = finterp_z_half1(cfg.scanner.focalspotLengthThreshold)
        finterp_z_half2 = interpolate.interp1d(Iz[max_idx:], fs_pos_z[max_idx:])
        pos2 = finterp_z_half2(cfg.scanner.focalspotWidthThreshold)
        L0 = pos2 - pos1

        fs_pos_x *= cfg.scanner.focalspotWidth/W0
        fs_pos_z *= cfg.scanner.focalspotLength/L0

    fs_pos_y = -fs_pos_z/np.tan(cfg.scanner.targetAngle*np.pi/180.);

    # down sampling to match oversampling
    os_nx = cfg.physics.srcXSampleCount
    os_ny = cfg.physics.srcYSampleCount
    os_nz = os_ny
    os_dx = (fs_pos_x[-1] - fs_pos_x[0]+dx)/os_nx # TODO: why there is dx here?
    os_dz = (fs_pos_z[-1] - fs_pos_z[0]+dz)/os_nz
    os_x = (np.arange(os_nx)-(os_nx-1)*0.5)*os_dx
    os_z = (np.arange(os_nz)-(os_nz-1)*0.5)*os_dz
    #[xx, zz] = np.meshgrid(fs_pos_x, fs_pos_z)
    [os_xx, os_zz] = np.meshgrid(os_x, os_z)
    os_yy = os_zz/np.tan(cfg.scanner.targetAngle*np.pi/180.);
    # according to python, I.shape=[fs_pos_x.shape, fs_pos_z.shape]
    os_interp = interpolate.interp2d(fs_pos_z, fs_pos_x, I, kind='linear')
    os_I = os_interp(os_z, os_x)
    os_I /= np.max(os_I)

    # remove low-weight sampling and normalized
    weights = os_I
    valid_idx = weights>0.03
    nSamples = np.sum(valid_idx)
    samples = np.c_[os_xx[valid_idx], cfg.scanner.sid+os_yy[valid_idx], os_zz[valid_idx]]
    weights = weights[weights>0.03]
    weights /= np.sum(weights)

    # TODO: offset

    # find corners
    if nx==1 and ny==1:
        corners = samples
    elif nx>1 and ny>1:
        #corners = np.c_[samples[0, :], samples[os_nx-1, :], samples[(os_ny-1)*os_nx, :], samples[-1, :]].T
        corners = np.c_[os_xx, cfg.scanner.sid+os_yy, os_zz]
        corners = np.c_[corners[0, 0], corners[0, -1], corners[-1, 0], corners[-1, -1]].T
    else:
        corners = np.c_[samples[0, :], samples[-1, :]].T
    nCorners = corners.shape[0]
    
    # source definition
    if not cfg.src:
        cfg.src = CFG()
    cfg.src.nSamples = nSamples
    cfg.src.samples = np.single(samples)
    cfg.src.weights = np.single(weights)
    cfg.src.front   = np.array([[0, -1, 0]], dtype=np.single)
    cfg.src.lateral = np.array([[1, 0, 0]], dtype=np.single)
    cfg.src.long    = np.array([[0, 0, 1]], dtype=np.single)
    cfg.src.nCorners = nCorners
    cfg.src.corners = np.single(corners)

    breakpoint()
    return cfg
