# Copyright 2022, General Electric Company. All rights reserved. See https://github.com/xcist/code/blob/master/LICENSE

import numpy.matlib as nm
from scipy import interpolate, io
import matplotlib.pyplot as plt
from catsim.pyfiles.CommonTools import *
from collections import defaultdict, OrderedDict

def ValidateFocalspot(samples, weights):
    print("X center: ", np.average(samples[:,0], weights=weights))
    print("Y center: ", np.average(samples[:,1], weights=weights))
    print("Z center: ", np.average(samples[:,2], weights=weights))
    print("total intensity: ", np.sum(weights))

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
        scanner.fs_performix_width = 0.92
        scanner.fs_performix_length = 0.76
    elif shape=="pharos_small":
        width = 1.
        length = 1.
    elif shape=="pharos_large":
        width = 1.
        length = 1.
    elif shape=="gemini_small":
        width = 1.
        length = 1.
    elif shape=="gemini_large":
        width = 1.
        length = 1.

    return width, length

def GetIntensity(cfg):
    nx = max(10, 2*cfg.physics.srcXSampleCount)
    ny = max(10, 2*cfg.physics.srcYSampleCount)
    x_grid, y_grid = np.mgrid[0:nx, 0:ny]
    if cfg.scanner.focalspotShape.lower() == "gaussian":
        # one sigma in units of pixel number(or should we define sigma in mm?)
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

    _, ext = os.path.splitext(path)
    # ending in .mat
    if '.mat' == ext:
        alldata = io.loadmat(path)
        data = alldata['I'].T
        xstart = alldata['xstart'][0,0]
        ystart = alldata['zstart'][0,0]
        pixsize_x = alldata['dx'][0,0]
        pixsize_z = alldata['dz'][0,0]
    # ending in .npz
    elif '.npz' == ext:
        alldata = np.load(path)
        data = alldata['data']
        xstart = alldata['xstart']
        ystart = alldata['ystart']
        pixsize_x = alldata['pixsize_x']
        pixsize_z = alldata['pixsize_z']

    return data, pixsize_x, pixsize_z, xstart, ystart

# plot profiles
def PlotProfile(cfg, data, weights, save=True, show=True, filename=None):
    # data are (x, y, I)
    allx = data[:,0]
    allz = data[:,2]
    #plt.xlabel()
    #assert len(set(allx))==cfg.physics.srcXSampleCount, "x pos has small bounding errors, may lead to wrong results"
    #assert len(set(allz))==cfg.physics.srcYSampleCount, "z pos has small bounding errors, may lead to wrong results"

    xprofile = defaultdict(float)
    zprofile = defaultdict(float)
    for i in range(len(weights)):
       xprofile[allx[i]] += weights[i]
       zprofile[allz[i]] += weights[i]
    xprofile = OrderedDict(sorted(xprofile.items()))
    max_xprof = np.max([*xprofile.values()])
    max_zprof = np.max([*zprofile.values()])
    zprofile = OrderedDict(sorted(zprofile.items()))

    plt.plot([*xprofile.keys()], [*xprofile.values()], 'r', label='X')
    plt.axhline(y=cfg.scanner.focalspotWidthThreshold*max_xprof, color='r', linestyle='--')
    plt.plot([*zprofile.keys()], [*zprofile.values()], 'b', label='Z')
    plt.axhline(y=cfg.scanner.focalspotLengthThreshold*max_zprof, color='b', linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.grid()
    if save:
        if cfg is not None:
            name = ""
            if hasattr(cfg.scanner, "focalspotShape"): name += cfg.scanner.focalspotShape
            else: name += os.path.basename(cfg.scanner.focalspotData)
            name += "_src-{}x{}".format(cfg.physics.srcXSampleCount, cfg.physics.srcYSampleCount)
            name += "_spotsize-{:.2f}x{:.2f}".format(cfg.scanner.focalspotWidth, cfg.scanner.focalspotLength)
            name += "_thre-{:.2f}x{:.2f}".format(cfg.scanner.focalspotWidthThreshold, cfg.scanner.focalspotLengthThreshold)
            name += "_pixsize-{:.2f}x{:.2f}".format(cfg.scanner.focalspotPixSizeX, cfg.scanner.focalspotPixSizeZ)
            plt.savefig(name+"_profiles.png")
        elif filename is not None:
            plt.savefig(filename)
    if show: plt.show()
    plt.close()

# input data is pixel cloud
def TriSurfData(cfg, data, weights, save=True, show=True):
    import matplotlib.tri as mtri

    allx = data[:,0]
    #ally = data[:,1]
    allz = data[:,2]
    #fig, ax = plt.subplots(projection='3d')
    #breakpoint()
    tri = mtri.Triangulation(allx, allz)
    fig = plt.figure(figsize=(15,10))
    #ax = fig.gca(projection='3d')
    ax = fig.add_subplot(111, projection='3d')
    im = ax.plot_trisurf(allx, allz, weights, triangles=tri.triangles, cmap="jet", antialiased=False, linewidth=0.05, edgecolors='k')
    #im = ax.plot_trisurf(allx, allz, weights, linewidth=0.2, cmap=antialiased=True)
    #ax.set_aspect((np.max(allz)-np.min(allz))/(np.max(allx)-np.min(allx)))
    _z = (np.max(allz)-np.min(allz))/(np.max(allx)-np.min(allx))
    #_z = (np.max(allx)-np.min(allx))/(np.max(allz)-np.min(allz))
    ax.set_box_aspect([1, _z, 1])
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('weights')
    fig.colorbar(im)
    #ax.view_init(30, -90-37.5) #if performix
    ax.view_init(30, -37.5) #else
    #ax.view_init(90, 0) #else
    #plt.xlabel()
    plt.tight_layout()
    if save:
        name = ""
        if hasattr(cfg.scanner, "focalspotShape"): name += cfg.scanner.focalspotShape
        else: name += os.path.basename(cfg.scanner.focalspotData)
        name += "_src-{}x{}".format(cfg.physics.srcXSampleCount, cfg.physics.srcYSampleCount)
        name += "_spotsize-{:.2f}x{:.2f}".format(cfg.scanner.focalspotWidth, cfg.scanner.focalspotLength)
        name += "_thre-{:.2f}x{:.2f}".format(cfg.scanner.focalspotWidthThreshold, cfg.scanner.focalspotLengthThreshold)
        name += "_pixsize-{:.2f}x{:.2f}".format(cfg.scanner.focalspotPixSizeX, cfg.scanner.focalspotPixSizeZ)
        plt.savefig(name+".png")
    if show: plt.show()
    plt.close()
    #breakpoint()

def SetFocalspot(cfg):
    # if shape and data is not defined, defaults to Uniform; 
    if (not hasattr(cfg.scanner, "focalspotShape")) and (not hasattr(cfg.scanner, "focalspotData")):
        cfg.scanner.focalspotShape = "Uniform"
    elif hasattr(cfg.scanner, "focalspotShape") and hasattr(cfg.scanner, "focalspotData"):
        print("Error: Both shape and data are provided in focal spot.")
        sys.exit()
    # load default width and length
    if not all([hasattr(cfg.scanner, "focalspotWidth"), hasattr(cfg.scanner, "focalspotLength")]):
        cfg.scanner.focalspotWidth, cfg.scanner.focalspotLength = GetDefaultWidthLength(cfg.scanner.focalspotShape)

    # load npz focus spot image, the measured intensity will always be in the xz plane, but we need to add y axis
    if hasattr(cfg.scanner, "focalspotData"):
        I, pixsize_x, pixsize_z, xstart, zstart = ParseFocalspotData(cfg.scanner.focalspotData)
        cfg.scanner.focalspotPixSizeX = pixsize_x
        cfg.scanner.focalspotPixSizeZ = pixsize_z
    else:
        I = GetIntensity(cfg)
        xstart, zstart = 0, 0
        pixsize_x = cfg.scanner.focalspotPixSizeX
        pixsize_z = cfg.scanner.focalspotPixSizeZ

    nx, nz = I.shape
    ny = nz
    if hasattr(cfg.scanner, 'focalspotShape') and cfg.scanner.focalspotShape.lower() == 'uniform':
        dx = cfg.scanner.focalspotWidth/nx
        dy = -cfg.scanner.focalspotLength/np.tan(cfg.scanner.targetAngle*np.pi/180.)/ny
        dz = cfg.scanner.focalspotLength/nz
    else:
        dx = pixsize_x
        dy = -pixsize_z/np.tan(cfg.scanner.targetAngle*np.pi/180)
        dz = pixsize_z

    # remove too small values
    I /= np.max(I)
    I[I<0.02] = 0 # necessary, especially for image data with negative intensities
    valid_idx_x = np.sum(I, axis=1)>0.
    valid_idx_z = np.sum(I, axis=0)>0.
    I = I[np.ix_(valid_idx_x, valid_idx_z)]
    fs_pos_x = (xstart + dx*(np.arange(nx)+0.5))[valid_idx_x]
    fs_pos_z = (zstart + dz*(np.arange(nz)+0.5))[valid_idx_z]
    nx, nz = I.shape

    # recenter
    fs_pos_x -= (fs_pos_x[0] + fs_pos_x[-1])*0.5
    fs_pos_z -= (fs_pos_z[0] + fs_pos_z[-1])*0.5
    
    # rescale
    if not hasattr(cfg.scanner, 'focalspotShape') or cfg.scanner.focalspotShape.lower() != 'uniform':
        Ix = np.sum(I, axis=1)
        Ix /= np.max(Ix)
        #max_idx = np.argmax(Ix)
        # np.interp will have wrong results, don't know why
        #pos1 = np.interp(cfg.scanner.focalspotWidthThreshold, Ix[0:max_idx], fs_pos_x[0:max_idx])
        #pos2 = np.interp(cfg.scanner.focalspotWidthThreshold, Ix[max_idx:], fs_pos_x[max_idx:])
        idx = np.argwhere(Ix>cfg.scanner.focalspotWidthThreshold)[0][0]
        if idx==0:
            pos1 = fs_pos_x[0]
        else:
            pos1 = np.interp(cfg.scanner.focalspotWidthThreshold, Ix[idx-1:idx], fs_pos_x[idx-1:idx])
        idx = np.argwhere(Ix>cfg.scanner.focalspotWidthThreshold)[-1][0]
        if idx==len(fs_pos_x)-1:
            pos2 = fs_pos_x[-1]
        else:
            pos2 = np.interp(cfg.scanner.focalspotWidthThreshold, Ix[idx:idx+1], fs_pos_x[idx:idx+1])
        # I think we should not use this interpolateion because fs_pos could be not a function of I
        # because the I may not be monotonical
        #finterp_x_half1 = interpolate.interp1d(Ix[0:max_idx], fs_pos_x[0:max_idx])
        #pos1 = finterp_x_half1(cfg.scanner.focalspotWidthThreshold)
        #finterp_x_half2 = interpolate.interp1d(Ix[max_idx:], fs_pos_x[max_idx:])
        #pos2 = finterp_x_half2(cfg.scanner.focalspotWidthThreshold)
        W0 = np.abs(pos2 - pos1) # in units of mm

        Iz = np.sum(I, axis=0)
        Iz /= np.max(Iz)
        #max_idx = np.argmax(Iz)
        #pos1 = np.interp(cfg.scanner.focalspotLengthThreshold, Iz[0:max_idx], fs_pos_z[0:max_idx])
        #pos2 = np.interp(cfg.scanner.focalspotLengthThreshold, Iz[max_idx:], fs_pos_z[max_idx:])
        #finterp_z_half1 = interpolate.interp1d(Iz[0:max_idx], fs_pos_z[0:max_idx])
        #pos1 = finterp_z_half1(cfg.scanner.focalspotLengthThreshold)
        #finterp_z_half2 = interpolate.interp1d(Iz[max_idx:], fs_pos_z[max_idx:])
        #pos2 = finterp_z_half2(cfg.scanner.focalspotLengthThreshold)
        idx = np.argwhere(Iz>cfg.scanner.focalspotLengthThreshold)[0][0]
        if idx==0:
            pos1 = fs_pos_z[0]
        else:
            pos1 = np.interp(cfg.scanner.focalspotLengthThreshold, Iz[idx-1:idx], fs_pos_z[idx-1:idx])
        idx = np.argwhere(Iz>cfg.scanner.focalspotLengthThreshold)[-1][0]
        if idx==len(fs_pos_z)-1:
            pos2 = fs_pos_z[-1]
        else:
            pos2 = np.interp(cfg.scanner.focalspotLengthThreshold, Iz[idx:idx+1], fs_pos_z[idx:idx+1])
        L0 = np.abs(pos2 - pos1)

    # down sampling to match oversampling
    os_nx = cfg.physics.srcXSampleCount
    os_ny = cfg.physics.srcYSampleCount
    os_nz = os_ny
    def GetRange(pos, profile, th):
        '''
        clever way of determining sampling range based on position, profile, and threshold
        '''
        profile /= np.max(profile)
        idx = np.argwhere(profile>th)[0][0]
        if idx==0:
            new_begin = pos[0]
        else:
            new_begin = np.interp(th, profile[idx-1:idx], pos[idx-1:idx])
        idx = np.argwhere(profile>th)[-1][0]
        if idx==len(pos)-1:
            new_end = pos[-1]
        else:
            new_end = np.interp(th, profile[idx:idx+1], pos[idx:idx+1])

        return new_begin, new_end

    # clever way of sampling
    os_range_x = GetRange(fs_pos_x, np.sum(I, axis=1), 0.02)
    os_range_z = GetRange(fs_pos_z, np.sum(I, axis=0), 0.02)
    os_dx = (os_range_x[1] - os_range_x[0]+dx)/os_nx # TODO: why there is dx here?
    os_dz = (os_range_z[1] - os_range_z[0]+dz)/os_nz
    os_x = (np.arange(os_nx)-(os_nx-1)*0.5)*os_dx
    os_z = (np.arange(os_nz)-(os_nz-1)*0.5)*os_dz
    [os_xx, os_zz] = np.meshgrid(os_x, os_z)
    #When on a regular grid with x.size = m and y.size = n, if z.ndim == 2, then z must have shape (n, m)
    os_interp = interpolate.interp2d(fs_pos_x, fs_pos_z, I.T, kind='linear')
    os_I = os_interp(os_x, os_z)
    os_I /= np.max(os_I)

    if hasattr(cfg.scanner, 'focalspotData') or cfg.scanner.focalspotShape.lower() != 'uniform':
        os_xx *= cfg.scanner.focalspotWidth/W0
        os_zz *= cfg.scanner.focalspotLength/L0

    os_yy = os_zz/np.tan(cfg.scanner.targetAngle*np.pi/180.);

    # remove low-weight sampling and normalized
    weights = os_I
    valid_idx = weights>0.02
    nSamples = np.sum(valid_idx)
    samples = np.c_[os_xx[valid_idx], cfg.scanner.sid+os_yy[valid_idx], os_zz[valid_idx]]
    weights = weights[weights>0.02]
    weights /= np.sum(weights)

    # re-center samples based on center of mass
    #samples[:,0] -= (np.max(samples[:,0]) + np.min(samples[:,0]))*0.5
    #samples[:,1] -= (np.max(samples[:,1]) + np.min(samples[:,1]))*0.5
    #samples[:,2] -= (np.max(samples[:,2]) + np.min(samples[:,2]))*0.5
    samples[:,0] -= np.average(samples[:,0], weights=weights)
    samples[:,1] -= np.average(samples[:,1], weights=weights)
    samples[:,2] -= np.average(samples[:,2], weights=weights)
    # validate samples and weights
    #ValidateFocalspot(samples, weights)
    #name = ""
    #if hasattr(cfg.scanner, "focalspotShape"): name += cfg.scanner.focalspotShape
    #else: name += os.path.basename(cfg.scanner.focalspotData)
    #name += "_src-{}x{}".format(cfg.physics.srcXSampleCount, cfg.physics.srcYSampleCount)
    #name += "_spotsize-{:.2f}x{:.2f}".format(cfg.scanner.focalspotWidth, cfg.scanner.focalspotLength)
    #if "uniform" not in name:
    #    name += "_thre-{:.2f}x{:.2f}".format(cfg.scanner.focalspotWidthThreshold, cfg.scanner.focalspotLengthThreshold)
    #name += "_pixsize-{:.2f}x{:.2f}".format(cfg.scanner.focalspotPixSizeX, cfg.scanner.focalspotPixSizeZ)
    #with open(name+".npz", "wb") as f:
    #    np.savez(f, samples=samples, weights=weights)

    #breakpoint()
    #print(*zip(samples[:,0], samples[:,2], weights))

    # offset
    if hasattr(cfg.protocol, 'focalspotOffset'):
        samples = samples + nm.repmat(cfg.protocol.focalspotOffset, nSamples, 1)

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

    #with open("performix_sampling.npz", 'wb') as f:
    #    np.savez(f, samples=samples, weights=weights)
    #TriSurfData(cfg, samples, weights, True, False)
    #PlotProfile(cfg, samples, weights, True, False)

    #VisInterpData(cfg, samples, weights, True, False)
    #exit()

    return cfg
