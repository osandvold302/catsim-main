# Copyright 2020, General Electric Company. All rights reserved. See https://github.com/xcist/code/blob/master/LICENSE

from catsim.CommonTools import *
import matplotlib.pyplot as plt
# Need to import new recons as they are added
from reconstruction.pyfiles.fdk_equiAngle import fdk_equiAngle


def recon(cfg):

    # If doing the recon, load the projection data, do the recon, and save the resulting image volume.
    if cfg.do_Recon:
        prep = load_prep(cfg)

        # The following line doesn't work - need to fix it when new recons are added.
        # imageVolume3D = feval("reconstruction." + cfg.recon.reconType, cfg, prep)
        # imageVolume3D = feval("reconstruction.pyfiles." + cfg.recon.reconType, cfg, prep)

        # A hack until the previous line is fixed.
        imageVolume3D = fdk_equiAngle(cfg, prep)
        imageVolume3D = scaleReconData(cfg, imageVolume3D)

        if cfg.recon.saveImageVolume:
            saveImageVolume(cfg, imageVolume3D)

    # If not doing the recon, load the previously-saved recon image volume.
    else:
        imageVolume3D = loadImageVolume(cfg)

    # In either case, save the results as individual images and display results at the specified window/level.
    if cfg.recon.saveSingleImages:
        saveSingleImages(cfg, imageVolume3D)
            
    if cfg.recon.displayImagePictures:
        cfg = displayImagePictures(cfg, imageVolume3D)

    if cfg.recon.saveImagePictureFiles:
        cfg = saveImagePictureFiles(cfg, imageVolume3D)

    return cfg

     
def load_prep(cfg):

    print("* Loading the projection data...")
    prep = rawread(cfg.resultsName + ".prep",
                  [cfg.protocol.viewCount, cfg.scanner.detectorRowCount, cfg.scanner.detectorColCount],
                  'float')
                  
    return prep


def scaleReconData(cfg, imageVolume3D):

    print('* Scaling recon data...')
    if cfg.recon.unit =='HU':
        imageVolume3D = imageVolume3D*(1000/(cfg.recon.mu)) + cfg.recon.huOffset
    elif cfg.recon.unit == '/mm':
        pass
    elif cfg.recon.unit == '/cm':
        imageVolume3D = imageVolume3D*10
    else:
        raise Exception('******** Error! An unsupported recon unit was specified: {:s}. ********'.format(cfg.recon.unit))

    return imageVolume3D


def saveImageVolume(cfg, imageVolume3D):

    print('* Writing the recon results to one big file...')

    imageVolume3D_size_string = str(cfg.recon.imageSize) + 'x' + str(cfg.recon.imageSize) + 'x' + str(cfg.recon.sliceCount)
    fname = cfg.resultsName + '_' + imageVolume3D_size_string + '.raw'
    imageVolume3D = imageVolume3D.transpose(2, 0, 1)
    imageVolume3D = imageVolume3D.copy(order='C')
    rawwrite(fname, imageVolume3D)


def loadImageVolume(cfg):

    print('* Reading the recon results from one big file...')

    imageVolume3D_size_string = str(cfg.recon.imageSize) + 'x' + str(cfg.recon.imageSize) + 'x' + str(cfg.recon.sliceCount)
    fname = cfg.resultsName + '_' + imageVolume3D_size_string + '.raw'
    imageVolume3D = rawread(fname,
                    [cfg.recon.sliceCount, cfg.recon.imageSize, cfg.recon.imageSize],
                    'float')
    imageVolume3D = imageVolume3D.copy(order='C')
    imageVolume3D = imageVolume3D.transpose(1, 2, 0)

    return imageVolume3D


def saveSingleImages(cfg, imageVolume3D):

    print('* Writing the recon results to individual files...')

    sliceIndicesToSave = range(0, cfg.recon.sliceCount)
    for sliceIndexToSave in sliceIndicesToSave:
        imageVolume3D_size_string = str(cfg.recon.imageSize) + 'x' + str(cfg.recon.imageSize) + 'x1'
        sliceNumberString = 'slice' + str(sliceIndexToSave+1).zfill(3) + 'of' + str(cfg.recon.sliceCount).zfill(3)
        fileName = cfg.resultsName + '_' + sliceNumberString + '_' + imageVolume3D_size_string + '.raw'
        sliceToSave = imageVolume3D[:, :, sliceIndexToSave]
        sliceToSave = sliceToSave.copy(order='C')
        rawwrite(fileName, sliceToSave)


def displayImagePictures(cfg, imageVolume3D):

    cfg = drawImages('screen', cfg, imageVolume3D)
    
    return cfg


def saveImagePictureFiles(cfg, imageVolume3D):

    print('* Saving the recon results to individual .png files...')

    cfg = drawImages('file', cfg, imageVolume3D)
    
    return cfg


def drawImages(drawTo, cfg, imageVolume3D):

    # Draw all images.
    # Future improvement: allow caller to specifiy a list of images to draw.
    sliceIndicesToDraw = range(0, cfg.recon.sliceCount)

    # If displayWindowMin and displayWindowMax were not passed in,
    # get them from the image data, so all images are displayed using the same W/L.
    if not hasattr(cfg, 'displayWindowMin'):
        cfg.displayWindowMin = np.min(imageVolume3D)
    if not hasattr(cfg, 'displayWindowMax'):
        cfg.displayWindowMax = np.max(imageVolume3D)

    for sliceIndexToDraw in sliceIndicesToDraw:
        sliceToDraw = imageVolume3D[:, :, sliceIndexToDraw]
        sliceToDraw = sliceToDraw.copy(order='C')
        sliceNumberString = 'slice' + str(sliceIndexToDraw+1).zfill(3) + 'of' + str(cfg.recon.sliceCount).zfill(3)
        fileName = cfg.resultsName + '_' + sliceNumberString + '.png'
        plt.figure(int(sliceIndexToDraw+1))
        plt.imshow(sliceToDraw, cmap='gray', vmin=cfg.displayWindowMin, vmax=cfg.displayWindowMax)
        if not cfg.recon.displayImagePictureAxes:
            plt.axis('off')

        if cfg.recon.displayImagePictureTitles:
            sliceString = "slice " + str(sliceIndexToDraw+1) + " of " + str(cfg.recon.sliceCount)
            if hasattr(cfg, 'reconImageTitle'):
                # If a plot title is specified, use it, and add the slice info if specified.
                if hasattr(cfg, 'addSliceInfoToReconImageTitle') \
                and cfg.addSliceInfoToReconImageTitle:
                    titleString = cfg.reconImageTitle + "\n" + sliceString
                else:
                    titleString = cfg.reconImageTitle
            else:
                # Otherwise, title the plot with the slice info.
                titleString = sliceString
            plt.title(titleString, fontsize=10)

        if drawTo == 'file':
            plt.savefig(fileName, bbox_inches='tight')
            plt.close()
        elif drawTo == 'screen':
            plt.draw()
            
    if drawTo == 'screen':
        plt.pause(1)
        if cfg.waitForKeypress:
            print('********************************************')
            print('* Press Enter to close images and continue *')
            input('********************************************')
        plt.close('all')

    return cfg
