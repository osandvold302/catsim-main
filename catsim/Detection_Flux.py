# Copyright 2020, General Electric Company. All rights reserved. See https://github.com/xcist/code/blob/master/LICENSE

import numpy as np
import numpy.matlib as nm
from catsim.CommonTools import *

def Detection_Flux(cfg):
    '''
    Compute the photon flux at the detector per cell per subview
    Flux dim: [pixel, Ebin] ([col, row, Ebin])
    Note: the unit of spectrum file: photons per mAs per cm^2 at 1-m distance
          and spec.Ivec is already scaled to mA and view time, i.e. mAs
    Mingye Wu, GE Research
    
    '''
    ###------- offset scan, flux = 0
    if cfg.sim.isOffsetScan:
        cfg.detFlux = np.zeros([cfg.det.totalNumCells, cfg.spec.nEbin], dtype=np.single)
        return cfg
    
    ###------- air or phantom scan
    detActiveArea = cfg.det.activeArea/100*cfg.det.cosBetas # cm^2
    detActiveArea = nm.repmat(detActiveArea, 1, cfg.spec.nEbin)
    
    distanceFactor = np.square(1000/cfg.det.rayDistance) # mm
    distanceFactor = nm.repmat(distanceFactor, 1, cfg.spec.nEbin)
    
    cfg.spec.netIvec = cfg.spec.Ivec*cfg.src.filterTrans
    cfg.detFlux = np.single(cfg.spec.netIvec*detActiveArea*distanceFactor)

    return cfg

if __name__ == "__main__":

    cfg = source_cfg("./cfg/default.cfg")
    
    cfg.sim.isOffsetScan = 0
    
    cfg = feval(cfg.scanner.detectorCallback, cfg)
    cfg = feval(cfg.scanner.focalspotCallback, cfg)
    cfg = feval(cfg.physics.rayAngleCallback, cfg)
    cfg = feval(cfg.protocol.spectrumCallback, cfg)
    cfg = feval(cfg.protocol.filterCallback, cfg)
    
    cfg = Detection_Flux(cfg)
    check_value(cfg.detFlux)