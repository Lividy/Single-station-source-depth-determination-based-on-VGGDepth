import numpy as np
import os
import segyio
import struct
import copy
import random
# from tqdm.notebook import tqdm
import mmap
import matplotlib.pyplot as plt

import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

def binread(filename,nx,nz):
    f = open(filename,'rb')
    m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    data = np.zeros(nx*nz)
    for i in range(nz):
        temp = struct.unpack('f'*nx,m.read(4*nx))
        data[i*nx:(i+1)*nx] = temp
    f.close()
    data = np.asarray(data).reshape(nx,nz).transpose()
    return data

def binwrite(filename,data):
    data = np.float32(np.ascontiguousarray(data))
    f = open(filename,'wb')
    f.write(data)
    f.close()

def plotwigb(data, h=1, perc=100, linewidth=0.5, color='k', title=None, xlabel=None, ylabel=None, aspect=None, **kwargs):
    """
    Plot a wiggle trace with constant amplitude h.
    """
    nt, nx = data.shape
    t = range(nt,0,-1)    
    x = np.arange(nx)
    if perc < 100:
        clip = np.percentile(np.abs(data), perc)
    else:
        clip = np.max(np.abs(data))
    data = np.clip(data, -clip, clip)
    data = data / clip * h / 2

    for i in range(nx):
        plt.plot(data[:, i] + x[i],t, color=color, linewidth=linewidth, **kwargs)
    # plt.xlim(x[0] - h / 2, x[-1] + h / 2)
    
    ax = plt.gca()
    yticks = ax.get_yticks()    
    yticks = list(map(int, yticks))
    ax.set_yticklabels(reversed(yticks))

    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if aspect is not None:
        plt.gca().set_aspect(aspect)

def plotseis(data, h=1, perc=100, title=None, xlabel=None, ylabel=None, aspect=None, **kwargs):
    """
    Plot a seismic section.
    """
    if perc < 100:
        clip = np.percentile(np.abs(data), perc)
    else:
        clip = np.max(np.abs(data))
    data = np.clip(data, -clip, clip)
    data = data / clip * h / 2

    plt.imshow(data, aspect='auto', cmap=plt.cm.seismic, **kwargs)   
    
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if aspect is not None:
        plt.gca().set_aspect(aspect)

def random_select_without_adjacent(lst, count):
    selected = []

    if count > len(lst):
        raise ValueError('抽取的数量超过列表长度')

    while len(selected) < count:
        element = random.choice(lst)
        if not selected:
            selected.append(element)
        else:
            switch = True
            for i in selected:
                if abs(i - element) < 2:
                    switch = False
                    break
            if switch:
                selected.append(element)

    return selected

def random_select_without_adjacentN(lst, count, N):
    selected = []

    if count * (N+1) > len(lst):
        raise ValueError('抽取的数量超过列表长度')

    i0 = 0
    for i in range(count):
        index = random.choice(range(i0,len(lst)-(count-i-1)*(N+1)))
        
        selected.append(lst[index])
        i0 = index + N + 1

    return selected

def Segy2CMP(SegyPath):
    CMPGahter = []
    OffsetVec = []
    SegyFile = segyio.open(SegyPath, "r", strict=False)
    cdp = list(set(SegyFile.attributes(segyio.TraceField.CDP)[:]))
    cdpindex = SegyFile.attributes(segyio.TraceField.CDP)[:]
    bar = tqdm(total = len(cdp))
    bar.set_description('Processing')
    for i in cdp:
        index = np.where(cdpindex == i)[0]
        offset = SegyFile.attributes(segyio.TraceField.offset)[index]        
        cmpdata = []
        for j in range(len(index)):
            cmpdata.append(SegyFile.trace[index[j]])
        CMPGahter.append(np.array(cmpdata))
        OffsetVec.append(np.array(offset))
        bar.update(1)
    cmpRange = [np.min(cdp),np.max(cdp)]
    return CMPGahter,cmpRange,OffsetVec

def CMP2ZeroOffset(SegyPath):
    CMPGahter = []
    OffsetVec = []
    SegyFile = segyio.open(SegyPath, "r", strict=False)
    cdp = list(set(SegyFile.attributes(segyio.TraceField.CDP)[:]))
    cdpindex = SegyFile.attributes(segyio.TraceField.CDP)[:]    
    for i in cdp:
        index = np.where(cdpindex == i)[0]
        offset = abs(SegyFile.attributes(segyio.TraceField.offset)[index])
        offsetsort = offset.argsort()
        cmpdata = []
        for j in range(len(index)):
            cmpdata.append(SegyFile.trace[int(index[offsetsort[j]])])
        CMPGahter.append(np.array(cmpdata))
        OffsetVec.append(np.array(offset[offsetsort]))        
    cmpRange = [np.min(cdp),np.max(cdp)]
    return CMPGahter,cmpRange,OffsetVec

def CMP2ZeroOffset_tqdm(SegyPath):
    CMPGahter = []
    OffsetVec = [] 
    SegyFile = segyio.open(SegyPath, "r", strict=False)
    cdp = list(set(SegyFile.attributes(segyio.TraceField.CDP)[:]))
    cdpindex = SegyFile.attributes(segyio.TraceField.CDP)[:]
    bar = tqdm(total = len(cdp))
    bar.set_description('Processing')
    for i in cdp:
        index = np.where(cdpindex == i)[0]
        offset = abs(SegyFile.attributes(segyio.TraceField.offset)[index])
        offsetsort = offset.argsort()
        cmpdata = []
        for j in range(len(index)):
            cmpdata.append(SegyFile.trace[int(index[offsetsort[j]])])
        CMPGahter.append(np.array(cmpdata))
        OffsetVec.append(np.array(offset[offsetsort]))
        bar.update(1)
    cmpRange = [np.min(cdp),np.max(cdp)]
    return CMPGahter,cmpRange,OffsetVec

def CMP2ZeroOffset_check(SegyPath):
    CMPGahter = []
    OffsetVec = [] 
    SegyFile = segyio.open(SegyPath, "r", strict=False)
    cdp = list(set(SegyFile.attributes(segyio.TraceField.CDP)[:]))
    cdpindex = SegyFile.attributes(segyio.TraceField.CDP)[:]    
    for i in cdp:
        index = np.where(cdpindex == i)[0]
        offset = abs(SegyFile.attributes(segyio.TraceField.offset)[index])
        offsetsort = offset.argsort()
        cmpdata = []
        for j in range(len(index)):
            cmpdata.append(SegyFile.trace[int(index[offsetsort[j]])])
        CMPGahter.append(np.array(cmpdata))
        OffsetVec.append(np.array(offset[offsetsort]))
    cmpRange = [np.min(cdp),np.max(cdp)]
    return CMPGahter,cmpRange,OffsetVec

def CMP2ZeroOffset2(SegyPath):
    with segyio.open(SegyPath, "r", strict=False) as SegyFile:
        cdp = np.unique(SegyFile.attributes(segyio.TraceField.CDP)[:])
        cdpindex = SegyFile.attributes(segyio.TraceField.CDP)[:]
        bar = tqdm(total = len(cdp))
        bar.set_description('Processing')
        CMPGahter = []
        OffsetVec = [] 
        for i in cdp:
            index = np.where(cdpindex == i)[0]
            offset = np.abs(SegyFile.attributes(segyio.TraceField.offset)[index])
            offsetsort = offset.argsort()
            cmpdata = SegyFile.trace[index[offsetsort]]
            CMPGahter.append(cmpdata)
            OffsetVec.append(offset[offsetsort])
            bar.update(1)
        cmpRange = [np.min(cdp),np.max(cdp)]
    return CMPGahter,cmpRange,OffsetVec

def Segy2MinOffset(SegyPath):        
    with segyio.open(SegyPath, "r", strict=False) as SegyFile:
        cdp = list(set(SegyFile.attributes(segyio.TraceField.CDP)[:]))
        cdpindex = SegyFile.attributes(segyio.TraceField.CDP)[:]        
        nt = SegyFile.attributes(segyio.TraceField.TRACE_SAMPLE_COUNT)[:][0]
        CMPGahter0 = np.zeros([len(cdp),nt])
        OffsetVec0 = np.zeros([len(cdp),1])
        for i in range(len(cdp)):
            index = np.where(cdpindex == cdp[i])[0]
            offset = SegyFile.attributes(segyio.TraceField.offset)[index]
            absOffset = abs(offset)
            OffsetVec0[i] = offset[np.argmin(absOffset)]
            CMPGahter0[i,:] = SegyFile.trace[int(index[np.argmin(absOffset)])]            
        cmpRange = [np.min(cdp),np.max(cdp)]
    return CMPGahter0,cmpRange,OffsetVec0

def CutCMP(CMPGather0,OffsetVec0,cmpRange, cutRange):
    if cutRange[0] < cmpRange[0] or cutRange[1] > cmpRange[1]:
        print('Cut Range Error')
        return
    CutCMPGather = CMPGather0[cutRange[0]-cmpRange[0]:cutRange[1]-cmpRange[0]+1]
    CutOffsetVec = OffsetVec0[cutRange[0]-cmpRange[0]:cutRange[1]-cmpRange[0]+1]
    CutCMPRange = [cutRange[0],cutRange[1]]

    return CutCMPGather,CutCMPRange,CutOffsetVec

def CutOffset(CMPGather,OffsetVec,cutOffset):
    bar = tqdm(total = len(CMPGather))
    bar.set_description('Processing')
    CMPGather2 = []
    OffsetVec2 = []
    for i,data in enumerate(CMPGather):        
        ncmp = np.shape(np.where(abs(OffsetVec[i])<cutOffset))[1]        
        if ncmp > 1:
            cmpindex = np.squeeze(np.where(abs(OffsetVec[i])<cutOffset))
            CMPGather2.append(copy.deepcopy(data[cmpindex,:]))
            OffsetVec2.append(copy.deepcopy(OffsetVec[i][cmpindex]))
        elif ncmp == 1:
            CMPGather2.append(copy.deepcopy(data))
            OffsetVec2.append(copy.deepcopy(OffsetVec[i]))
        else:
            raise Exception("cutOffset is too small!")    
        
        bar.update(1)
            
    return CMPGather2,OffsetVec2

def SortZeroOffset(CMPGather,OffsetVec,nt):    
    Zoffset = np.zeros([nt,len(CMPGather)])
    for i, CMPdata in enumerate(CMPGather):
        Zoffset[:,i] = CMPdata[0,:]
    return Zoffset

def MakeCutMask(CMPGather,OffsetVec,MinT,MaxT,OffsetRange):
    k = (MaxT - MinT)/(OffsetRange[1] - OffsetRange[0])
    t0 = MinT - k * OffsetRange[0]
    Mask = copy.deepcopy(CMPGather)
    for i,data in enumerate(CMPGather):
        Masktemp = np.ones_like(data)
        offset = abs(OffsetVec[i])
        for j,x in enumerate(offset):
            t = int(t0 + k * x)
            Masktemp[j,:t] = 0
        Mask[i] = Masktemp
    return Mask

def MakeCutMask_Single(CMPGather,OffsetVec,MinT,MaxT,OffsetRange):
    k = (MaxT - MinT)/(OffsetRange[1] - OffsetRange[0])
    t0 = MinT - k * OffsetRange[0]
    Mask = np.ones_like(CMPGather)
    for itrace in range(CMPGather.shape[0]):
        x = abs(OffsetVec[itrace])
        t = int(t0 + k * abs(x))
        Mask[itrace,:t] = 0    
    return Mask

def MakeCutiMask_Single(OffsetVec,ntrace,nt,cutMinT,cutMaxT,OffsetRange,Vrms,tVec,dt):
    Mask = np.ones([ntrace,nt])
    for itrace in range(ntrace):
        a = (cutMaxT - cutMinT) / (OffsetRange[1] - OffsetRange[0])
        it = int(a * (OffsetVec[itrace] - OffsetRange[0]) + cutMinT)
        TravelT = np.sqrt(tVec[it]**2 + (OffsetVec[itrace] / Vrms[it])**2)
        tfloor = np.floor((TravelT - tVec[0]) / dt) 

        Mask[itrace,:int(tfloor)] = 0
    return Mask

def CutNMO(NMOGather,Mask):
    NMOcut = copy.deepcopy(NMOGather)
    for i,data in enumerate(NMOGather):
        NMOcut[i] = data * Mask[i]
    return NMOcut 

def StackNMO(NMOGather,Mask):
    bar = tqdm(total = len(NMOGather))
    bar.set_description('Processing')
    nt = (NMOGather[0].shape)[1]
    ncmp = len(NMOGather)
    Stackdata = np.zeros([nt,ncmp])
    for i,data in enumerate(NMOGather):
        for t in range(nt):
            for j,trace in enumerate(data):
                Stackdata[t,i] += trace[t]
            sn = np.count_nonzero(Mask[i][:,t]) if np.count_nonzero(Mask[i][:,t]) > 0 else 1
            Stackdata[t,i] /= sn
        bar.update(1)
    return Stackdata

def Amplitude_Compensation(CMPGather0, Vrms):
    cmpnum, nt = CMPGather0.shape
    Output = np.zeros([cmpnum,nt])
    Vmin = np.max(Vrms)

    Output = CMPGather0 * (Vrms.T**2) / (Vmin)**2 

    return Output

def GenerateRandomPhase(datasize,pixelsize,ratio):
    cmpnum,nx,nt = datasize
    px,pz = pixelsize
    pnx,pnz = int(np.ceil(nx / px)),int(np.ceil(nt / pz))
      
    random_phase_pixel = np.random.uniform(-np.pi*ratio,np.pi*ratio,[cmpnum,pnx,pnz])
    random_phase_temp = np.zeros([cmpnum,pnx * px,pnz * pz])
        
    random_phase_temp = np.tile(random_phase_pixel.reshape(cmpnum, pnx, 1, pnz, 1), (1, 1, px, 1, pz)).reshape(cmpnum, pnx * px, pnz * pz)
    
    random_phase = random_phase_temp[:,:nx,:nt]
    
    return random_phase

def GenerateBackgroundSigma(datasize,layerRange,mu,sigma):    
    cmpnum,nt = datasize
    layermin,layermax = layerRange
    layerlocal = []
    i = 0
    while i < nt:
        layerlocal.append(i)
        i += np.random.randint(layermin,layermax)
    layerlocal.append(nt)
    layerlocal = np.array(layerlocal)
    layernum = len(layerlocal) - 1
    layermeans = np.random.randint(0,sigma,[layernum]) + mu
    output = np.zeros([cmpnum,nt])
    for i in range(layernum):
        output[:,layerlocal[i]:layerlocal[i+1]] = layermeans[i]

    return output    

def GenerateCmpSigma(datasize,pixelsize,sigma_b,sigma_cmp):
    cmpnum,nx,nt = datasize
    px,pz = pixelsize
    pnx,pnz = int(np.ceil(nx / px)),int(np.ceil(nt / pz))

    background = np.ones([cmpnum,nx,nt])
    for i in range(cmpnum):
        for j in range(nx):
            background[i,j,:] = background[i,j,:] * sigma_b[i,:]

    sigma_add = np.random.randint(0,sigma_cmp,[cmpnum,pnx,pnz])
    sigma_temp = np.zeros([cmpnum,pnx * px,pnz * pz])
        
    sigma_temp = np.tile(sigma_add.reshape(cmpnum, pnx, 1, pnz, 1), (1, 1, px, 1, pz)).reshape(cmpnum, pnx * px, pnz * pz)
    
    output = sigma_temp[:,:nx,:nt] + background
    
    return output

def GenerateCmpSigma_Single(pixelsize,ntrace,nt,sigma_b,sigma_cmp,Masknum):
    Masknum_max = np.max(Masknum)
    px,pz = pixelsize
    pnx,pnz = int(np.ceil(ntrace / px)),int(np.ceil(nt / pz))

    background = np.ones([ntrace,nt]) * sigma_b

    sigma_add = np.random.randint(0,sigma_cmp,[pnx,pnz])
    sigma_temp = np.zeros([pnx * px,pnz * pz])

    sigma_temp = np.tile(sigma_add.reshape(pnx, 1, pnz, 1), (1, px, 1, pz)).reshape(pnx * px, pnz * pz)
    
    output = sigma_temp[:ntrace,:nt] + background

    MaskModified = Masknum / Masknum_max

    output = output * MaskModified
    
    return output

def MakeReferenceVel(Vrms,ratio,k):
    Vref = np.zeros([Vrms.shape[1],2 * k + 1,Vrms.shape[0]])
    for i in range(Vrms.shape[1]):        
        for j in range(-k,k+1):
            Vref[i,j+k,:] = Vrms[:,i] * (1 + ratio * j)
    return Vref

def MakeConstantVel(Vel,velint,k,cmpnum):
    Vref = np.zeros([cmpnum,2 * k + 1,Vel.shape[0]])
    for i in range(cmpnum):
        for j in range(-k,k+1):
            Vref[i,j+k,:] = Vel + velint * j
    return Vref

def MakeReferenceVel_Single(Vrms,ratio,k):
    Vref = np.zeros([2 * k + 1,Vrms.shape[0]])       
    for j in range(-k,k+1):
        Vref[j+k,:] = Vrms * (1 + ratio * j)
    return Vref

def MakeConstantVel_Single(Vel,velint,k):
    Vref = np.zeros([2 * k + 1,Vel.shape[0]])
    for j in range(-k,k+1):
        Vref[j+k,:] = Vel + velint * j
    return Vref

def GetTransPointGPU(diff,cmpnum,ntrace,nt,maxtpnum):

    kernel_code = r"""
    
    __global__ void cuda_GetTransPoint(float *diff,float *transpoint,int cmpnum,int ntrace,int nt,int ntp)
    {
        int icmp = blockIdx.x;
        int itrace = threadIdx.x;
        
        int tpnum = 0;

        transpoint[icmp*ntrace*ntp + itrace*ntp + 1] = 0;

        for(int i = 0; i < nt-1; i++)
        {
            if(diff[icmp*ntrace*(nt-1)+itrace*(nt-1)+i] < -3.1415926)
            {
                transpoint[icmp*ntrace*ntp + itrace*ntp + tpnum + 1] = i + 1;
                tpnum++;
            }
        }

        transpoint[icmp*ntrace*ntp + itrace*ntp] = tpnum;
        
    }
    """

    mod = SourceModule(kernel_code)
    cuda_GetTransPoint = mod.get_function("cuda_GetTransPoint")
    
    ntp = maxtpnum + 2
    diff_gpu = gpuarray.to_gpu(diff.astype(np.float32))
    transpoint_gpu = gpuarray.zeros((cmpnum,ntrace,ntp),np.float32)

    block = (ntrace,1,1)
    grid = (cmpnum,1,1)

    cuda_GetTransPoint(diff_gpu,transpoint_gpu,np.int32(cmpnum),np.int32(ntrace),np.int32(nt),np.int32(ntp),block=block,grid=grid)

    transpoint = transpoint_gpu.get()
    
    diff_gpu.gpudata.free()
    transpoint_gpu.gpudata.free()
    
    return transpoint.astype(np.int32)

def NMOGPU(CMPGather,OffsetVec,Vrms,tVec,nt,dt,cmpnum,ntrace):
    # import pycuda.driver as cuda
    # import pycuda.autoinit
    # import pycuda.gpuarray as gpuarray
    # from pycuda.compiler import SourceModule

    kernel_code = r"""
    __device__ float device_cubic_spline4(float y0, float y1, float y2, float y3, float dt, float t, int type)
    {
        float alpha1, alpha2;
        float l1, l2;
        float mu0, mu1, mu2;
        float z0, z1, z2;
        float a0, a1, a2;
        float b0, b1, b2;
        float c0, c1, c2, c3;
        float d0, d1, d2;

        alpha1 = 3.0f / dt * (y2 - y1) - 3.0f / dt * (y1 - y0);
        alpha2 = 3.0f / dt * (y3 - y2) - 3.0f / dt * (y2 - y1);

        mu0 = 0.0f;
        z0 = 0.0f;

        l1 = 4.0f * dt - dt * mu0;
        mu1 = dt / l1;
        z1 = (alpha1 - dt * z0) / l1;

        l2 = 4.0f * dt - dt * mu1;
        mu2 = dt / l2;
        z2 = (alpha2 - dt * z1) / l2;

        c3 = 0.0f;

        c2 = z2 - mu2 * c3;
        b2 = (y3 - y2) / dt - dt * (c3 + 2.0f * c2) / 3.0f;
        d2 = (c3 - c2) / (3.0f * dt);
        a2 = y2;

        c1 = z1 - mu1 * c2;
        b1 = (y2 - y1) / dt - dt * (c2 + 2.0f * c1) / 3.0f;
        d1 = (c2 - c1) / (3.0f * dt);
        a1 = y1;

        c0 = z0 - mu0 * c1;
        b0 = (y1 - y0) / dt - dt * (c1 + 2.0f * c0) / 3.0f;
        d0 = (c1 - c0) / (3.0f * dt);
        a0 = y0;

        if(type == 1)
        {
            return a1 + b1 * (t - dt) + c1 * (t - dt) * (t - dt) + d1 * (t - dt) * (t - dt) * (t - dt);
        }
        else if(type == 0)
        {
            return a0 + b0 * t + c0 * t * t + d0 * t * t * t;
        }	
        else if(type == 2)
        {
            return a2 + b2 * (t - 2 * dt) + c2 * (t - 2 * dt) * (t - 2 * dt) + d2 * (t - 2 * dt) * (t - 2 * dt) * (t - 2 * dt);
        }	
    }

    __global__ void cuda_NMO(float *CMPGather,float *OffsetVec,float *tVec,
                                    float *NMOGather,float *Vrms, 
                                    int nt,float dt,int ntrace)
    {
        int icmp = blockIdx.x;
        int itrace = threadIdx.x;
        int it = blockIdx.y * blockDim.y + threadIdx.y;
        
        float TravelT, t0Index, ChangeS, CutC = 1.2;
        int tfloor, type;
        
        if(it >= 0 && it < nt)
        {
            TravelT = sqrt(tVec[it] * tVec[it] + 
                            (OffsetVec[icmp*ntrace+itrace] / Vrms[icmp*nt+it]) * 
                            (OffsetVec[icmp*ntrace+itrace] / Vrms[icmp*nt+it]));

            ChangeS = TravelT / (tVec[it] + 1);            
            t0Index = (TravelT - tVec[0]) / dt;
            tfloor = floorf(t0Index);

            if(t0Index >= 0 && t0Index < nt && ChangeS < CutC)
            {
                if(t0Index >=1 && t0Index < nt-2)
                {
                    type = 1;
                    NMOGather[icmp*ntrace*nt+itrace*nt+it] = device_cubic_spline4(CMPGather[icmp*ntrace*nt+itrace*nt+tfloor-1],
                                                                                CMPGather[icmp*ntrace*nt+itrace*nt+tfloor],
                                                                                CMPGather[icmp*ntrace*nt+itrace*nt+tfloor+1],
                                                                                CMPGather[icmp*ntrace*nt+itrace*nt+tfloor+2],
                                                                                dt,(t0Index-(tfloor-1))*dt,type);
                }
                else if(t0Index >= 0 && t0Index < 1)
                {
                    type = 0;
                    NMOGather[icmp*ntrace*nt+itrace*nt+it] = device_cubic_spline4(CMPGather[icmp*ntrace*nt+itrace*nt+tfloor],
                                                                                CMPGather[icmp*ntrace*nt+itrace*nt+tfloor],
                                                                                CMPGather[icmp*ntrace*nt+itrace*nt+tfloor+1],
                                                                                CMPGather[icmp*ntrace*nt+itrace*nt+tfloor+2],
                                                                                dt,(t0Index-(tfloor))*dt,type);
                }
                else if(t0Index >= nt-2 && t0Index < nt-1)
                {
                    type = 2;
                    NMOGather[icmp*ntrace*nt+itrace*nt+it] = device_cubic_spline4(CMPGather[icmp*ntrace*nt+itrace*nt+tfloor-2],
                                                                                CMPGather[icmp*ntrace*nt+itrace*nt+tfloor-1],
                                                                                CMPGather[icmp*ntrace*nt+itrace*nt+tfloor],
                                                                                CMPGather[icmp*ntrace*nt+itrace*nt+tfloor+1],
                                                                                dt,(t0Index-(tfloor-2))*dt,type);
                }
                else if(t0Index == nt-1 || t0Index == 0) 
                    NMOGather[icmp*ntrace*nt+itrace*nt+it] = CMPGather[icmp*ntrace*nt+itrace*nt+tfloor];
            }
        }           
    }
    """

    mod = SourceModule(kernel_code)
    cuda_NMO = mod.get_function("cuda_NMO")

    CMPGather_gpu = gpuarray.to_gpu(CMPGather.astype(np.float32) )
    OffsetVec_gpu = gpuarray.to_gpu(OffsetVec.astype(np.float32) )
    tVec_gpu = gpuarray.to_gpu(tVec.astype(np.float32) )
    Vrms_gpu = gpuarray.to_gpu(Vrms.astype(np.float32) )

    NMOGather_gpu = gpuarray.zeros((cmpnum,ntrace,nt),np.float32)

    BLOCK_SIZE = 16 
    block = (ntrace, BLOCK_SIZE, 1)
    grid = (cmpnum, int(nt/BLOCK_SIZE)+1, 1)

    cuda_NMO(CMPGather_gpu,OffsetVec_gpu,tVec_gpu,
                    NMOGather_gpu,Vrms_gpu,
                    np.int32(nt),np.float32(dt),np.int32(ntrace),
                    block=block,grid=grid)
    
    NMOGather = NMOGather_gpu.get()

    CMPGather_gpu.gpudata.free()
    OffsetVec_gpu.gpudata.free()
    tVec_gpu.gpudata.free()
    Vrms_gpu.gpudata.free()
    NMOGather_gpu.gpudata.free()
        
    return NMOGather

def NMOGPU_Single(CMPGather,OffsetVec,Vrms,tVec,nt,dt,ntrace):
    # import pycuda.driver as cuda
    # import pycuda.autoinit
    # import pycuda.gpuarray as gpuarray
    # from pycuda.compiler import SourceModule

    kernel_code = r"""
    __device__ float device_cubic_spline4(float y0, float y1, float y2, float y3, float dt, float t, int type)
    {
        float alpha1, alpha2;
        float l1, l2;
        float mu0, mu1, mu2;
        float z0, z1, z2;
        float a0, a1, a2;
        float b0, b1, b2;
        float c0, c1, c2, c3;
        float d0, d1, d2;

        alpha1 = 3.0f / dt * (y2 - y1) - 3.0f / dt * (y1 - y0);
        alpha2 = 3.0f / dt * (y3 - y2) - 3.0f / dt * (y2 - y1);

        mu0 = 0.0f;
        z0 = 0.0f;

        l1 = 4.0f * dt - dt * mu0;
        mu1 = dt / l1;
        z1 = (alpha1 - dt * z0) / l1;

        l2 = 4.0f * dt - dt * mu1;
        mu2 = dt / l2;
        z2 = (alpha2 - dt * z1) / l2;

        c3 = 0.0f;

        c2 = z2 - mu2 * c3;
        b2 = (y3 - y2) / dt - dt * (c3 + 2.0f * c2) / 3.0f;
        d2 = (c3 - c2) / (3.0f * dt);
        a2 = y2;

        c1 = z1 - mu1 * c2;
        b1 = (y2 - y1) / dt - dt * (c2 + 2.0f * c1) / 3.0f;
        d1 = (c2 - c1) / (3.0f * dt);
        a1 = y1;

        c0 = z0 - mu0 * c1;
        b0 = (y1 - y0) / dt - dt * (c1 + 2.0f * c0) / 3.0f;
        d0 = (c1 - c0) / (3.0f * dt);
        a0 = y0;

        if(type == 1)
        {
            return a1 + b1 * (t - dt) + c1 * (t - dt) * (t - dt) + d1 * (t - dt) * (t - dt) * (t - dt);
        }
        else if(type == 0)
        {
            return a0 + b0 * t + c0 * t * t + d0 * t * t * t;
        }	
        else if(type == 2)
        {
            return a2 + b2 * (t - 2 * dt) + c2 * (t - 2 * dt) * (t - 2 * dt) + d2 * (t - 2 * dt) * (t - 2 * dt) * (t - 2 * dt);
        }	
    }

    __global__ void cuda_NMO(float *CMPGather,float *OffsetVec,float *tVec,
                                    float *NMOGather,float *Vrms, 
                                    int nt,float dt,int ntrace)
    {       
        int itrace = blockIdx.x * blockDim.x + threadIdx.x;
        int it = blockIdx.y * blockDim.y + threadIdx.y;
        
        float TravelT, t0Index, ChangeS, CutC = 1.2;
        int tfloor, type;
        
        if(it >= 0 && it < nt)
        {
            TravelT = sqrt(tVec[it] * tVec[it] + 
                            (OffsetVec[itrace] / Vrms[it]) * 
                            (OffsetVec[itrace] / Vrms[it]));

            ChangeS = TravelT / (tVec[it] + 1);            
            t0Index = (TravelT - tVec[0]) / dt;
            tfloor = floorf(t0Index);

            if(t0Index >= 0 && t0Index < nt && ChangeS < CutC)
            {
                if(t0Index >=1 && t0Index < nt-2)
                {
                    type = 1;
                    NMOGather[itrace*nt+it] = device_cubic_spline4(CMPGather[itrace*nt+tfloor-1],
                                                                                CMPGather[itrace*nt+tfloor],
                                                                                CMPGather[itrace*nt+tfloor+1],
                                                                                CMPGather[itrace*nt+tfloor+2],
                                                                                dt,(t0Index-(tfloor-1))*dt,type);
                }
                else if(t0Index >= 0 && t0Index < 1)
                {
                    type = 0;
                    NMOGather[itrace*nt+it] = device_cubic_spline4(CMPGather[itrace*nt+tfloor],
                                                                                CMPGather[itrace*nt+tfloor],
                                                                                CMPGather[itrace*nt+tfloor+1],
                                                                                CMPGather[itrace*nt+tfloor+2],
                                                                                dt,(t0Index-(tfloor))*dt,type);
                }
                else if(t0Index >= nt-2 && t0Index < nt-1)
                {
                    type = 2;
                    NMOGather[itrace*nt+it] = device_cubic_spline4(CMPGather[itrace*nt+tfloor-2],
                                                                                CMPGather[itrace*nt+tfloor-1],
                                                                                CMPGather[itrace*nt+tfloor],
                                                                                CMPGather[itrace*nt+tfloor+1],
                                                                                dt,(t0Index-(tfloor-2))*dt,type);
                }
                else if(t0Index == nt-1 || t0Index == 0) 
                    NMOGather[itrace*nt+it] = CMPGather[itrace*nt+tfloor];
            }
        }           
    }
    """

    mod = SourceModule(kernel_code)
    cuda_NMO = mod.get_function("cuda_NMO")

    CMPGather_gpu = gpuarray.to_gpu(CMPGather.astype(np.float32) )
    OffsetVec_gpu = gpuarray.to_gpu(OffsetVec.astype(np.float32) )
    tVec_gpu = gpuarray.to_gpu(tVec.astype(np.float32) )
    Vrms_gpu = gpuarray.to_gpu(Vrms.astype(np.float32) )

    NMOGather_gpu = gpuarray.zeros((ntrace,nt),np.float32)

    BLOCK_SIZE = 16
    block = (BLOCK_SIZE, BLOCK_SIZE, 1)
    grid = (int(ntrace/BLOCK_SIZE)+1, int(nt/BLOCK_SIZE)+1, 1)

    cuda_NMO(CMPGather_gpu,OffsetVec_gpu,tVec_gpu,
                    NMOGather_gpu,Vrms_gpu,
                    np.int32(nt),np.float32(dt),np.int32(ntrace),
                    block=block,grid=grid)
    
    NMOGather = NMOGather_gpu.get()

    CMPGather_gpu.gpudata.free()
    OffsetVec_gpu.gpudata.free()
    tVec_gpu.gpudata.free()
    Vrms_gpu.gpudata.free()
    NMOGather_gpu.gpudata.free()
        
    return NMOGather

def invertNMOGPU(NMOGather,OffsetVec,Mask,Vrms,tVec,nt,dt,cmpnum,ntrace):
    
    kernel_code = r"""
    __device__ float device_cubic_spline4(float y0, float y1, float y2, float y3, float dt, float t, int type)
    {
        float alpha1, alpha2;
        float l1, l2;
        float mu0, mu1, mu2;
        float z0, z1, z2;
        float a0, a1, a2;
        float b0, b1, b2;
        float c0, c1, c2, c3;
        float d0, d1, d2;

        alpha1 = 3.0f / dt * (y2 - y1) - 3.0f / dt * (y1 - y0);
        alpha2 = 3.0f / dt * (y3 - y2) - 3.0f / dt * (y2 - y1);

        mu0 = 0.0f;
        z0 = 0.0f;

        l1 = 4.0f * dt - dt * mu0;
        mu1 = dt / l1;
        z1 = (alpha1 - dt * z0) / l1;

        l2 = 4.0f * dt - dt * mu1;
        mu2 = dt / l2;
        z2 = (alpha2 - dt * z1) / l2;

        c3 = 0.0f;

        c2 = z2 - mu2 * c3;
        b2 = (y3 - y2) / dt - dt * (c3 + 2.0f * c2) / 3.0f;
        d2 = (c3 - c2) / (3.0f * dt);
        a2 = y2;

        c1 = z1 - mu1 * c2;
        b1 = (y2 - y1) / dt - dt * (c2 + 2.0f * c1) / 3.0f;
        d1 = (c2 - c1) / (3.0f * dt);
        a1 = y1;

        c0 = z0 - mu0 * c1;
        b0 = (y1 - y0) / dt - dt * (c1 + 2.0f * c0) / 3.0f;
        d0 = (c1 - c0) / (3.0f * dt);
        a0 = y0;

        if(type == 1)
        {
            return a1 + b1 * (t - dt) + c1 * (t - dt) * (t - dt) + d1 * (t - dt) * (t - dt) * (t - dt);
        }
        else if(type == 0)
        {
            return a0 + b0 * t + c0 * t * t + d0 * t * t * t;
        }	
        else if(type == 2)
        {
            return a2 + b2 * (t - 2 * dt) + c2 * (t - 2 * dt) * (t - 2 * dt) + d2 * (t - 2 * dt) * (t - 2 * dt) * (t - 2 * dt);
        }	
    }

    __global__ void cuda_invertNMO(float *NMOGather,float *OffsetVec,float *tVec,
                                    float *iNMOGather,float *Vrms, 
                                    int nt,float dt,int ntrace)
    {
        int itrace = blockIdx.x * blockDim.x + threadIdx.x;
        int it = blockIdx.y * blockDim.y + threadIdx.y;
        
        float iTravelT, t0Index, Ttemp, Tsub;
        int tfloor, type;
        int ot, oIndex;
        
        if(it >= 0 && it < nt)
        {
            oIndex = 0;
            Tsub = nt;
            for(ot = 0; ot < nt; ot++)
            {
                Ttemp = sqrt(tVec[ot] * tVec[ot] +
                         (OffsetVec[itrace] / Vrms[ot]) * 
                         (OffsetVec[itrace] / Vrms[ot]));
                if(fabs(Ttemp - tVec[it]) < Tsub)
                {
                    Tsub = fabs(Ttemp - tVec[it]);
                    oIndex = ot;
                }
            }

            Ttemp = tVec[it] * tVec[it] -
                    (OffsetVec[itrace] / Vrms[oIndex]) *
                    (OffsetVec[itrace] / Vrms[oIndex]);

            if(Ttemp >= 0)
            {
                iTravelT = sqrt(Ttemp);
            }
            else
            {
                iTravelT = -1;
            }
          
            t0Index = (iTravelT - tVec[0]) / dt;
            tfloor = floorf(t0Index);

            if(t0Index >= 0 && t0Index < nt)
            {
                if(t0Index >=1 && t0Index < nt-2)
                {
                    type = 1;
                    iNMOGather[itrace*nt+it] = device_cubic_spline4(NMOGather[itrace*nt+tfloor-1],
                                                                                NMOGather[itrace*nt+tfloor],
                                                                                NMOGather[itrace*nt+tfloor+1],
                                                                                NMOGather[itrace*nt+tfloor+2],
                                                                                dt,(t0Index-(tfloor-1))*dt,type);
                    iMask[icmp*ntrace*nt+itrace*nt+it] = Mask[icmp*ntrace*nt+itrace*nt+tfloor];
                }
                else if(t0Index >= 0 && t0Index < 1)
                {
                    type = 0;
                    iNMOGather[icmp*ntrace*nt+itrace*nt+it] = device_cubic_spline4(NMOGather[icmp*ntrace*nt+itrace*nt+tfloor],
                                                                                NMOGather[icmp*ntrace*nt+itrace*nt+tfloor],
                                                                                NMOGather[icmp*ntrace*nt+itrace*nt+tfloor+1],
                                                                                NMOGather[icmp*ntrace*nt+itrace*nt+tfloor+2],
                                                                                dt,(t0Index-(tfloor))*dt,type);
                    iMask[icmp*ntrace*nt+itrace*nt+it] = Mask[icmp*ntrace*nt+itrace*nt+tfloor];
                }
                else if(t0Index >= nt-2 && t0Index < nt-1)
                {
                    type = 2;
                    iNMOGather[icmp*ntrace*nt+itrace*nt+it] = device_cubic_spline4(NMOGather[icmp*ntrace*nt+itrace*nt+tfloor-2],
                                                                                NMOGather[icmp*ntrace*nt+itrace*nt+tfloor-1],
                                                                                NMOGather[icmp*ntrace*nt+itrace*nt+tfloor],
                                                                                NMOGather[icmp*ntrace*nt+itrace*nt+tfloor+1],
                                                                                dt,(t0Index-(tfloor-2))*dt,type);
                    iMask[icmp*ntrace*nt+itrace*nt+it] = Mask[icmp*ntrace*nt+itrace*nt+tfloor];
                }
                else if(t0Index == nt-1 || t0Index == 0)
                {
                    iNMOGather[icmp*ntrace*nt+itrace*nt+it] = NMOGather[icmp*ntrace*nt+itrace*nt+tfloor];
                    iMask[icmp*ntrace*nt+itrace*nt+it] = Mask[icmp*ntrace*nt+itrace*nt+tfloor];
                }
            }
        }
    }
    """
            
    mod = SourceModule(kernel_code)
    cuda_invertNMO = mod.get_function("cuda_invertNMO")

    NMOGather_gpu = gpuarray.to_gpu(NMOGather.astype(np.float32) )
    OffsetVec_gpu = gpuarray.to_gpu(OffsetVec.astype(np.float32) )
    tVec_gpu = gpuarray.to_gpu(tVec.astype(np.float32) )
    Mask_gpu = gpuarray.to_gpu(Mask.astype(np.float32) )
    Vrms_gpu = gpuarray.to_gpu(Vrms.astype(np.float32) )

    iNMOGather_gpu = gpuarray.zeros((cmpnum,ntrace,nt),np.float32)
    iMask_gpu = gpuarray.zeros((cmpnum,ntrace,nt),np.float32) 

    BLOCK_SIZE = 16
    block = (BLOCK_SIZE, BLOCK_SIZE, 1)
    grid = (int(ntrace/BLOCK_SIZE)+1, int(nt/BLOCK_SIZE)+1, 1)

    cuda_invertNMO(NMOGather_gpu,Mask_gpu,OffsetVec_gpu,tVec_gpu,
                    iNMOGather_gpu,iMask_gpu,Vrms_gpu,
                    np.int32(nt),np.float32(dt),np.int32(ntrace),
                    block=block,grid=grid)
    
    iNMOGather = iNMOGather_gpu.get()
    iMask = iMask_gpu.get()

    NMOGather_gpu.gpudata.free()
    OffsetVec_gpu.gpudata.free()
    tVec_gpu.gpudata.free()
    Mask_gpu.gpudata.free()
    Vrms_gpu.gpudata.free()
    iNMOGather_gpu.gpudata.free()
    iMask_gpu.gpudata.free()    

    return iNMOGather,iMask

def invertNMOGPU_VarVel(NMOGather,OffsetVec,Mask,Vrms,tVec,nt,dt,cmpnum,ntrace):
    
    kernel_code = r"""
    __device__ float device_cubic_spline4(float y0, float y1, float y2, float y3, float dt, float t, int type)
    {
        float alpha1, alpha2;
        float l1, l2;
        float mu0, mu1, mu2;
        float z0, z1, z2;
        float a0, a1, a2;
        float b0, b1, b2;
        float c0, c1, c2, c3;
        float d0, d1, d2;

        alpha1 = 3.0f / dt * (y2 - y1) - 3.0f / dt * (y1 - y0);
        alpha2 = 3.0f / dt * (y3 - y2) - 3.0f / dt * (y2 - y1);

        mu0 = 0.0f;
        z0 = 0.0f;

        l1 = 4.0f * dt - dt * mu0;
        mu1 = dt / l1;
        z1 = (alpha1 - dt * z0) / l1;

        l2 = 4.0f * dt - dt * mu1;
        mu2 = dt / l2;
        z2 = (alpha2 - dt * z1) / l2;

        c3 = 0.0f;

        c2 = z2 - mu2 * c3;
        b2 = (y3 - y2) / dt - dt * (c3 + 2.0f * c2) / 3.0f;
        d2 = (c3 - c2) / (3.0f * dt);
        a2 = y2;

        c1 = z1 - mu1 * c2;
        b1 = (y2 - y1) / dt - dt * (c2 + 2.0f * c1) / 3.0f;
        d1 = (c2 - c1) / (3.0f * dt);
        a1 = y1;

        c0 = z0 - mu0 * c1;
        b0 = (y1 - y0) / dt - dt * (c1 + 2.0f * c0) / 3.0f;
        d0 = (c1 - c0) / (3.0f * dt);
        a0 = y0;

        if(type == 1)
        {
            return a1 + b1 * (t - dt) + c1 * (t - dt) * (t - dt) + d1 * (t - dt) * (t - dt) * (t - dt);
        }
        else if(type == 0)
        {
            return a0 + b0 * t + c0 * t * t + d0 * t * t * t;
        }	
        else if(type == 2)
        {
            return a2 + b2 * (t - 2 * dt) + c2 * (t - 2 * dt) * (t - 2 * dt) + d2 * (t - 2 * dt) * (t - 2 * dt) * (t - 2 * dt);
        }	
    }

    __global__ void cuda_invertNMO(float *NMOGather,float *Mask,float *OffsetVec,float *tVec,
                                    float *iNMOGather,float *iMask,float *Vrms, 
                                    int nt,float dt,int ntrace)
    {
        int icmp = blockIdx.x;
        int itrace = threadIdx.x;
        int it = blockIdx.y * blockDim.y + threadIdx.y;
        
        float iTravelT, t0Index, Ttemp, Tsub;
        int tfloor, type;
        int ot, oIndex;
        
        if(it >= 0 && it < nt)
        {
            oIndex = 0;
            Tsub = nt;
            for(ot = 0; ot < nt; ot++)
            {
                Ttemp = sqrt(tVec[ot] * tVec[ot] +
                         (OffsetVec[icmp*ntrace+itrace] / Vrms[icmp*ntrace*nt+itrace*nt+ot]) * 
                         (OffsetVec[icmp*ntrace+itrace] / Vrms[icmp*ntrace*nt+itrace*nt+ot]));
                if(fabs(Ttemp - tVec[it]) < Tsub)
                {
                    Tsub = fabs(Ttemp - tVec[it]);
                    oIndex = ot;
                }
            }

            Ttemp = tVec[it] * tVec[it] -
                    (OffsetVec[icmp*ntrace+itrace] / Vrms[icmp*ntrace*nt+itrace*nt+oIndex]) *
                    (OffsetVec[icmp*ntrace+itrace] / Vrms[icmp*ntrace*nt+itrace*nt+oIndex]);

            if(Ttemp >= 0)
            {
                iTravelT = sqrt(Ttemp);
            }
            else
            {
                iTravelT = -1;
            }
          
            t0Index = (iTravelT - tVec[0]) / dt;
            tfloor = floorf(t0Index);

            if(t0Index >= 0 && t0Index < nt)
            {
                if(t0Index >=1 && t0Index < nt-2)
                {
                    type = 1;
                    iNMOGather[icmp*ntrace*nt+itrace*nt+it] = device_cubic_spline4(NMOGather[icmp*ntrace*nt+itrace*nt+tfloor-1],
                                                                                NMOGather[icmp*ntrace*nt+itrace*nt+tfloor],
                                                                                NMOGather[icmp*ntrace*nt+itrace*nt+tfloor+1],
                                                                                NMOGather[icmp*ntrace*nt+itrace*nt+tfloor+2],
                                                                                dt,(t0Index-(tfloor-1))*dt,type);
                    iMask[icmp*ntrace*nt+itrace*nt+it] = Mask[icmp*ntrace*nt+itrace*nt+tfloor];
                }
                else if(t0Index >= 0 && t0Index < 1)
                {
                    type = 0;
                    iNMOGather[icmp*ntrace*nt+itrace*nt+it] = device_cubic_spline4(NMOGather[icmp*ntrace*nt+itrace*nt+tfloor],
                                                                                NMOGather[icmp*ntrace*nt+itrace*nt+tfloor],
                                                                                NMOGather[icmp*ntrace*nt+itrace*nt+tfloor+1],
                                                                                NMOGather[icmp*ntrace*nt+itrace*nt+tfloor+2],
                                                                                dt,(t0Index-(tfloor))*dt,type);
                    iMask[icmp*ntrace*nt+itrace*nt+it] = Mask[icmp*ntrace*nt+itrace*nt+tfloor];
                }
                else if(t0Index >= nt-2 && t0Index < nt-1)
                {
                    type = 2;
                    iNMOGather[icmp*ntrace*nt+itrace*nt+it] = device_cubic_spline4(NMOGather[icmp*ntrace*nt+itrace*nt+tfloor-2],
                                                                                NMOGather[icmp*ntrace*nt+itrace*nt+tfloor-1],
                                                                                NMOGather[icmp*ntrace*nt+itrace*nt+tfloor],
                                                                                NMOGather[icmp*ntrace*nt+itrace*nt+tfloor+1],
                                                                                dt,(t0Index-(tfloor-2))*dt,type);
                    iMask[icmp*ntrace*nt+itrace*nt+it] = Mask[icmp*ntrace*nt+itrace*nt+tfloor];
                }
                else if(t0Index == nt-1 || t0Index == 0)
                {
                    iNMOGather[icmp*ntrace*nt+itrace*nt+it] = NMOGather[icmp*ntrace*nt+itrace*nt+tfloor];
                    iMask[icmp*ntrace*nt+itrace*nt+it] = Mask[icmp*ntrace*nt+itrace*nt+tfloor];
                }
            }
        }
    }
    """
            
    mod = SourceModule(kernel_code)
    cuda_invertNMO = mod.get_function("cuda_invertNMO")

    NMOGather_gpu = gpuarray.to_gpu(NMOGather.astype(np.float32) )
    OffsetVec_gpu = gpuarray.to_gpu(OffsetVec.astype(np.float32) )
    tVec_gpu = gpuarray.to_gpu(tVec.astype(np.float32) )
    Mask_gpu = gpuarray.to_gpu(Mask.astype(np.float32) )
    Vrms_gpu = gpuarray.to_gpu(Vrms.astype(np.float32) )

    iNMOGather_gpu = gpuarray.zeros((cmpnum,ntrace,nt),np.float32)
    iMask_gpu = gpuarray.zeros((cmpnum,ntrace,nt),np.float32) 

    BLOCK_SIZE = 16 
    block = (ntrace, BLOCK_SIZE, 1)
    grid = (cmpnum, int(nt/BLOCK_SIZE)+1, 1)

    cuda_invertNMO(NMOGather_gpu,Mask_gpu,OffsetVec_gpu,tVec_gpu,
                    iNMOGather_gpu,iMask_gpu,Vrms_gpu,
                    np.int32(nt),np.float32(dt),np.int32(ntrace),
                    block=block,grid=grid)
    
    iNMOGather = iNMOGather_gpu.get()
    iMask = iMask_gpu.get()

    NMOGather_gpu.gpudata.free()
    OffsetVec_gpu.gpudata.free()
    tVec_gpu.gpudata.free()
    Mask_gpu.gpudata.free()
    Vrms_gpu.gpudata.free()
    iNMOGather_gpu.gpudata.free()
    iMask_gpu.gpudata.free()    

    return iNMOGather,iMask

def GenerateSemblanceGPU(CMPGather,OffsetVec,Mask,tVec,nt,dt,cmpnum,ntrace,nv,dv,v0,nb):

    kernel_code = r"""
    __device__ float device_cubic_spline4(float y0, float y1, float y2, float y3, float dt, float t, int type)
    {
        float alpha1, alpha2;
        float l1, l2;
        float mu0, mu1, mu2;
        float z0, z1, z2;
        float a0, a1, a2;
        float b0, b1, b2;
        float c0, c1, c2, c3;
        float d0, d1, d2;

        alpha1 = 3.0f / dt * (y2 - y1) - 3.0f / dt * (y1 - y0);
        alpha2 = 3.0f / dt * (y3 - y2) - 3.0f / dt * (y2 - y1);

        mu0 = 0.0f;
        z0 = 0.0f;

        l1 = 4.0f * dt - dt * mu0;
        mu1 = dt / l1;
        z1 = (alpha1 - dt * z0) / l1;

        l2 = 4.0f * dt - dt * mu1;
        mu2 = dt / l2;
        z2 = (alpha2 - dt * z1) / l2;

        c3 = 0.0f;

        c2 = z2 - mu2 * c3;
        b2 = (y3 - y2) / dt - dt * (c3 + 2.0f * c2) / 3.0f;
        d2 = (c3 - c2) / (3.0f * dt);
        a2 = y2;

        c1 = z1 - mu1 * c2;
        b1 = (y2 - y1) / dt - dt * (c2 + 2.0f * c1) / 3.0f;
        d1 = (c2 - c1) / (3.0f * dt);
        a1 = y1;

        c0 = z0 - mu0 * c1;
        b0 = (y1 - y0) / dt - dt * (c1 + 2.0f * c0) / 3.0f;
        d0 = (c1 - c0) / (3.0f * dt);
        a0 = y0;

        if(type == 1)
        {
            return a1 + b1 * (t - dt) + c1 * (t - dt) * (t - dt) + d1 * (t - dt) * (t - dt) * (t - dt);
        }
        else if(type == 0)
        {
            return a0 + b0 * t + c0 * t * t + d0 * t * t * t;
        }	
        else if(type == 2)
        {
            return a2 + b2 * (t - 2 * dt) + c2 * (t - 2 * dt) * (t - 2 * dt) + d2 * (t - 2 * dt) * (t - 2 * dt) * (t - 2 * dt);
        }	
    }
    
    __global__ void cuda_GenerateSemblance(float *CMPGather,float *Mask,float *OffsetVec,float *tVec,
                                        float *temp1,float *temp2,float *Semblance,
                                        int nt,float dt,int ntrace,int nv,float dv,float v0,int nb)
    {
        int icmp = blockIdx.x;
        int iv = threadIdx.x;
        int it = blockIdx.y * blockDim.y + threadIdx.y;
        
        float TravelT, ChangeS, CutC = 1.2, nmoTemp, t0Index;
        int sn = 0, maskTemp;
        float v = v0 + iv * dv;
        float num = 0, den = 0;
        int ib = 0, ie = 0, tfloor, type;
        
        if(it >= 0 && it < nt)
        {
            for(int itrace = 0;itrace < ntrace;itrace++)
            {                
                TravelT = sqrt(tVec[it] * tVec[it] + 
                                (OffsetVec[icmp*ntrace+itrace] / v) * 
                                (OffsetVec[icmp*ntrace+itrace] / v));

                ChangeS = TravelT / (tVec[it] + 1);            
                t0Index = (TravelT - tVec[0]) / dt;
                tfloor = floorf(t0Index);

                if(t0Index >= 0 && t0Index < nt && ChangeS < CutC)
                {
                    if(t0Index >= 1 && t0Index < nt-2)
                    {
                        type = 1;
                        nmoTemp = device_cubic_spline4(CMPGather[icmp*ntrace*nt+itrace*nt+tfloor-1],
                                                                CMPGather[icmp*ntrace*nt+itrace*nt+tfloor],
                                                                CMPGather[icmp*ntrace*nt+itrace*nt+tfloor+1],
                                                                CMPGather[icmp*ntrace*nt+itrace*nt+tfloor+2],
                                                                dt,(t0Index-(tfloor-1))*dt,type);
                    }
                    else if(t0Index >= 0 && t0Index < 1)
                    {
                        type = 0;
                        nmoTemp = device_cubic_spline4(CMPGather[icmp*ntrace*nt+itrace*nt+tfloor],
                                                                CMPGather[icmp*ntrace*nt+itrace*nt+tfloor],
                                                                CMPGather[icmp*ntrace*nt+itrace*nt+tfloor+1],
                                                                CMPGather[icmp*ntrace*nt+itrace*nt+tfloor+2],
                                                                dt,(t0Index-(tfloor))*dt,type);
                    }
                    else if(t0Index >= nt-2 && t0Index < nt-1)
                    {
                        type = 2;
                        nmoTemp = device_cubic_spline4(CMPGather[icmp*ntrace*nt+itrace*nt+tfloor-2],
                                                                CMPGather[icmp*ntrace*nt+itrace*nt+tfloor-1],
                                                                CMPGather[icmp*ntrace*nt+itrace*nt+tfloor],
                                                                CMPGather[icmp*ntrace*nt+itrace*nt+tfloor+1],
                                                                dt,(t0Index-(tfloor-2))*dt,type);
                    }
                    else if(t0Index == nt-1 || t0Index == 0) 
                    {
                        nmoTemp = CMPGather[icmp*ntrace*nt+itrace*nt+tfloor];
                    }
                    
                    maskTemp = Mask[icmp*ntrace*nt+itrace*nt+int(t0Index)];
                }
                else
                {
                    nmoTemp = 0;
                    maskTemp = 0;
                }
            
                temp1[icmp*nv*nt+iv*nt+it] += nmoTemp * maskTemp;
                temp2[icmp*nv*nt+iv*nt+it] += nmoTemp * nmoTemp * maskTemp;

                if(maskTemp != 0)
                    sn++;
            }

            
            ib = it - nb;
            ie = it + nb + 1;

            if(ib < 0)
                ib = 0;
            if(ie > nt)
                ie = nt;

            for(int itt = ib;itt < ie;itt++)
            {
                num += temp1[icmp*nv*nt+iv*nt+itt] * temp1[icmp*nv*nt+iv*nt+itt];
                den += temp2[icmp*nv*nt+iv*nt+itt];
            }
            den /= ntrace;
            
            Semblance[icmp*nv*nt+iv*nt+it] = (den > 0.)? num/den: 0.;
        }           
    }
    """

    mod = SourceModule(kernel_code)
    cuda_GenerateSemblance = mod.get_function("cuda_GenerateSemblance")

    Mem = CMPGather.nbytes+ OffsetVec.nbytes + tVec.nbytes + Mask.nbytes
    Mem = Mem + cmpnum*nv*nt*4*3
    Mem = Mem/1024/1024/1024

    mem_free, mem_total = cuda.mem_get_info()

    if Mem > mem_free/1024/1024/1024 * 0.9:
        calnum = int(np.ceil(Mem/(mem_free/1024/1024/1024*0.9)))
        cmpeach = int(np.ceil(cmpnum/calnum))
    else:
        calnum = 1
        cmpeach = cmpnum

    Semblance = np.zeros((cmpnum,nv,nt),np.float32)

    for i in range(int(calnum)):
        if i == calnum-1:
            cmpend = cmpnum
        else:
            cmpend = (i+1)*cmpeach
        CMPGather_gpu = gpuarray.to_gpu(CMPGather[i*cmpeach:cmpend].astype(np.float32) )
        OffsetVec_gpu = gpuarray.to_gpu(OffsetVec[i*cmpeach:cmpend].astype(np.float32) )
        tVec_gpu = gpuarray.to_gpu(tVec.astype(np.float32) )
        Mask_gpu = gpuarray.to_gpu(Mask[i*cmpeach:cmpend].astype(np.float32) )

        Semblance_gpu = gpuarray.zeros((cmpend-i*cmpeach,nv,nt),np.float32)
        temp1_gpu = gpuarray.zeros((cmpend-i*cmpeach,nv,nt),np.float32)
        temp2_gpu = gpuarray.zeros((cmpend-i*cmpeach,nv,nt),np.float32)

        BLOCK_SIZE = 4
        block = (nv, BLOCK_SIZE, 1)
        grid = (cmpend-i*cmpeach, int(nt/BLOCK_SIZE)+1, 1)

        cuda_GenerateSemblance(CMPGather_gpu,Mask_gpu,OffsetVec_gpu,tVec_gpu,temp1_gpu,temp2_gpu,Semblance_gpu,
                                    np.int32(nt),np.float32(dt),np.int32(ntrace),
                                    np.int32(nv),np.float32(dv),np.float32(v0),np.int32(nb),block=block,grid=grid)

        Semblance[i*cmpeach:cmpend] = Semblance_gpu.get()
        
        CMPGather_gpu.gpudata.free()
        OffsetVec_gpu.gpudata.free()
        tVec_gpu.gpudata.free()
        Mask_gpu.gpudata.free()
        temp1_gpu.gpudata.free()
        temp2_gpu.gpudata.free()
        Semblance_gpu.gpudata.free()        
    
    return Semblance

def GenerateSemblanceGPU_Single(CMPGather,OffsetVec,Mask,tVec,nt,dt,ntrace,nv,dv,v0,nb):

    kernel_code = r"""
    __device__ float device_cubic_spline4(float y0, float y1, float y2, float y3, float dt, float t, int type)
    {
        float alpha1, alpha2;
        float l1, l2;
        float mu0, mu1, mu2;
        float z0, z1, z2;
        float a0, a1, a2;
        float b0, b1, b2;
        float c0, c1, c2, c3;
        float d0, d1, d2;

        alpha1 = 3.0f / dt * (y2 - y1) - 3.0f / dt * (y1 - y0);
        alpha2 = 3.0f / dt * (y3 - y2) - 3.0f / dt * (y2 - y1);

        mu0 = 0.0f;
        z0 = 0.0f;

        l1 = 4.0f * dt - dt * mu0;
        mu1 = dt / l1;
        z1 = (alpha1 - dt * z0) / l1;

        l2 = 4.0f * dt - dt * mu1;
        mu2 = dt / l2;
        z2 = (alpha2 - dt * z1) / l2;

        c3 = 0.0f;

        c2 = z2 - mu2 * c3;
        b2 = (y3 - y2) / dt - dt * (c3 + 2.0f * c2) / 3.0f;
        d2 = (c3 - c2) / (3.0f * dt);
        a2 = y2;

        c1 = z1 - mu1 * c2;
        b1 = (y2 - y1) / dt - dt * (c2 + 2.0f * c1) / 3.0f;
        d1 = (c2 - c1) / (3.0f * dt);
        a1 = y1;

        c0 = z0 - mu0 * c1;
        b0 = (y1 - y0) / dt - dt * (c1 + 2.0f * c0) / 3.0f;
        d0 = (c1 - c0) / (3.0f * dt);
        a0 = y0;

        if(type == 1)
        {
            return a1 + b1 * (t - dt) + c1 * (t - dt) * (t - dt) + d1 * (t - dt) * (t - dt) * (t - dt);
        }
        else if(type == 0)
        {
            return a0 + b0 * t + c0 * t * t + d0 * t * t * t;
        }	
        else if(type == 2)
        {
            return a2 + b2 * (t - 2 * dt) + c2 * (t - 2 * dt) * (t - 2 * dt) + d2 * (t - 2 * dt) * (t - 2 * dt) * (t - 2 * dt);
        }	
    }
    
    __global__ void cuda_GenerateTemp(float *CMPGather,float *Mask,float *OffsetVec,float *tVec,
                                        float *temp1,float *temp2,
                                        int nt,float dt,int ntrace,int nv,float dv,float v0)
    {        
        int iv = blockIdx.x * blockDim.x + threadIdx.x;
        int it = blockIdx.y * blockDim.y + threadIdx.y;
        
        float TravelT, ChangeS, CutC = 1.2, nmoTemp, t0Index;
        int sn = 0, maskTemp;
        float v = v0 + iv * dv;        
        int tfloor, type;
        
        if(it >= 0 && it < nt)
        {
            if(iv >= 0 && iv < nv)
            {
                for(int itrace = 0;itrace < ntrace;itrace++)
                {                
                    TravelT = sqrt(tVec[it] * tVec[it] + 
                                    (OffsetVec[itrace] / v) * 
                                    (OffsetVec[itrace] / v));

                    ChangeS = TravelT / (tVec[it] + 1);            
                    t0Index = (TravelT - tVec[0]) / dt;
                    tfloor = floorf(t0Index);

                    if(t0Index >= 0 && t0Index < nt && ChangeS < CutC)
                    {
                        if(t0Index >= 1 && t0Index < nt-2)
                        {
                            type = 1;
                            nmoTemp = device_cubic_spline4(CMPGather[itrace*nt+tfloor-1],
                                                                    CMPGather[itrace*nt+tfloor],
                                                                    CMPGather[itrace*nt+tfloor+1],
                                                                    CMPGather[itrace*nt+tfloor+2],
                                                                    dt,(t0Index-(tfloor-1))*dt,type);
                        }
                        else if(t0Index >= 0 && t0Index < 1)
                        {
                            type = 0;
                            nmoTemp = device_cubic_spline4(CMPGather[itrace*nt+tfloor],
                                                                    CMPGather[itrace*nt+tfloor],
                                                                    CMPGather[itrace*nt+tfloor+1],
                                                                    CMPGather[itrace*nt+tfloor+2],
                                                                    dt,(t0Index-(tfloor))*dt,type);
                        }
                        else if(t0Index >= nt-2 && t0Index < nt-1)
                        {
                            type = 2;
                            nmoTemp = device_cubic_spline4(CMPGather[itrace*nt+tfloor-2],
                                                                    CMPGather[itrace*nt+tfloor-1],
                                                                    CMPGather[itrace*nt+tfloor],
                                                                    CMPGather[itrace*nt+tfloor+1],
                                                                    dt,(t0Index-(tfloor-2))*dt,type);
                        }
                        else if(t0Index == nt-1 || t0Index == 0) 
                        {
                            nmoTemp = CMPGather[itrace*nt+tfloor];
                        }
                        else
                        {
                            nmoTemp = 0;
                        }
                        
                        maskTemp = Mask[itrace*nt+int(t0Index)];

                    }
                    else
                    {
                        nmoTemp = 0;
                        maskTemp = 0;
                    }
                
                    temp1[iv*nt+it] += nmoTemp * maskTemp;
                    temp2[iv*nt+it] += nmoTemp * nmoTemp * maskTemp;

                    if(maskTemp != 0)
                        sn++;
                }
            }
        }           
    }

    __global__ void cuda_GenerateSemblance(float *temp1,float *temp2,float *Semblance,int nt,int ntrace,int nv,int nb)
    {        
        int iv = blockIdx.x * blockDim.x + threadIdx.x;
        int it = blockIdx.y * blockDim.y + threadIdx.y;
                       
        float num = 0, den = 0;
        int ib = 0, ie = 0;
        
        if(it >= 0 && it < nt)
        {
            if(iv >= 0 && iv < nv)
            {
                ib = it - nb;
                ie = it + nb + 1;

                if(ib < 0)
                    ib = 0;
                if(ie > nt)
                    ie = nt;

                for(int itt = ib;itt < ie;itt++)
                {
                    num += temp1[iv*nt+itt] * temp1[iv*nt+itt];
                    den += temp2[iv*nt+itt];
                }

                den /= ntrace;

                Semblance[iv*nt+it] = (den > 0.)? num/den: 0.;
            }
        } 
    }           
    """

    mod = SourceModule(kernel_code)
    cuda_GenerateTemp = mod.get_function("cuda_GenerateTemp")
    cuda_GenerateSemblance = mod.get_function("cuda_GenerateSemblance")

    Semblance = np.zeros((nv,nt),np.float32)
    CMPGather_gpu = gpuarray.to_gpu(CMPGather.astype(np.float32) )
    OffsetVec_gpu = gpuarray.to_gpu(OffsetVec.astype(np.float32) )
    tVec_gpu = gpuarray.to_gpu(tVec.astype(np.float32) )
    Mask_gpu = gpuarray.to_gpu(Mask.astype(np.float32) )

    Semblance_gpu = gpuarray.zeros((nv,nt),np.float32)
    temp1_gpu = gpuarray.zeros((nv,nt),np.float32)
    temp2_gpu = gpuarray.zeros((nv,nt),np.float32)

    BLOCK_SIZE = 16
    block = (BLOCK_SIZE, BLOCK_SIZE, 1)
    grid = (int(nv/BLOCK_SIZE)+1, int(nt/BLOCK_SIZE)+1, 1)

    cuda_GenerateTemp(CMPGather_gpu,Mask_gpu,OffsetVec_gpu,tVec_gpu,temp1_gpu,temp2_gpu,
                            np.int32(nt),np.float32(dt),np.int32(ntrace),
                            np.int32(nv),np.float32(dv),np.float32(v0),block=block,grid=grid)
    

    cuda_GenerateSemblance(temp1_gpu,temp2_gpu,Semblance_gpu,
                            np.int32(nt),np.int32(ntrace),np.int32(nv),np.int32(nb),block=block,grid=grid)

    Semblance = Semblance_gpu.get()
    
    CMPGather_gpu.gpudata.free()
    OffsetVec_gpu.gpudata.free()
    tVec_gpu.gpudata.free()
    Mask_gpu.gpudata.free()
    temp1_gpu.gpudata.free()
    temp2_gpu.gpudata.free()
    Semblance_gpu.gpudata.free()        
    
    return Semblance


def GenerateSemblanceNormalGPU(CMPGather,OffsetVec,Mask,tVec,nt,dt,cmpnum,ntrace,ctrace,nv,dv,v0,nb):

    kernel_code = r"""
    __device__ float device_cubic_spline4(float y0, float y1, float y2, float y3, float dt, float t, int type)
    {
        float alpha1, alpha2;
        float l1, l2;
        float mu0, mu1, mu2;
        float z0, z1, z2;
        float a0, a1, a2;
        float b0, b1, b2;
        float c0, c1, c2, c3;
        float d0, d1, d2;

        alpha1 = 3.0f / dt * (y2 - y1) - 3.0f / dt * (y1 - y0);
        alpha2 = 3.0f / dt * (y3 - y2) - 3.0f / dt * (y2 - y1);

        mu0 = 0.0f;
        z0 = 0.0f;

        l1 = 4.0f * dt - dt * mu0;
        mu1 = dt / l1;
        z1 = (alpha1 - dt * z0) / l1;

        l2 = 4.0f * dt - dt * mu1;
        mu2 = dt / l2;
        z2 = (alpha2 - dt * z1) / l2;

        c3 = 0.0f;

        c2 = z2 - mu2 * c3;
        b2 = (y3 - y2) / dt - dt * (c3 + 2.0f * c2) / 3.0f;
        d2 = (c3 - c2) / (3.0f * dt);
        a2 = y2;

        c1 = z1 - mu1 * c2;
        b1 = (y2 - y1) / dt - dt * (c2 + 2.0f * c1) / 3.0f;
        d1 = (c2 - c1) / (3.0f * dt);
        a1 = y1;

        c0 = z0 - mu0 * c1;
        b0 = (y1 - y0) / dt - dt * (c1 + 2.0f * c0) / 3.0f;
        d0 = (c1 - c0) / (3.0f * dt);
        a0 = y0;

        if(type == 1)
        {
            return a1 + b1 * (t - dt) + c1 * (t - dt) * (t - dt) + d1 * (t - dt) * (t - dt) * (t - dt);
        }
        else if(type == 0)
        {
            return a0 + b0 * t + c0 * t * t + d0 * t * t * t;
        }	
        else if(type == 2)
        {
            return a2 + b2 * (t - 2 * dt) + c2 * (t - 2 * dt) * (t - 2 * dt) + d2 * (t - 2 * dt) * (t - 2 * dt) * (t - 2 * dt);
        }	
    }
    
    __global__ void cuda_GenerateSemblance(float *CMPGather,float *Mask,float *OffsetVec,float *tVec,
                                        float *temp1,float *temp2,float *Semblance,
                                        int nt,float dt,int ntrace,int ctrace,int nv,float dv,float v0,int nb)
    {
        int icmp = blockIdx.x;
        int iv = threadIdx.x;
        int it = blockIdx.y * blockDim.y + threadIdx.y;
        
        float TravelT, ChangeS, CutC = 1.2, nmoTemp, t0Index;
        int sn = 0, maskTemp;
        float v = v0 + iv * dv;
        float num = 0, den = 0;
        int ib = 0, ie = 0, tfloor, type;
        
        if(it >= 0 && it < nt)
        {
            for(int itrace = 0;itrace < ctrace;itrace++)
            {                
                TravelT = sqrt(tVec[it] * tVec[it] + 
                                (OffsetVec[icmp*ntrace+itrace] / v) * 
                                (OffsetVec[icmp*ntrace+itrace] / v));

                ChangeS = TravelT / (tVec[it] + 1);            
                t0Index = (TravelT - tVec[0]) / dt;
                tfloor = floorf(t0Index);

                if(t0Index >= 0 && t0Index < nt && ChangeS < CutC)
                {
                    if(t0Index >= 1 && t0Index < nt-2)
                    {
                        type = 1;
                        nmoTemp = device_cubic_spline4(CMPGather[icmp*ntrace*nt+itrace*nt+tfloor-1],
                                                                CMPGather[icmp*ntrace*nt+itrace*nt+tfloor],
                                                                CMPGather[icmp*ntrace*nt+itrace*nt+tfloor+1],
                                                                CMPGather[icmp*ntrace*nt+itrace*nt+tfloor+2],
                                                                dt,(t0Index-(tfloor-1))*dt,type);
                    }
                    else if(t0Index >= 0 && t0Index < 1)
                    {
                        type = 0;
                        nmoTemp = device_cubic_spline4(CMPGather[icmp*ntrace*nt+itrace*nt+tfloor],
                                                                CMPGather[icmp*ntrace*nt+itrace*nt+tfloor],
                                                                CMPGather[icmp*ntrace*nt+itrace*nt+tfloor+1],
                                                                CMPGather[icmp*ntrace*nt+itrace*nt+tfloor+2],
                                                                dt,(t0Index-(tfloor))*dt,type);
                    }
                    else if(t0Index >= nt-2 && t0Index < nt-1)
                    {
                        type = 2;
                        nmoTemp = device_cubic_spline4(CMPGather[icmp*ntrace*nt+itrace*nt+tfloor-2],
                                                                CMPGather[icmp*ntrace*nt+itrace*nt+tfloor-1],
                                                                CMPGather[icmp*ntrace*nt+itrace*nt+tfloor],
                                                                CMPGather[icmp*ntrace*nt+itrace*nt+tfloor+1],
                                                                dt,(t0Index-(tfloor-2))*dt,type);
                    }
                    else if(t0Index == nt-1 || t0Index == 0) 
                    {
                        nmoTemp = CMPGather[icmp*ntrace*nt+itrace*nt+tfloor];
                    }
                    
                    maskTemp = Mask[icmp*ntrace*nt+itrace*nt+int(t0Index)];
                }
                else
                {
                    nmoTemp = 0;
                    maskTemp = 0;
                }
            
                temp1[icmp*nv*nt+iv*nt+it] += nmoTemp * maskTemp;
                temp2[icmp*nv*nt+iv*nt+it] += nmoTemp * nmoTemp * maskTemp;

                if(maskTemp != 0)
                    sn++;
                
            }

            if(sn != 0)
            {
                temp1[icmp*nv*nt+iv*nt+it] /= sn;
                temp2[icmp*nv*nt+iv*nt+it] /= sn;
            }

            __syncthreads(); 

            ib = it - nb;
            ie = it + nb + 1;

            if(ib < 0)
                ib = 0;
            if(ie > nt)
                ie = nt;

            for(int itt = ib;itt < ie;itt++)
            {
                num += temp1[icmp*nv*nt+iv*nt+itt] * temp1[icmp*nv*nt+iv*nt+itt];
                den += temp2[icmp*nv*nt+iv*nt+itt];
            }
            den /= ctrace;
            
            Semblance[icmp*nv*nt+iv*nt+it] = (den > 0.)? num/den: 0.;
        }
    }
    """

    mod = SourceModule(kernel_code)
    cuda_GenerateSemblance = mod.get_function("cuda_GenerateSemblance")

    Mem = CMPGather.nbytes+ OffsetVec.nbytes + tVec.nbytes + Mask.nbytes
    Mem = Mem + cmpnum*nv*nt*4*3
    Mem = Mem/1024/1024/1024

    mem_free, mem_total = cuda.mem_get_info()

    if Mem > mem_free/1024/1024/1024 * 0.9:
        calnum = int(np.ceil(Mem/(mem_free/1024/1024/1024*0.9)))
        cmpeach = int(np.ceil(cmpnum/calnum))
    else:
        calnum = 1
        cmpeach = cmpnum

    Semblance = np.zeros((cmpnum,nv,nt),np.float32)

    for i in range(int(calnum)):
        if i == calnum-1:
            cmpend = cmpnum
        else:
            cmpend = (i+1)*cmpeach
        CMPGather_gpu = gpuarray.to_gpu(CMPGather[i*cmpeach:cmpend].astype(np.float32) )
        OffsetVec_gpu = gpuarray.to_gpu(OffsetVec[i*cmpeach:cmpend].astype(np.float32) )
        tVec_gpu = gpuarray.to_gpu(tVec.astype(np.float32) )
        Mask_gpu = gpuarray.to_gpu(Mask[i*cmpeach:cmpend].astype(np.float32) )

        Semblance_gpu = gpuarray.zeros((cmpend-i*cmpeach,nv,nt),np.float32)
        temp1_gpu = gpuarray.zeros((cmpend-i*cmpeach,nv,nt),np.float32)
        temp2_gpu = gpuarray.zeros((cmpend-i*cmpeach,nv,nt),np.float32)

        BLOCK_SIZE = 4
        block = (nv, BLOCK_SIZE, 1)
        grid = (cmpend-i*cmpeach, int(nt/BLOCK_SIZE)+1, 1)

        cuda_GenerateSemblance(CMPGather_gpu,Mask_gpu,OffsetVec_gpu,tVec_gpu,temp1_gpu,temp2_gpu,Semblance_gpu,
                                    np.int32(nt),np.float32(dt),np.int32(ntrace),np.int32(ctrace),
                                    np.int32(nv),np.float32(dv),np.float32(v0),np.int32(nb),block=block,grid=grid)

        Semblance[i*cmpeach:cmpend] = Semblance_gpu.get()
        
        CMPGather_gpu.gpudata.free()
        OffsetVec_gpu.gpudata.free()
        tVec_gpu.gpudata.free()
        Mask_gpu.gpudata.free()
        temp1_gpu.gpudata.free()
        temp2_gpu.gpudata.free()
        Semblance_gpu.gpudata.free()        
    
    return Semblance

def GenerateSemblanceGPU_Field(CMPGather,OffsetVec,Mask,CMPTraceNum,tVec,nt,dt,cmpnum,ntrace,nv,dv,v0,nb):

    kernel_code = r"""
    __device__ float device_cubic_spline4(float y0, float y1, float y2, float y3, float dt, float t, int type)
    {
        float alpha1, alpha2;
        float l1, l2;
        float mu0, mu1, mu2;
        float z0, z1, z2;
        float a0, a1, a2;
        float b0, b1, b2;
        float c0, c1, c2, c3;
        float d0, d1, d2;

        alpha1 = 3.0f / dt * (y2 - y1) - 3.0f / dt * (y1 - y0);
        alpha2 = 3.0f / dt * (y3 - y2) - 3.0f / dt * (y2 - y1);

        mu0 = 0.0f;
        z0 = 0.0f;

        l1 = 4.0f * dt - dt * mu0;
        mu1 = dt / l1;
        z1 = (alpha1 - dt * z0) / l1;

        l2 = 4.0f * dt - dt * mu1;
        mu2 = dt / l2;
        z2 = (alpha2 - dt * z1) / l2;

        c3 = 0.0f;

        c2 = z2 - mu2 * c3;
        b2 = (y3 - y2) / dt - dt * (c3 + 2.0f * c2) / 3.0f;
        d2 = (c3 - c2) / (3.0f * dt);
        a2 = y2;

        c1 = z1 - mu1 * c2;
        b1 = (y2 - y1) / dt - dt * (c2 + 2.0f * c1) / 3.0f;
        d1 = (c2 - c1) / (3.0f * dt);
        a1 = y1;

        c0 = z0 - mu0 * c1;
        b0 = (y1 - y0) / dt - dt * (c1 + 2.0f * c0) / 3.0f;
        d0 = (c1 - c0) / (3.0f * dt);
        a0 = y0;

        if(type == 1)
        {
            return a1 + b1 * (t - dt) + c1 * (t - dt) * (t - dt) + d1 * (t - dt) * (t - dt) * (t - dt);
        }
        else if(type == 0)
        {
            return a0 + b0 * t + c0 * t * t + d0 * t * t * t;
        }	
        else if(type == 2)
        {
            return a2 + b2 * (t - 2 * dt) + c2 * (t - 2 * dt) * (t - 2 * dt) + d2 * (t - 2 * dt) * (t - 2 * dt) * (t - 2 * dt);
        }	
    }
    
    __global__ void cuda_GenerateTemp(float *CMPGather,float *Mask,float *OffsetVec,float *tVec,
                                        float *temp1,float *temp2,int *CMPTraceNum,
                                        int nt,float dt,int ntrace,int cmpnum,int nv,float dv,float v0)
    {
        int IndexX = blockIdx.x * blockDim.x + threadIdx.x;
        int icmp = floorf(IndexX / nv);
        int iv = IndexX % nv;
        int it = blockIdx.y * blockDim.y + threadIdx.y;
        
        float TravelT, ChangeS, CutC = 1.2, nmoTemp, t0Index;
        int sn = 0, maskTemp;
        float v = v0 + iv * dv;        
        int tfloor, type;
        
        if(it >= 0 && it < nt)
        {
            if(IndexX < cmpnum*nv)
            {
                for(int itrace = 0;itrace < CMPTraceNum[icmp];itrace++)
                {
                    TravelT = sqrt(tVec[it] * tVec[it] + 
                                    (OffsetVec[icmp*ntrace+itrace] / v) * 
                                    (OffsetVec[icmp*ntrace+itrace] / v));

                    ChangeS = TravelT / (tVec[it] + 1);            
                    t0Index = (TravelT - tVec[0]) / dt;
                    tfloor = floorf(t0Index);

                    if(t0Index >= 0 && t0Index < nt && ChangeS < CutC)
                    {
                        if(t0Index >= 1 && t0Index < nt-2)
                        {
                            type = 1;
                            nmoTemp = device_cubic_spline4(CMPGather[icmp*ntrace*nt+itrace*nt+tfloor-1],
                                                                    CMPGather[icmp*ntrace*nt+itrace*nt+tfloor],
                                                                    CMPGather[icmp*ntrace*nt+itrace*nt+tfloor+1],
                                                                    CMPGather[icmp*ntrace*nt+itrace*nt+tfloor+2],
                                                                    dt,(t0Index-(tfloor-1))*dt,type);
                        }
                        else if(t0Index >= 0 && t0Index < 1)
                        {
                            type = 0;
                            nmoTemp = device_cubic_spline4(CMPGather[icmp*ntrace*nt+itrace*nt+tfloor],
                                                                    CMPGather[icmp*ntrace*nt+itrace*nt+tfloor],
                                                                    CMPGather[icmp*ntrace*nt+itrace*nt+tfloor+1],
                                                                    CMPGather[icmp*ntrace*nt+itrace*nt+tfloor+2],
                                                                    dt,(t0Index-(tfloor))*dt,type);
                        }
                        else if(t0Index >= nt-2 && t0Index < nt-1)
                        {
                            type = 2;
                            nmoTemp = device_cubic_spline4(CMPGather[icmp*ntrace*nt+itrace*nt+tfloor-2],
                                                                    CMPGather[icmp*ntrace*nt+itrace*nt+tfloor-1],
                                                                    CMPGather[icmp*ntrace*nt+itrace*nt+tfloor],
                                                                    CMPGather[icmp*ntrace*nt+itrace*nt+tfloor+1],
                                                                    dt,(t0Index-(tfloor-2))*dt,type);
                        }
                        else if(t0Index == nt-1 || t0Index == 0) 
                        {
                            nmoTemp = CMPGather[icmp*ntrace*nt+itrace*nt+tfloor];
                        }
                        else
                        {
                            nmoTemp = 0;
                        }
                        
                        maskTemp = Mask[icmp*ntrace*nt+itrace*nt+int(t0Index)];
                    }
                    else
                    {
                        nmoTemp = 0;
                        maskTemp = 0;
                    }
                
                    temp1[icmp*nv*nt+iv*nt+it] += nmoTemp * maskTemp;
                    temp2[icmp*nv*nt+iv*nt+it] += nmoTemp * nmoTemp * maskTemp;

                    if(maskTemp != 0)
                        sn++;
                }
            }
        }           
    }

    __global__ void cuda_GenerateSemblance(float *temp1,float *temp2,float *Semblance,int *CMPTraceNum,int nt,int cmpnum,int nv,int nb)
    {
        int IndexX = blockIdx.x * blockDim.x + threadIdx.x;
        int icmp = floorf(IndexX / nv);
        int iv = IndexX % nv;
        int it = blockIdx.y * blockDim.y + threadIdx.y;

        float num = 0, den = 0;
        int ib = 0, ie = 0;
        
        if(it >= 0 && it < nt)
        {
            if(IndexX < cmpnum*nv)            
            {
                ib = it - nb;
                ie = it + nb + 1;

                if(ib < 0)
                    ib = 0;
                if(ie > nt)
                    ie = nt;

                for(int itt = ib;itt < ie;itt++)
                {
                    num += temp1[icmp*nv*nt+iv*nt+itt] * temp1[icmp*nv*nt+iv*nt+itt];
                    den += temp2[icmp*nv*nt+iv*nt+itt];
                }

                den /= CMPTraceNum[icmp];

                Semblance[icmp*nv*nt+iv*nt+it] = (den > 0.)? num/den: 0.;
            }
        }
    }
    """

    mod = SourceModule(kernel_code)
    cuda_GenerateTemp = mod.get_function("cuda_GenerateTemp")
    cuda_GenerateSemblance = mod.get_function("cuda_GenerateSemblance")

    Mem = CMPGather.nbytes/1024/1024/1024 + OffsetVec.nbytes/1024/1024/1024 + tVec.nbytes/1024/1024/1024 + Mask.nbytes/1024/1024/1024 + CMPTraceNum.nbytes/1024/1024/1024
    Mem = Mem + cmpnum/1024*nv/1024*nt/1024*4*3

    mem_free, mem_total = cuda.mem_get_info()
    # print("Memory: Used %f Free %f Total %f"%(Mem,mem_free/1024/1024/1024,mem_total/1024/1024/1024))

    if Mem > mem_free/1024/1024/1024 * 0.8:
        calnum = int(np.ceil(Mem/(mem_free/1024/1024/1024*0.8)))
        cmpeach = int(np.ceil(cmpnum/calnum))
    else:
        calnum = 1
        cmpeach = cmpnum

    # print("calnum: %d cmpeach: %d"%(calnum,cmpeach))
    # print("Mem total: %f, Mem each: %f"%(Mem,Mem/calnum))

    Semblance = np.zeros((cmpnum,nv,nt),np.float32)

    for i in tqdm(range(calnum)):
        cmpend = min((i+1)*cmpeach,cmpnum)
            
        CMPGather_gpu = gpuarray.to_gpu(CMPGather[i*cmpeach:cmpend].astype(np.float32) )
        OffsetVec_gpu = gpuarray.to_gpu(OffsetVec[i*cmpeach:cmpend].astype(np.float32) )
        tVec_gpu = gpuarray.to_gpu(tVec.astype(np.float32) )
        Mask_gpu = gpuarray.to_gpu(Mask[i*cmpeach:cmpend].astype(np.float32) )
        CMPTraceNum_gpu = gpuarray.to_gpu(CMPTraceNum[i*cmpeach:cmpend].astype(np.int32) )

        Semblance_gpu = gpuarray.zeros((cmpend-i*cmpeach,nv,nt),np.float32)
        temp1_gpu = gpuarray.zeros((cmpend-i*cmpeach,nv,nt),np.float32)
        temp2_gpu = gpuarray.zeros((cmpend-i*cmpeach,nv,nt),np.float32)

        BLOCK_SIZE = 32
        block = (BLOCK_SIZE, BLOCK_SIZE, 1)
        grid = (int((cmpend-i*cmpeach)*nv/BLOCK_SIZE)+1, int(nt/BLOCK_SIZE)+1, 1)

        cuda_GenerateTemp(CMPGather_gpu,Mask_gpu,OffsetVec_gpu,tVec_gpu,temp1_gpu,temp2_gpu,CMPTraceNum_gpu,
                            np.int32(nt),np.float32(dt),np.int32(ntrace),
                            np.int32(cmpend-i*cmpeach),np.int32(nv),np.float32(dv),np.float32(v0),block=block,grid=grid)

        cuda_GenerateSemblance(temp1_gpu,temp2_gpu,Semblance_gpu,CMPTraceNum_gpu,
                                np.int32(nt),np.int32(cmpend-i*cmpeach),np.int32(nv),np.int32(nb),block=block,grid=grid)
        
        Semblance[i*cmpeach:cmpend] = Semblance_gpu.get()
        
        CMPGather_gpu.gpudata.free()
        OffsetVec_gpu.gpudata.free()
        tVec_gpu.gpudata.free()
        Mask_gpu.gpudata.free()
        CMPTraceNum_gpu.gpudata.free()
        temp1_gpu.gpudata.free()
        temp2_gpu.gpudata.free()
        Semblance_gpu.gpudata.free()        
    
    return Semblance

def Mute_SemblanceGPU(Semblance, Vrms, ratio_mute, cmpnum, nt, v0, nv, dv):
        
    kernel_code = r"""
        
    __global__ void cuda_Mute_Semblance(float *Semblance,float *Vrms,int nt,int nv,float dv,float v0,float ratio_mute)
    {
        int icmp = blockIdx.x;
        int iv = threadIdx.x;
        int it = blockIdx.y * blockDim.y + threadIdx.y;
        
        int v1, v2;
        
        if(it >= 0 && it < nt)
        {
            v1 = (int)((Vrms[icmp*nt+it]*(1-ratio_mute)-v0)/dv);
            v2 = (int)((Vrms[icmp*nt+it]*(1+ratio_mute)-v0)/dv);
            if(iv <= v1 && v1 > 0)
            {
                Semblance[icmp*nv*nt+iv*nt+it] = 0;
            }
            if(iv >= v2 && v2 < nv)
            {
                Semblance[icmp*nv*nt+iv*nt+it] = 0;
            }
        }
    }
    """

    mod = SourceModule(kernel_code)
    cuda_Mute_Semblance = mod.get_function("cuda_Mute_Semblance")

    Semblance_gpu = gpuarray.to_gpu(Semblance.astype(np.float32) )
    Vrms_gpu = gpuarray.to_gpu(Vrms.astype(np.float32) )
    
    BLOCK_SIZE = 8
    block = (nv, BLOCK_SIZE, 1)
    grid = (cmpnum, int(nt/BLOCK_SIZE)+1, 1)

    cuda_Mute_Semblance(Semblance_gpu,Vrms_gpu,np.int32(nt),np.int32(nv),np.float32(dv),np.float32(v0),np.float32(ratio_mute),block=block,grid=grid)

    Semblance_mute = Semblance_gpu.get()
    
    Semblance_gpu.gpudata.free()
    Vrms_gpu.gpudata.free()

    
    return Semblance_mute

def LocalStackSliceGPU(CMPGather,Mask,OffsetVec,tVec,Vref,m,k,nt,dt,cmpnum,ntrace,mutetime):
  
    kernel_code = r"""
    __device__ float device_cubic_spline4(float y0, float y1, float y2, float y3, float dt, float t, int type)
    {
        float alpha1, alpha2;
        float l1, l2;
        float mu0, mu1, mu2;
        float z0, z1, z2;
        float a0, a1, a2;
        float b0, b1, b2;
        float c0, c1, c2, c3;
        float d0, d1, d2;

        alpha1 = 3.0f / dt * (y2 - y1) - 3.0f / dt * (y1 - y0);
        alpha2 = 3.0f / dt * (y3 - y2) - 3.0f / dt * (y2 - y1);

        mu0 = 0.0f;
        z0 = 0.0f;

        l1 = 4.0f * dt - dt * mu0;
        mu1 = dt / l1;
        z1 = (alpha1 - dt * z0) / l1;

        l2 = 4.0f * dt - dt * mu1;
        mu2 = dt / l2;
        z2 = (alpha2 - dt * z1) / l2;

        c3 = 0.0f;

        c2 = z2 - mu2 * c3;
        b2 = (y3 - y2) / dt - dt * (c3 + 2.0f * c2) / 3.0f;
        d2 = (c3 - c2) / (3.0f * dt);
        a2 = y2;

        c1 = z1 - mu1 * c2;
        b1 = (y2 - y1) / dt - dt * (c2 + 2.0f * c1) / 3.0f;
        d1 = (c2 - c1) / (3.0f * dt);
        a1 = y1;

        c0 = z0 - mu0 * c1;
        b0 = (y1 - y0) / dt - dt * (c1 + 2.0f * c0) / 3.0f;
        d0 = (c1 - c0) / (3.0f * dt);
        a0 = y0;

        if(type == 1)
        {
            return a1 + b1 * (t - dt) + c1 * (t - dt) * (t - dt) + d1 * (t - dt) * (t - dt) * (t - dt);
        }
        else if(type == 0)
        {
            return a0 + b0 * t + c0 * t * t + d0 * t * t * t;
        }	
        else if(type == 2)
        {
            return a2 + b2 * (t - 2 * dt) + c2 * (t - 2 * dt) * (t - 2 * dt) + d2 * (t - 2 * dt) * (t - 2 * dt) * (t - 2 * dt);
        }	
    }

    __global__ void cuda_LocalStackSlice(float *CMPGather,float *Mask,float *OffsetVec,
                                    float *tVec,float *Vref,float *LSGather,
                                    int nt,float dt,int cmpnum,int ntrace,int m,int k,int mutetime)
    {
        int icmp = blockIdx.x;
        int ik = threadIdx.x;
        int it = blockIdx.y * blockDim.y + threadIdx.y;

        int nk = 2 * k + 1;
        int nm = 2 * m + 3;
        float TravelT, t0Index, ChangeS, CutC = 1.2, nmoTemp;
        int sn, maskTemp, tfloor, type, iicmp;
        unsigned int index;        

        if(it >= 0 && it < nt)
        {
            for(int im = 0; im < nm-2; im++)
            {

                iicmp = icmp + im - m;
                if(iicmp < 0)
                    iicmp = 0;
                else if(iicmp >= cmpnum)
                    iicmp = cmpnum - 1;
                
                sn = 0;
                for(int itrace = 0; itrace < ntrace; itrace++)
                {
                    TravelT = sqrt(tVec[it] * tVec[it] + 
                            (OffsetVec[iicmp*ntrace+itrace] / Vref[icmp*nk*nt+ik*nt+it]) * 
                            (OffsetVec[iicmp*ntrace+itrace] / Vref[icmp*nk*nt+ik*nt+it]));

                    ChangeS = TravelT / (tVec[it] + 1);
                    t0Index = (TravelT - tVec[0]) / dt;
                    tfloor = floorf(t0Index);

                    if(t0Index >= 0 && t0Index < nt && ChangeS < CutC)
                    {
                        if(t0Index >=1 && t0Index < nt-2)
                        {
                            type = 1;
                            nmoTemp = device_cubic_spline4(CMPGather[iicmp*ntrace*nt+itrace*nt+tfloor-1],
                                                        CMPGather[iicmp*ntrace*nt+itrace*nt+tfloor],
                                                        CMPGather[iicmp*ntrace*nt+itrace*nt+tfloor+1],
                                                        CMPGather[iicmp*ntrace*nt+itrace*nt+tfloor+2],
                                                        dt,(t0Index-(tfloor-1))*dt,type);
                        }
                        else if(t0Index >= 0 && t0Index < 1)
                        {
                            type = 0;
                            nmoTemp = device_cubic_spline4(CMPGather[iicmp*ntrace*nt+itrace*nt+tfloor],
                                                        CMPGather[iicmp*ntrace*nt+itrace*nt+tfloor],
                                                        CMPGather[iicmp*ntrace*nt+itrace*nt+tfloor+1],
                                                        CMPGather[iicmp*ntrace*nt+itrace*nt+tfloor+2],
                                                        dt,(t0Index-(tfloor))*dt,type);
                        }
                        else if(t0Index >= nt-2 && t0Index < nt-1)
                        {
                            type = 2;
                            nmoTemp = device_cubic_spline4(CMPGather[iicmp*ntrace*nt+itrace*nt+tfloor-2],
                                                        CMPGather[iicmp*ntrace*nt+itrace*nt+tfloor-1],
                                                        CMPGather[iicmp*ntrace*nt+itrace*nt+tfloor],
                                                        CMPGather[iicmp*ntrace*nt+itrace*nt+tfloor+1],
                                                        dt,(t0Index-(tfloor-2))*dt,type);
                        }
                        else if(t0Index == nt-1 || t0Index == 0)
                        {
                            nmoTemp = CMPGather[iicmp*ntrace*nt+itrace*nt+tfloor];
                        }
                        else
                        {
                            nmoTemp = 0;
                        }
                        maskTemp = Mask[iicmp*ntrace*nt+itrace*nt+tfloor];                            
                    }
                    else
                    {
                        nmoTemp = 0;
                        maskTemp = 0;
                    }
                    index = icmp*nk*nm*nt+ik*nm*nt+im*nt+it;
                    LSGather[index] += nmoTemp;
                    if(maskTemp != 0)
                        sn++;
                }
                if(sn != 0 && it >= mutetime)
                    LSGather[index] /= sn;
                else
                    LSGather[index] /= ntrace;
            }
            index = icmp*nk*nm*nt+ik*nm*nt+(2*m+1)*nt+it;
            LSGather[index] = tVec[it] * 1000;
            index = icmp*nk*nm*nt+ik*nm*nt+(2*m+2)*nt+it;
            LSGather[index] = Vref[icmp*nk*nt+ik*nt+it];                         
        }
    }
    """

    mod = SourceModule(kernel_code)
    cuda_LocalStackSlice = mod.get_function("cuda_LocalStackSlice")

    Mem = CMPGather.nbytes + Mask.nbytes + OffsetVec.nbytes + tVec.nbytes + Vref.nbytes
    Mem += Mem + cmpnum * (2*m+3) * (2*k+1) * nt * 4
    Mem = Mem/1024/1024/1024

    mem_free, mem_total = cuda.mem_get_info()

    if Mem > mem_free/1024/1024/1024 * 0.9:
        calnum = int(np.ceil(Mem/(mem_free/1024/1024/1024)))
        cmpeach = int(np.ceil(cmpnum/calnum))
    else:
        calnum = 1
        cmpeach = cmpnum

    LSGather = np.zeros([cmpnum,(2*m+3)*(2*k+1),nt],np.float32)

    for i in range(calnum):
        if i == calnum-1:
            cmpend = cmpnum
        else:
            cmpend = (i+1)*cmpeach
        
        CMPGather_gpu = gpuarray.to_gpu(CMPGather[i*cmpeach:cmpend].astype(np.float32))
        Mask_gpu = gpuarray.to_gpu(Mask[i*cmpeach:cmpend].astype(np.float32))
        OffsetVec_gpu = gpuarray.to_gpu(OffsetVec[i*cmpeach:cmpend].astype(np.float32))
        Vref_gpu = gpuarray.to_gpu(Vref[i*cmpeach:cmpend].astype(np.float32))
        tVec_gpu = gpuarray.to_gpu(tVec.astype(np.float32))

        LSGather_gpu = gpuarray.zeros([cmpend-i*cmpeach,(2*m+3)*(2*k+1),nt],np.float32)

        BLOCK_SIZE = 16
        block = (2*k+1,BLOCK_SIZE,1)    
        if nt%BLOCK_SIZE != 0:
            grid=(cmpend-i*cmpeach,nt//BLOCK_SIZE+1,1)
        else:
            grid=(cmpend-i*cmpeach,nt//BLOCK_SIZE,1)
        
        cuda_LocalStackSlice(CMPGather_gpu,Mask_gpu,OffsetVec_gpu,tVec_gpu,Vref_gpu,LSGather_gpu,
                        np.int32(nt),np.float32(dt),np.int32(cmpnum),np.int32(ntrace),np.int32(m),np.int32(k),np.int32(mutetime),
                        block=block,grid=grid)
        
        LSGather[i*cmpeach:cmpend] = LSGather_gpu.get()
            
        CMPGather_gpu.gpudata.free()
        Mask_gpu.gpudata.free()
        OffsetVec_gpu.gpudata.free()
        tVec_gpu.gpudata.free()
        Vref_gpu.gpudata.free()
        LSGather_gpu.gpudata.free()

    return LSGather

def LocalStackSliceGPU_Single(CMPGather,Mask,OffsetVec,tVec,Vref,m,k,nt,dt,ntrace):
  
    kernel_code = r"""
    __device__ float device_cubic_spline4(float y0, float y1, float y2, float y3, float dt, float t, int type)
    {
        float alpha1, alpha2;
        float l1, l2;
        float mu0, mu1, mu2;
        float z0, z1, z2;
        float a0, a1, a2;
        float b0, b1, b2;
        float c0, c1, c2, c3;
        float d0, d1, d2;

        alpha1 = 3.0f / dt * (y2 - y1) - 3.0f / dt * (y1 - y0);
        alpha2 = 3.0f / dt * (y3 - y2) - 3.0f / dt * (y2 - y1);

        mu0 = 0.0f;
        z0 = 0.0f;

        l1 = 4.0f * dt - dt * mu0;
        mu1 = dt / l1;
        z1 = (alpha1 - dt * z0) / l1;

        l2 = 4.0f * dt - dt * mu1;
        mu2 = dt / l2;
        z2 = (alpha2 - dt * z1) / l2;

        c3 = 0.0f;

        c2 = z2 - mu2 * c3;
        b2 = (y3 - y2) / dt - dt * (c3 + 2.0f * c2) / 3.0f;
        d2 = (c3 - c2) / (3.0f * dt);
        a2 = y2;

        c1 = z1 - mu1 * c2;
        b1 = (y2 - y1) / dt - dt * (c2 + 2.0f * c1) / 3.0f;
        d1 = (c2 - c1) / (3.0f * dt);
        a1 = y1;

        c0 = z0 - mu0 * c1;
        b0 = (y1 - y0) / dt - dt * (c1 + 2.0f * c0) / 3.0f;
        d0 = (c1 - c0) / (3.0f * dt);
        a0 = y0;

        if(type == 1)
        {
            return a1 + b1 * (t - dt) + c1 * (t - dt) * (t - dt) + d1 * (t - dt) * (t - dt) * (t - dt);
        }
        else if(type == 0)
        {
            return a0 + b0 * t + c0 * t * t + d0 * t * t * t;
        }	
        else if(type == 2)
        {
            return a2 + b2 * (t - 2 * dt) + c2 * (t - 2 * dt) * (t - 2 * dt) + d2 * (t - 2 * dt) * (t - 2 * dt) * (t - 2 * dt);
        }	
    }

    __global__ void cuda_LocalStackSlice(float *CMPGather,float *Mask,float *OffsetVec,
                                    float *tVec,float *Vref,float *LSGather,
                                    int nt,float dt,int ntrace,int m,int k)
    {
        int im = blockIdx.x;
        int ik = threadIdx.x;
        int it = blockIdx.y * blockDim.y + threadIdx.y;

        int nm = 2 * m + 3;
        float TravelT, t0Index, ChangeS, CutC = 1.2, nmoTemp;
        int sn, maskTemp, tfloor, type;
          
        if(it >= 0 && it < nt)
        {
            if(im >= 0 && im < nm-2)
            {
                sn = 0;
                for(int itrace = 0; itrace < ntrace; itrace++)
                {
                    TravelT = sqrt(tVec[it] * tVec[it] +
                            (OffsetVec[im*ntrace+itrace] / Vref[ik*nt+it]) *
                            (OffsetVec[im*ntrace+itrace] / Vref[ik*nt+it]));

                    ChangeS = TravelT / (tVec[it] + 1);
                    t0Index = (TravelT - tVec[0]) / dt;
                    tfloor = floorf(t0Index);

                    if(t0Index >= 0 && t0Index < nt && ChangeS < CutC)
                    {
                        if(t0Index >=1 && t0Index < nt-2)
                        {
                            type = 1;
                            nmoTemp = device_cubic_spline4(CMPGather[im*ntrace*nt+itrace*nt+tfloor-1],
                                                        CMPGather[im*ntrace*nt+itrace*nt+tfloor],
                                                        CMPGather[im*ntrace*nt+itrace*nt+tfloor+1],
                                                        CMPGather[im*ntrace*nt+itrace*nt+tfloor+2],
                                                        dt,(t0Index-(tfloor-1))*dt,type);
                        }
                        else if(t0Index >= 0 && t0Index < 1)
                        {
                            type = 0;
                            nmoTemp = device_cubic_spline4(CMPGather[im*ntrace*nt+itrace*nt+tfloor],
                                                        CMPGather[im*ntrace*nt+itrace*nt+tfloor],
                                                        CMPGather[im*ntrace*nt+itrace*nt+tfloor+1],
                                                        CMPGather[im*ntrace*nt+itrace*nt+tfloor+2],
                                                        dt,(t0Index-(tfloor))*dt,type);
                        }
                        else if(t0Index >= nt-2 && t0Index < nt-1)
                        {
                            type = 2;
                            nmoTemp = device_cubic_spline4(CMPGather[im*ntrace*nt+itrace*nt+tfloor-2],
                                                        CMPGather[im*ntrace*nt+itrace*nt+tfloor-1],
                                                        CMPGather[im*ntrace*nt+itrace*nt+tfloor],
                                                        CMPGather[im*ntrace*nt+itrace*nt+tfloor+1],
                                                        dt,(t0Index-(tfloor-2))*dt,type);
                        }
                        else if(t0Index == nt-1 || t0Index == 0)
                        {
                            nmoTemp = CMPGather[im*ntrace*nt+itrace*nt+tfloor];
                        }
                        else
                        {
                            nmoTemp = 0;
                        }
                        maskTemp = Mask[itrace*nt+tfloor];                            
                    }
                    else
                    {
                        nmoTemp = 0;
                        maskTemp = 0;
                    }
                    LSGather[ik*nm*nt+im*nt+it] += nmoTemp * maskTemp;
                    if(maskTemp != 0)
                        sn++;
                }
                if(sn != 0)
                    LSGather[ik*nm*nt+im*nt+it] /= sn;
                else
                    LSGather[ik*nm*nt+im*nt+it] /= ntrace;
            }
            else if(im == nm-2)
            {
                LSGather[ik*nm*nt+im*nt+it] = tVec[it] * 1000;
            }
            else if(im == nm-1)
            {
                LSGather[ik*nm*nt+im*nt+it] = Vref[ik*nt+it];
            }                        
        }
    }
    """

    mod = SourceModule(kernel_code)
    cuda_LocalStackSlice = mod.get_function("cuda_LocalStackSlice")

    CMPGather_gpu = gpuarray.to_gpu(CMPGather.astype(np.float32))
    Mask_gpu = gpuarray.to_gpu(Mask.astype(np.float32))
    OffsetVec_gpu = gpuarray.to_gpu(OffsetVec.astype(np.float32))
    Vref_gpu = gpuarray.to_gpu(Vref.astype(np.float32))
    tVec_gpu = gpuarray.to_gpu(tVec.astype(np.float32))

    LSGather_gpu = gpuarray.zeros([(2*m+3)*(2*k+1),nt],np.float32)

    BLOCK_SIZE = 16
    block = (2*k+1,BLOCK_SIZE,1)
    grid = (2*m+3,nt//BLOCK_SIZE+1,1)

    cuda_LocalStackSlice(CMPGather_gpu,Mask_gpu,OffsetVec_gpu,tVec_gpu,Vref_gpu,LSGather_gpu,
                    np.int32(nt),np.float32(dt),np.int32(ntrace),np.int32(m),np.int32(k),
                    block=block,grid=grid)

    LSGather = LSGather_gpu.get()        
               
    CMPGather_gpu.gpudata.free()
    Mask_gpu.gpudata.free()
    OffsetVec_gpu.gpudata.free()
    tVec_gpu.gpudata.free()
    Vref_gpu.gpudata.free()
    LSGather_gpu.gpudata.free()

    return LSGather

def LocalStackSliceGPU_field(CMPGather,Mask,OffsetVec,CMPTraceNum,tVec,Vref,m,k,nt,dt,cmpnum,ntrace):
  
    kernel_code = r"""
    __device__ float device_cubic_spline4(float y0, float y1, float y2, float y3, float dt, float t, int type)
    {
        float alpha1, alpha2;
        float l1, l2;
        float mu0, mu1, mu2;
        float z0, z1, z2;
        float a0, a1, a2;
        float b0, b1, b2;
        float c0, c1, c2, c3;
        float d0, d1, d2;

        alpha1 = 3.0f / dt * (y2 - y1) - 3.0f / dt * (y1 - y0);
        alpha2 = 3.0f / dt * (y3 - y2) - 3.0f / dt * (y2 - y1);

        mu0 = 0.0f;
        z0 = 0.0f;

        l1 = 4.0f * dt - dt * mu0;
        mu1 = dt / l1;
        z1 = (alpha1 - dt * z0) / l1;

        l2 = 4.0f * dt - dt * mu1;
        mu2 = dt / l2;
        z2 = (alpha2 - dt * z1) / l2;

        c3 = 0.0f;

        c2 = z2 - mu2 * c3;
        b2 = (y3 - y2) / dt - dt * (c3 + 2.0f * c2) / 3.0f;
        d2 = (c3 - c2) / (3.0f * dt);
        a2 = y2;

        c1 = z1 - mu1 * c2;
        b1 = (y2 - y1) / dt - dt * (c2 + 2.0f * c1) / 3.0f;
        d1 = (c2 - c1) / (3.0f * dt);
        a1 = y1;

        c0 = z0 - mu0 * c1;
        b0 = (y1 - y0) / dt - dt * (c1 + 2.0f * c0) / 3.0f;
        d0 = (c1 - c0) / (3.0f * dt);
        a0 = y0;

        if(type == 1)
        {
            return a1 + b1 * (t - dt) + c1 * (t - dt) * (t - dt) + d1 * (t - dt) * (t - dt) * (t - dt);
        }
        else if(type == 0)
        {
            return a0 + b0 * t + c0 * t * t + d0 * t * t * t;
        }	
        else if(type == 2)
        {
            return a2 + b2 * (t - 2 * dt) + c2 * (t - 2 * dt) * (t - 2 * dt) + d2 * (t - 2 * dt) * (t - 2 * dt) * (t - 2 * dt);
        }	
    }

    __global__ void cuda_LocalStackSlice(float *CMPGather,float *Mask,float *OffsetVec,
                                    float *tVec,float *Vref,float *LSGather,int *CMPTraceNum,
                                    int nt,float dt,int ntrace,int m,int k)
    {
        int icmp = blockIdx.x;
        int ik = threadIdx.x;
        int it = blockIdx.y * blockDim.y + threadIdx.y;

        int nk = 2 * k + 1;
        int nm = 2 * m + 3;
        float TravelT, t0Index, ChangeS, CutC = 1.2, nmoTemp;
        int sn, maskTemp, tfloor, type, iicmp;              

        if(it >= 0 && it < nt)
        {
            for(int im = 0; im < nm-2; im++)
            {
                iicmp = icmp + im;
                
                sn = 0;
                for(int itrace = 0; itrace < CMPTraceNum[iicmp]; itrace++)
                {
                    TravelT = sqrt(tVec[it] * tVec[it] + 
                            (OffsetVec[iicmp*ntrace+itrace] / Vref[icmp*nk*nt+ik*nt+it]) * 
                            (OffsetVec[iicmp*ntrace+itrace] / Vref[icmp*nk*nt+ik*nt+it]));

                    ChangeS = TravelT / (tVec[it] + 1);
                    t0Index = (TravelT - tVec[0]) / dt;
                    tfloor = floorf(t0Index);

                    if(t0Index >= 0 && t0Index < nt && ChangeS < CutC)
                    {
                        if(t0Index >=1 && t0Index < nt-2)
                        {
                            type = 1;
                            nmoTemp = device_cubic_spline4(CMPGather[iicmp*ntrace*nt+itrace*nt+tfloor-1],
                                                        CMPGather[iicmp*ntrace*nt+itrace*nt+tfloor],
                                                        CMPGather[iicmp*ntrace*nt+itrace*nt+tfloor+1],
                                                        CMPGather[iicmp*ntrace*nt+itrace*nt+tfloor+2],
                                                        dt,(t0Index-(tfloor-1))*dt,type);
                        }
                        else if(t0Index >= 0 && t0Index < 1)
                        {
                            type = 0;
                            nmoTemp = device_cubic_spline4(CMPGather[iicmp*ntrace*nt+itrace*nt+tfloor],
                                                        CMPGather[iicmp*ntrace*nt+itrace*nt+tfloor],
                                                        CMPGather[iicmp*ntrace*nt+itrace*nt+tfloor+1],
                                                        CMPGather[iicmp*ntrace*nt+itrace*nt+tfloor+2],
                                                        dt,(t0Index-(tfloor))*dt,type);
                        }
                        else if(t0Index >= nt-2 && t0Index < nt-1)
                        {
                            type = 2;
                            nmoTemp = device_cubic_spline4(CMPGather[iicmp*ntrace*nt+itrace*nt+tfloor-2],
                                                        CMPGather[iicmp*ntrace*nt+itrace*nt+tfloor-1],
                                                        CMPGather[iicmp*ntrace*nt+itrace*nt+tfloor],
                                                        CMPGather[iicmp*ntrace*nt+itrace*nt+tfloor+1],
                                                        dt,(t0Index-(tfloor-2))*dt,type);
                        }
                        else if(t0Index == nt-1 || t0Index == 0)
                        {
                            nmoTemp = CMPGather[iicmp*ntrace*nt+itrace*nt+tfloor];
                        }
                        else
                        {
                            nmoTemp = 0;
                        }
                        maskTemp = Mask[iicmp*ntrace*nt+itrace*nt+tfloor];                            
                    }
                    else
                    {
                        nmoTemp = 0;
                        maskTemp = 0;
                    }                    
                    LSGather[icmp*nk*nm*nt+ik*nm*nt+im*nt+it] += nmoTemp;
                    if(maskTemp != 0)
                        sn++;
                }
                if(sn != 0)
                    LSGather[icmp*nk*nm*nt+ik*nm*nt+im*nt+it] /= sn;
                else
                    LSGather[icmp*nk*nm*nt+ik*nm*nt+im*nt+it] /= CMPTraceNum[iicmp];
            }
            LSGather[icmp*nk*nm*nt+ik*nm*nt+(2*m+1)*nt+it] = tVec[it] * 1000;
            LSGather[icmp*nk*nm*nt+ik*nm*nt+(2*m+2)*nt+it] = Vref[icmp*nk*nt+ik*nt+it];                         
        }
    }
    """

    mod = SourceModule(kernel_code)
    cuda_LocalStackSlice = mod.get_function("cuda_LocalStackSlice")

    CMPGather_gpu = gpuarray.to_gpu(CMPGather.astype(np.float32))
    Mask_gpu = gpuarray.to_gpu(Mask.astype(np.float32))
    OffsetVec_gpu = gpuarray.to_gpu(OffsetVec.astype(np.float32))
    tVec_gpu = gpuarray.to_gpu(tVec.astype(np.float32))
    Vref_gpu = gpuarray.to_gpu(Vref.astype(np.float32))
    CMPTraceNum_gpu = gpuarray.to_gpu(CMPTraceNum.astype(np.int32))
    LSGather_gpu = gpuarray.zeros([cmpnum,(2*m+3)*(2*k+1),nt],np.float32)

    BLOCK_SIZE = 16
    block = (int(2*k+1),BLOCK_SIZE,1)    
    if nt%BLOCK_SIZE != 0:
        grid=(int(cmpnum),int(nt//BLOCK_SIZE)+1,1)
    else:
        grid=(int(cmpnum),int(nt//BLOCK_SIZE),1)
    
    cuda_LocalStackSlice(CMPGather_gpu,Mask_gpu,OffsetVec_gpu,tVec_gpu,Vref_gpu,LSGather_gpu,CMPTraceNum_gpu,
                    np.int32(nt),np.float32(dt),np.int32(ntrace),np.int32(m),np.int32(k),
                    block=block,grid=grid)
                           
    LSGather = LSGather_gpu.get()
        
    CMPGather_gpu.gpudata.free()
    Mask_gpu.gpudata.free()
    OffsetVec_gpu.gpudata.free()
    tVec_gpu.gpudata.free()
    Vref_gpu.gpudata.free()
    LSGather_gpu.gpudata.free()
    CMPTraceNum_gpu.gpudata.free()

    return LSGather

def ReScale(data,min,max):
    dmax = np.max(data)
    dmin = np.min(data)
    output = (data - dmin) / (dmax - dmin) * (max - min) + min

    return output

def ReScale2(data,max):
    
    output = data / np.max(np.abs(data)) * max

    return output

if __name__ == '__main__':
    RootPath = r'F:\\VelocityPicking\\HRAwCNN\\Xception'
    print(os.environ.get("CUDA_PATH"))
    
    
