// Code generated by "enumer -type=CUPTI_RUNTIME_TRACE_CBID -json"; DO NOT EDIT

package types

import (
	"encoding/json"
	"fmt"
)

const _CUPTI_RUNTIME_TRACE_CBID_name = "CUPTI_RUNTIME_TRACE_CBID_INVALIDCUPTI_RUNTIME_TRACE_CBID_cudaDriverGetVersion_v3020CUPTI_RUNTIME_TRACE_CBID_cudaRuntimeGetVersion_v3020CUPTI_RUNTIME_TRACE_CBID_cudaGetDeviceCount_v3020CUPTI_RUNTIME_TRACE_CBID_cudaGetDeviceProperties_v3020CUPTI_RUNTIME_TRACE_CBID_cudaChooseDevice_v3020CUPTI_RUNTIME_TRACE_CBID_cudaGetChannelDesc_v3020CUPTI_RUNTIME_TRACE_CBID_cudaCreateChannelDesc_v3020CUPTI_RUNTIME_TRACE_CBID_cudaConfigureCall_v3020CUPTI_RUNTIME_TRACE_CBID_cudaSetupArgument_v3020CUPTI_RUNTIME_TRACE_CBID_cudaGetLastError_v3020CUPTI_RUNTIME_TRACE_CBID_cudaPeekAtLastError_v3020CUPTI_RUNTIME_TRACE_CBID_cudaGetErrorString_v3020CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020CUPTI_RUNTIME_TRACE_CBID_cudaFuncSetCacheConfig_v3020CUPTI_RUNTIME_TRACE_CBID_cudaFuncGetAttributes_v3020CUPTI_RUNTIME_TRACE_CBID_cudaSetDevice_v3020CUPTI_RUNTIME_TRACE_CBID_cudaGetDevice_v3020CUPTI_RUNTIME_TRACE_CBID_cudaSetValidDevices_v3020CUPTI_RUNTIME_TRACE_CBID_cudaSetDeviceFlags_v3020CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020CUPTI_RUNTIME_TRACE_CBID_cudaMallocPitch_v3020CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020CUPTI_RUNTIME_TRACE_CBID_cudaMallocArray_v3020CUPTI_RUNTIME_TRACE_CBID_cudaFreeArray_v3020CUPTI_RUNTIME_TRACE_CBID_cudaMallocHost_v3020CUPTI_RUNTIME_TRACE_CBID_cudaFreeHost_v3020CUPTI_RUNTIME_TRACE_CBID_cudaHostAlloc_v3020CUPTI_RUNTIME_TRACE_CBID_cudaHostGetDevicePointer_v3020CUPTI_RUNTIME_TRACE_CBID_cudaHostGetFlags_v3020CUPTI_RUNTIME_TRACE_CBID_cudaMemGetInfo_v3020CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2D_v3020CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyToArray_v3020CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DToArray_v3020CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyFromArray_v3020CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DFromArray_v3020CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyArrayToArray_v3020CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DArrayToArray_v3020CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyToSymbol_v3020CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyFromSymbol_v3020CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyToArrayAsync_v3020CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyFromArrayAsync_v3020CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DAsync_v3020CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DToArrayAsync_v3020CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DFromArrayAsync_v3020CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyToSymbolAsync_v3020CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyFromSymbolAsync_v3020CUPTI_RUNTIME_TRACE_CBID_cudaMemset_v3020CUPTI_RUNTIME_TRACE_CBID_cudaMemset2D_v3020CUPTI_RUNTIME_TRACE_CBID_cudaMemsetAsync_v3020CUPTI_RUNTIME_TRACE_CBID_cudaMemset2DAsync_v3020CUPTI_RUNTIME_TRACE_CBID_cudaGetSymbolAddress_v3020CUPTI_RUNTIME_TRACE_CBID_cudaGetSymbolSize_v3020CUPTI_RUNTIME_TRACE_CBID_cudaBindTexture_v3020CUPTI_RUNTIME_TRACE_CBID_cudaBindTexture2D_v3020CUPTI_RUNTIME_TRACE_CBID_cudaBindTextureToArray_v3020CUPTI_RUNTIME_TRACE_CBID_cudaUnbindTexture_v3020CUPTI_RUNTIME_TRACE_CBID_cudaGetTextureAlignmentOffset_v3020CUPTI_RUNTIME_TRACE_CBID_cudaGetTextureReference_v3020CUPTI_RUNTIME_TRACE_CBID_cudaBindSurfaceToArray_v3020CUPTI_RUNTIME_TRACE_CBID_cudaGetSurfaceReference_v3020CUPTI_RUNTIME_TRACE_CBID_cudaGLSetGLDevice_v3020CUPTI_RUNTIME_TRACE_CBID_cudaGLRegisterBufferObject_v3020CUPTI_RUNTIME_TRACE_CBID_cudaGLMapBufferObject_v3020CUPTI_RUNTIME_TRACE_CBID_cudaGLUnmapBufferObject_v3020CUPTI_RUNTIME_TRACE_CBID_cudaGLUnregisterBufferObject_v3020CUPTI_RUNTIME_TRACE_CBID_cudaGLSetBufferObjectMapFlags_v3020CUPTI_RUNTIME_TRACE_CBID_cudaGLMapBufferObjectAsync_v3020CUPTI_RUNTIME_TRACE_CBID_cudaGLUnmapBufferObjectAsync_v3020CUPTI_RUNTIME_TRACE_CBID_cudaWGLGetDevice_v3020CUPTI_RUNTIME_TRACE_CBID_cudaGraphicsGLRegisterImage_v3020CUPTI_RUNTIME_TRACE_CBID_cudaGraphicsGLRegisterBuffer_v3020CUPTI_RUNTIME_TRACE_CBID_cudaGraphicsUnregisterResource_v3020CUPTI_RUNTIME_TRACE_CBID_cudaGraphicsResourceSetMapFlags_v3020CUPTI_RUNTIME_TRACE_CBID_cudaGraphicsMapResources_v3020CUPTI_RUNTIME_TRACE_CBID_cudaGraphicsUnmapResources_v3020CUPTI_RUNTIME_TRACE_CBID_cudaGraphicsResourceGetMappedPointer_v3020CUPTI_RUNTIME_TRACE_CBID_cudaGraphicsSubResourceGetMappedArray_v3020CUPTI_RUNTIME_TRACE_CBID_cudaVDPAUGetDevice_v3020CUPTI_RUNTIME_TRACE_CBID_cudaVDPAUSetVDPAUDevice_v3020CUPTI_RUNTIME_TRACE_CBID_cudaGraphicsVDPAURegisterVideoSurface_v3020CUPTI_RUNTIME_TRACE_CBID_cudaGraphicsVDPAURegisterOutputSurface_v3020CUPTI_RUNTIME_TRACE_CBID_cudaD3D11GetDevice_v3020CUPTI_RUNTIME_TRACE_CBID_cudaD3D11GetDevices_v3020CUPTI_RUNTIME_TRACE_CBID_cudaD3D11SetDirect3DDevice_v3020CUPTI_RUNTIME_TRACE_CBID_cudaGraphicsD3D11RegisterResource_v3020CUPTI_RUNTIME_TRACE_CBID_cudaD3D10GetDevice_v3020CUPTI_RUNTIME_TRACE_CBID_cudaD3D10GetDevices_v3020CUPTI_RUNTIME_TRACE_CBID_cudaD3D10SetDirect3DDevice_v3020CUPTI_RUNTIME_TRACE_CBID_cudaGraphicsD3D10RegisterResource_v3020CUPTI_RUNTIME_TRACE_CBID_cudaD3D10RegisterResource_v3020CUPTI_RUNTIME_TRACE_CBID_cudaD3D10UnregisterResource_v3020CUPTI_RUNTIME_TRACE_CBID_cudaD3D10MapResources_v3020CUPTI_RUNTIME_TRACE_CBID_cudaD3D10UnmapResources_v3020CUPTI_RUNTIME_TRACE_CBID_cudaD3D10ResourceSetMapFlags_v3020CUPTI_RUNTIME_TRACE_CBID_cudaD3D10ResourceGetSurfaceDimensions_v3020CUPTI_RUNTIME_TRACE_CBID_cudaD3D10ResourceGetMappedArray_v3020CUPTI_RUNTIME_TRACE_CBID_cudaD3D10ResourceGetMappedPointer_v3020CUPTI_RUNTIME_TRACE_CBID_cudaD3D10ResourceGetMappedSize_v3020CUPTI_RUNTIME_TRACE_CBID_cudaD3D10ResourceGetMappedPitch_v3020CUPTI_RUNTIME_TRACE_CBID_cudaD3D9GetDevice_v3020CUPTI_RUNTIME_TRACE_CBID_cudaD3D9GetDevices_v3020CUPTI_RUNTIME_TRACE_CBID_cudaD3D9SetDirect3DDevice_v3020CUPTI_RUNTIME_TRACE_CBID_cudaD3D9GetDirect3DDevice_v3020CUPTI_RUNTIME_TRACE_CBID_cudaGraphicsD3D9RegisterResource_v3020CUPTI_RUNTIME_TRACE_CBID_cudaD3D9RegisterResource_v3020CUPTI_RUNTIME_TRACE_CBID_cudaD3D9UnregisterResource_v3020CUPTI_RUNTIME_TRACE_CBID_cudaD3D9MapResources_v3020CUPTI_RUNTIME_TRACE_CBID_cudaD3D9UnmapResources_v3020CUPTI_RUNTIME_TRACE_CBID_cudaD3D9ResourceSetMapFlags_v3020CUPTI_RUNTIME_TRACE_CBID_cudaD3D9ResourceGetSurfaceDimensions_v3020CUPTI_RUNTIME_TRACE_CBID_cudaD3D9ResourceGetMappedArray_v3020CUPTI_RUNTIME_TRACE_CBID_cudaD3D9ResourceGetMappedPointer_v3020CUPTI_RUNTIME_TRACE_CBID_cudaD3D9ResourceGetMappedSize_v3020CUPTI_RUNTIME_TRACE_CBID_cudaD3D9ResourceGetMappedPitch_v3020CUPTI_RUNTIME_TRACE_CBID_cudaD3D9Begin_v3020CUPTI_RUNTIME_TRACE_CBID_cudaD3D9End_v3020CUPTI_RUNTIME_TRACE_CBID_cudaD3D9RegisterVertexBuffer_v3020CUPTI_RUNTIME_TRACE_CBID_cudaD3D9UnregisterVertexBuffer_v3020CUPTI_RUNTIME_TRACE_CBID_cudaD3D9MapVertexBuffer_v3020CUPTI_RUNTIME_TRACE_CBID_cudaD3D9UnmapVertexBuffer_v3020CUPTI_RUNTIME_TRACE_CBID_cudaThreadExit_v3020CUPTI_RUNTIME_TRACE_CBID_cudaSetDoubleForDevice_v3020CUPTI_RUNTIME_TRACE_CBID_cudaSetDoubleForHost_v3020CUPTI_RUNTIME_TRACE_CBID_cudaThreadSynchronize_v3020CUPTI_RUNTIME_TRACE_CBID_cudaThreadGetLimit_v3020CUPTI_RUNTIME_TRACE_CBID_cudaThreadSetLimit_v3020CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreate_v3020CUPTI_RUNTIME_TRACE_CBID_cudaStreamDestroy_v3020CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_v3020CUPTI_RUNTIME_TRACE_CBID_cudaStreamQuery_v3020CUPTI_RUNTIME_TRACE_CBID_cudaEventCreate_v3020CUPTI_RUNTIME_TRACE_CBID_cudaEventCreateWithFlags_v3020CUPTI_RUNTIME_TRACE_CBID_cudaEventRecord_v3020CUPTI_RUNTIME_TRACE_CBID_cudaEventDestroy_v3020CUPTI_RUNTIME_TRACE_CBID_cudaEventSynchronize_v3020CUPTI_RUNTIME_TRACE_CBID_cudaEventQuery_v3020CUPTI_RUNTIME_TRACE_CBID_cudaEventElapsedTime_v3020CUPTI_RUNTIME_TRACE_CBID_cudaMalloc3D_v3020CUPTI_RUNTIME_TRACE_CBID_cudaMalloc3DArray_v3020CUPTI_RUNTIME_TRACE_CBID_cudaMemset3D_v3020CUPTI_RUNTIME_TRACE_CBID_cudaMemset3DAsync_v3020CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3D_v3020CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3DAsync_v3020CUPTI_RUNTIME_TRACE_CBID_cudaThreadSetCacheConfig_v3020CUPTI_RUNTIME_TRACE_CBID_cudaStreamWaitEvent_v3020CUPTI_RUNTIME_TRACE_CBID_cudaD3D11GetDirect3DDevice_v3020CUPTI_RUNTIME_TRACE_CBID_cudaD3D10GetDirect3DDevice_v3020CUPTI_RUNTIME_TRACE_CBID_cudaThreadGetCacheConfig_v3020CUPTI_RUNTIME_TRACE_CBID_cudaPointerGetAttributes_v4000CUPTI_RUNTIME_TRACE_CBID_cudaHostRegister_v4000CUPTI_RUNTIME_TRACE_CBID_cudaHostUnregister_v4000CUPTI_RUNTIME_TRACE_CBID_cudaDeviceCanAccessPeer_v4000CUPTI_RUNTIME_TRACE_CBID_cudaDeviceEnablePeerAccess_v4000CUPTI_RUNTIME_TRACE_CBID_cudaDeviceDisablePeerAccess_v4000CUPTI_RUNTIME_TRACE_CBID_cudaPeerRegister_v4000CUPTI_RUNTIME_TRACE_CBID_cudaPeerUnregister_v4000CUPTI_RUNTIME_TRACE_CBID_cudaPeerGetDevicePointer_v4000CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyPeer_v4000CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyPeerAsync_v4000CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3DPeer_v4000CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3DPeerAsync_v4000CUPTI_RUNTIME_TRACE_CBID_cudaDeviceReset_v3020CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSynchronize_v3020CUPTI_RUNTIME_TRACE_CBID_cudaDeviceGetLimit_v3020CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSetLimit_v3020CUPTI_RUNTIME_TRACE_CBID_cudaDeviceGetCacheConfig_v3020CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSetCacheConfig_v3020CUPTI_RUNTIME_TRACE_CBID_cudaProfilerInitialize_v4000CUPTI_RUNTIME_TRACE_CBID_cudaProfilerStart_v4000CUPTI_RUNTIME_TRACE_CBID_cudaProfilerStop_v4000CUPTI_RUNTIME_TRACE_CBID_cudaDeviceGetByPCIBusId_v4010CUPTI_RUNTIME_TRACE_CBID_cudaDeviceGetPCIBusId_v4010CUPTI_RUNTIME_TRACE_CBID_cudaGLGetDevices_v4010CUPTI_RUNTIME_TRACE_CBID_cudaIpcGetEventHandle_v4010CUPTI_RUNTIME_TRACE_CBID_cudaIpcOpenEventHandle_v4010CUPTI_RUNTIME_TRACE_CBID_cudaIpcGetMemHandle_v4010CUPTI_RUNTIME_TRACE_CBID_cudaIpcOpenMemHandle_v4010CUPTI_RUNTIME_TRACE_CBID_cudaIpcCloseMemHandle_v4010CUPTI_RUNTIME_TRACE_CBID_cudaArrayGetInfo_v4010CUPTI_RUNTIME_TRACE_CBID_cudaFuncSetSharedMemConfig_v4020CUPTI_RUNTIME_TRACE_CBID_cudaDeviceGetSharedMemConfig_v4020CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSetSharedMemConfig_v4020CUPTI_RUNTIME_TRACE_CBID_cudaCreateTextureObject_v5000CUPTI_RUNTIME_TRACE_CBID_cudaDestroyTextureObject_v5000CUPTI_RUNTIME_TRACE_CBID_cudaGetTextureObjectResourceDesc_v5000CUPTI_RUNTIME_TRACE_CBID_cudaGetTextureObjectTextureDesc_v5000CUPTI_RUNTIME_TRACE_CBID_cudaCreateSurfaceObject_v5000CUPTI_RUNTIME_TRACE_CBID_cudaDestroySurfaceObject_v5000CUPTI_RUNTIME_TRACE_CBID_cudaGetSurfaceObjectResourceDesc_v5000CUPTI_RUNTIME_TRACE_CBID_cudaMallocMipmappedArray_v5000CUPTI_RUNTIME_TRACE_CBID_cudaGetMipmappedArrayLevel_v5000CUPTI_RUNTIME_TRACE_CBID_cudaFreeMipmappedArray_v5000CUPTI_RUNTIME_TRACE_CBID_cudaBindTextureToMipmappedArray_v5000CUPTI_RUNTIME_TRACE_CBID_cudaGraphicsResourceGetMappedMipmappedArray_v5000CUPTI_RUNTIME_TRACE_CBID_cudaStreamAddCallback_v5000CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreateWithFlags_v5000CUPTI_RUNTIME_TRACE_CBID_cudaGetTextureObjectResourceViewDesc_v5000CUPTI_RUNTIME_TRACE_CBID_cudaDeviceGetAttribute_v5000CUPTI_RUNTIME_TRACE_CBID_cudaStreamDestroy_v5050CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreateWithPriority_v5050CUPTI_RUNTIME_TRACE_CBID_cudaStreamGetPriority_v5050CUPTI_RUNTIME_TRACE_CBID_cudaStreamGetFlags_v5050CUPTI_RUNTIME_TRACE_CBID_cudaDeviceGetStreamPriorityRange_v5050CUPTI_RUNTIME_TRACE_CBID_cudaMallocManaged_v6000CUPTI_RUNTIME_TRACE_CBID_cudaOccupancyMaxActiveBlocksPerMultiprocessor_v6000CUPTI_RUNTIME_TRACE_CBID_cudaStreamAttachMemAsync_v6000CUPTI_RUNTIME_TRACE_CBID_cudaGetErrorName_v6050CUPTI_RUNTIME_TRACE_CBID_cudaOccupancyMaxActiveBlocksPerMultiprocessor_v6050CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000CUPTI_RUNTIME_TRACE_CBID_cudaGetDeviceFlags_v7000CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_ptsz_v7000CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_ptds_v7000CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2D_ptds_v7000CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyToArray_ptds_v7000CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DToArray_ptds_v7000CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyFromArray_ptds_v7000CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DFromArray_ptds_v7000CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyArrayToArray_ptds_v7000CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DArrayToArray_ptds_v7000CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyToSymbol_ptds_v7000CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyFromSymbol_ptds_v7000CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_ptsz_v7000CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyToArrayAsync_ptsz_v7000CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyFromArrayAsync_ptsz_v7000CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DAsync_ptsz_v7000CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DToArrayAsync_ptsz_v7000CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DFromArrayAsync_ptsz_v7000CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyToSymbolAsync_ptsz_v7000CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyFromSymbolAsync_ptsz_v7000CUPTI_RUNTIME_TRACE_CBID_cudaMemset_ptds_v7000CUPTI_RUNTIME_TRACE_CBID_cudaMemset2D_ptds_v7000CUPTI_RUNTIME_TRACE_CBID_cudaMemsetAsync_ptsz_v7000CUPTI_RUNTIME_TRACE_CBID_cudaMemset2DAsync_ptsz_v7000CUPTI_RUNTIME_TRACE_CBID_cudaStreamGetPriority_ptsz_v7000CUPTI_RUNTIME_TRACE_CBID_cudaStreamGetFlags_ptsz_v7000CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_ptsz_v7000CUPTI_RUNTIME_TRACE_CBID_cudaStreamQuery_ptsz_v7000CUPTI_RUNTIME_TRACE_CBID_cudaStreamAttachMemAsync_ptsz_v7000CUPTI_RUNTIME_TRACE_CBID_cudaEventRecord_ptsz_v7000CUPTI_RUNTIME_TRACE_CBID_cudaMemset3D_ptds_v7000CUPTI_RUNTIME_TRACE_CBID_cudaMemset3DAsync_ptsz_v7000CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3D_ptds_v7000CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3DAsync_ptsz_v7000CUPTI_RUNTIME_TRACE_CBID_cudaStreamWaitEvent_ptsz_v7000CUPTI_RUNTIME_TRACE_CBID_cudaStreamAddCallback_ptsz_v7000CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3DPeer_ptds_v7000CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3DPeerAsync_ptsz_v7000CUPTI_RUNTIME_TRACE_CBID_cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_v7000CUPTI_RUNTIME_TRACE_CBID_cudaMemPrefetchAsync_v8000CUPTI_RUNTIME_TRACE_CBID_cudaMemPrefetchAsync_ptsz_v8000CUPTI_RUNTIME_TRACE_CBID_cudaMemAdvise_v8000CUPTI_RUNTIME_TRACE_CBID_cudaDeviceGetP2PAttribute_v8000CUPTI_RUNTIME_TRACE_CBID_cudaGraphicsEGLRegisterImage_v7000CUPTI_RUNTIME_TRACE_CBID_cudaEGLStreamConsumerConnect_v7000CUPTI_RUNTIME_TRACE_CBID_cudaEGLStreamConsumerDisconnect_v7000CUPTI_RUNTIME_TRACE_CBID_cudaEGLStreamConsumerAcquireFrame_v7000CUPTI_RUNTIME_TRACE_CBID_cudaEGLStreamConsumerReleaseFrame_v7000CUPTI_RUNTIME_TRACE_CBID_cudaEGLStreamProducerConnect_v7000CUPTI_RUNTIME_TRACE_CBID_cudaEGLStreamProducerDisconnect_v7000CUPTI_RUNTIME_TRACE_CBID_cudaEGLStreamProducerPresentFrame_v7000CUPTI_RUNTIME_TRACE_CBID_cudaEGLStreamProducerReturnFrame_v7000CUPTI_RUNTIME_TRACE_CBID_cudaGraphicsResourceGetMappedEglFrame_v7000CUPTI_RUNTIME_TRACE_CBID_cudaMemRangeGetAttribute_v8000CUPTI_RUNTIME_TRACE_CBID_cudaMemRangeGetAttributes_v8000CUPTI_RUNTIME_TRACE_CBID_cudaEGLStreamConsumerConnectWithFlags_v7000CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_v9000CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_ptsz_v9000CUPTI_RUNTIME_TRACE_CBID_cudaEventCreateFromEGLSync_v9000CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernelMultiDevice_v9000CUPTI_RUNTIME_TRACE_CBID_cudaFuncSetAttribute_v9000CUPTI_RUNTIME_TRACE_CBID_SIZE"

var _CUPTI_RUNTIME_TRACE_CBID_index = [...]uint16{0, 32, 83, 135, 184, 238, 285, 334, 386, 434, 482, 529, 579, 628, 669, 722, 774, 818, 862, 912, 961, 1002, 1048, 1087, 1133, 1177, 1222, 1265, 1309, 1364, 1411, 1456, 1497, 1540, 1588, 1638, 1688, 1740, 1793, 1848, 1897, 1948, 1994, 2047, 2102, 2150, 2205, 2262, 2316, 2372, 2413, 2456, 2502, 2550, 2601, 2649, 2695, 2743, 2796, 2844, 2904, 2958, 3011, 3065, 3113, 3170, 3222, 3276, 3335, 3395, 3452, 3511, 3558, 3616, 3675, 3736, 3798, 3853, 3910, 3977, 4045, 4094, 4148, 4216, 4285, 4334, 4384, 4441, 4505, 4554, 4604, 4661, 4725, 4781, 4839, 4891, 4945, 5004, 5072, 5134, 5198, 5259, 5321, 5369, 5418, 5474, 5530, 5593, 5648, 5705, 5756, 5809, 5867, 5934, 5995, 6058, 6118, 6179, 6223, 6265, 6324, 6385, 6439, 6495, 6540, 6593, 6644, 6696, 6745, 6794, 6841, 6889, 6941, 6987, 7033, 7088, 7134, 7181, 7232, 7277, 7328, 7371, 7419, 7462, 7510, 7553, 7601, 7656, 7706, 7763, 7820, 7875, 7930, 7977, 8026, 8080, 8137, 8195, 8242, 8291, 8346, 8391, 8441, 8488, 8540, 8586, 8638, 8687, 8736, 8791, 8846, 8899, 8947, 8994, 9048, 9100, 9147, 9199, 9252, 9302, 9353, 9405, 9452, 9509, 9568, 9627, 9681, 9736, 9799, 9861, 9915, 9970, 10033, 10088, 10145, 10198, 10260, 10334, 10386, 10442, 10509, 10562, 10610, 10669, 10721, 10770, 10833, 10881, 10957, 11012, 11059, 11135, 11182, 11231, 11277, 11329, 11375, 11423, 11476, 11531, 11586, 11643, 11701, 11761, 11815, 11871, 11922, 11980, 12040, 12093, 12153, 12215, 12274, 12335, 12381, 12429, 12480, 12533, 12590, 12644, 12701, 12752, 12812, 12863, 12911, 12964, 13012, 13065, 13120, 13177, 13229, 13286, 13371, 13422, 13478, 13522, 13578, 13637, 13696, 13758, 13822, 13886, 13945, 14007, 14071, 14134, 14202, 14257, 14313, 14381, 14439, 14502, 14559, 14628, 14679, 14708}

func (i CUPTI_RUNTIME_TRACE_CBID) String() string {
	if i < 0 || i >= CUPTI_RUNTIME_TRACE_CBID(len(_CUPTI_RUNTIME_TRACE_CBID_index)-1) {
		return fmt.Sprintf("CUPTI_RUNTIME_TRACE_CBID(%d)", i)
	}
	return _CUPTI_RUNTIME_TRACE_CBID_name[_CUPTI_RUNTIME_TRACE_CBID_index[i]:_CUPTI_RUNTIME_TRACE_CBID_index[i+1]]
}

var _CUPTI_RUNTIME_TRACE_CBIDNameToValue_map = map[string]CUPTI_RUNTIME_TRACE_CBID{
	_CUPTI_RUNTIME_TRACE_CBID_name[0:32]:        0,
	_CUPTI_RUNTIME_TRACE_CBID_name[32:83]:       1,
	_CUPTI_RUNTIME_TRACE_CBID_name[83:135]:      2,
	_CUPTI_RUNTIME_TRACE_CBID_name[135:184]:     3,
	_CUPTI_RUNTIME_TRACE_CBID_name[184:238]:     4,
	_CUPTI_RUNTIME_TRACE_CBID_name[238:285]:     5,
	_CUPTI_RUNTIME_TRACE_CBID_name[285:334]:     6,
	_CUPTI_RUNTIME_TRACE_CBID_name[334:386]:     7,
	_CUPTI_RUNTIME_TRACE_CBID_name[386:434]:     8,
	_CUPTI_RUNTIME_TRACE_CBID_name[434:482]:     9,
	_CUPTI_RUNTIME_TRACE_CBID_name[482:529]:     10,
	_CUPTI_RUNTIME_TRACE_CBID_name[529:579]:     11,
	_CUPTI_RUNTIME_TRACE_CBID_name[579:628]:     12,
	_CUPTI_RUNTIME_TRACE_CBID_name[628:669]:     13,
	_CUPTI_RUNTIME_TRACE_CBID_name[669:722]:     14,
	_CUPTI_RUNTIME_TRACE_CBID_name[722:774]:     15,
	_CUPTI_RUNTIME_TRACE_CBID_name[774:818]:     16,
	_CUPTI_RUNTIME_TRACE_CBID_name[818:862]:     17,
	_CUPTI_RUNTIME_TRACE_CBID_name[862:912]:     18,
	_CUPTI_RUNTIME_TRACE_CBID_name[912:961]:     19,
	_CUPTI_RUNTIME_TRACE_CBID_name[961:1002]:    20,
	_CUPTI_RUNTIME_TRACE_CBID_name[1002:1048]:   21,
	_CUPTI_RUNTIME_TRACE_CBID_name[1048:1087]:   22,
	_CUPTI_RUNTIME_TRACE_CBID_name[1087:1133]:   23,
	_CUPTI_RUNTIME_TRACE_CBID_name[1133:1177]:   24,
	_CUPTI_RUNTIME_TRACE_CBID_name[1177:1222]:   25,
	_CUPTI_RUNTIME_TRACE_CBID_name[1222:1265]:   26,
	_CUPTI_RUNTIME_TRACE_CBID_name[1265:1309]:   27,
	_CUPTI_RUNTIME_TRACE_CBID_name[1309:1364]:   28,
	_CUPTI_RUNTIME_TRACE_CBID_name[1364:1411]:   29,
	_CUPTI_RUNTIME_TRACE_CBID_name[1411:1456]:   30,
	_CUPTI_RUNTIME_TRACE_CBID_name[1456:1497]:   31,
	_CUPTI_RUNTIME_TRACE_CBID_name[1497:1540]:   32,
	_CUPTI_RUNTIME_TRACE_CBID_name[1540:1588]:   33,
	_CUPTI_RUNTIME_TRACE_CBID_name[1588:1638]:   34,
	_CUPTI_RUNTIME_TRACE_CBID_name[1638:1688]:   35,
	_CUPTI_RUNTIME_TRACE_CBID_name[1688:1740]:   36,
	_CUPTI_RUNTIME_TRACE_CBID_name[1740:1793]:   37,
	_CUPTI_RUNTIME_TRACE_CBID_name[1793:1848]:   38,
	_CUPTI_RUNTIME_TRACE_CBID_name[1848:1897]:   39,
	_CUPTI_RUNTIME_TRACE_CBID_name[1897:1948]:   40,
	_CUPTI_RUNTIME_TRACE_CBID_name[1948:1994]:   41,
	_CUPTI_RUNTIME_TRACE_CBID_name[1994:2047]:   42,
	_CUPTI_RUNTIME_TRACE_CBID_name[2047:2102]:   43,
	_CUPTI_RUNTIME_TRACE_CBID_name[2102:2150]:   44,
	_CUPTI_RUNTIME_TRACE_CBID_name[2150:2205]:   45,
	_CUPTI_RUNTIME_TRACE_CBID_name[2205:2262]:   46,
	_CUPTI_RUNTIME_TRACE_CBID_name[2262:2316]:   47,
	_CUPTI_RUNTIME_TRACE_CBID_name[2316:2372]:   48,
	_CUPTI_RUNTIME_TRACE_CBID_name[2372:2413]:   49,
	_CUPTI_RUNTIME_TRACE_CBID_name[2413:2456]:   50,
	_CUPTI_RUNTIME_TRACE_CBID_name[2456:2502]:   51,
	_CUPTI_RUNTIME_TRACE_CBID_name[2502:2550]:   52,
	_CUPTI_RUNTIME_TRACE_CBID_name[2550:2601]:   53,
	_CUPTI_RUNTIME_TRACE_CBID_name[2601:2649]:   54,
	_CUPTI_RUNTIME_TRACE_CBID_name[2649:2695]:   55,
	_CUPTI_RUNTIME_TRACE_CBID_name[2695:2743]:   56,
	_CUPTI_RUNTIME_TRACE_CBID_name[2743:2796]:   57,
	_CUPTI_RUNTIME_TRACE_CBID_name[2796:2844]:   58,
	_CUPTI_RUNTIME_TRACE_CBID_name[2844:2904]:   59,
	_CUPTI_RUNTIME_TRACE_CBID_name[2904:2958]:   60,
	_CUPTI_RUNTIME_TRACE_CBID_name[2958:3011]:   61,
	_CUPTI_RUNTIME_TRACE_CBID_name[3011:3065]:   62,
	_CUPTI_RUNTIME_TRACE_CBID_name[3065:3113]:   63,
	_CUPTI_RUNTIME_TRACE_CBID_name[3113:3170]:   64,
	_CUPTI_RUNTIME_TRACE_CBID_name[3170:3222]:   65,
	_CUPTI_RUNTIME_TRACE_CBID_name[3222:3276]:   66,
	_CUPTI_RUNTIME_TRACE_CBID_name[3276:3335]:   67,
	_CUPTI_RUNTIME_TRACE_CBID_name[3335:3395]:   68,
	_CUPTI_RUNTIME_TRACE_CBID_name[3395:3452]:   69,
	_CUPTI_RUNTIME_TRACE_CBID_name[3452:3511]:   70,
	_CUPTI_RUNTIME_TRACE_CBID_name[3511:3558]:   71,
	_CUPTI_RUNTIME_TRACE_CBID_name[3558:3616]:   72,
	_CUPTI_RUNTIME_TRACE_CBID_name[3616:3675]:   73,
	_CUPTI_RUNTIME_TRACE_CBID_name[3675:3736]:   74,
	_CUPTI_RUNTIME_TRACE_CBID_name[3736:3798]:   75,
	_CUPTI_RUNTIME_TRACE_CBID_name[3798:3853]:   76,
	_CUPTI_RUNTIME_TRACE_CBID_name[3853:3910]:   77,
	_CUPTI_RUNTIME_TRACE_CBID_name[3910:3977]:   78,
	_CUPTI_RUNTIME_TRACE_CBID_name[3977:4045]:   79,
	_CUPTI_RUNTIME_TRACE_CBID_name[4045:4094]:   80,
	_CUPTI_RUNTIME_TRACE_CBID_name[4094:4148]:   81,
	_CUPTI_RUNTIME_TRACE_CBID_name[4148:4216]:   82,
	_CUPTI_RUNTIME_TRACE_CBID_name[4216:4285]:   83,
	_CUPTI_RUNTIME_TRACE_CBID_name[4285:4334]:   84,
	_CUPTI_RUNTIME_TRACE_CBID_name[4334:4384]:   85,
	_CUPTI_RUNTIME_TRACE_CBID_name[4384:4441]:   86,
	_CUPTI_RUNTIME_TRACE_CBID_name[4441:4505]:   87,
	_CUPTI_RUNTIME_TRACE_CBID_name[4505:4554]:   88,
	_CUPTI_RUNTIME_TRACE_CBID_name[4554:4604]:   89,
	_CUPTI_RUNTIME_TRACE_CBID_name[4604:4661]:   90,
	_CUPTI_RUNTIME_TRACE_CBID_name[4661:4725]:   91,
	_CUPTI_RUNTIME_TRACE_CBID_name[4725:4781]:   92,
	_CUPTI_RUNTIME_TRACE_CBID_name[4781:4839]:   93,
	_CUPTI_RUNTIME_TRACE_CBID_name[4839:4891]:   94,
	_CUPTI_RUNTIME_TRACE_CBID_name[4891:4945]:   95,
	_CUPTI_RUNTIME_TRACE_CBID_name[4945:5004]:   96,
	_CUPTI_RUNTIME_TRACE_CBID_name[5004:5072]:   97,
	_CUPTI_RUNTIME_TRACE_CBID_name[5072:5134]:   98,
	_CUPTI_RUNTIME_TRACE_CBID_name[5134:5198]:   99,
	_CUPTI_RUNTIME_TRACE_CBID_name[5198:5259]:   100,
	_CUPTI_RUNTIME_TRACE_CBID_name[5259:5321]:   101,
	_CUPTI_RUNTIME_TRACE_CBID_name[5321:5369]:   102,
	_CUPTI_RUNTIME_TRACE_CBID_name[5369:5418]:   103,
	_CUPTI_RUNTIME_TRACE_CBID_name[5418:5474]:   104,
	_CUPTI_RUNTIME_TRACE_CBID_name[5474:5530]:   105,
	_CUPTI_RUNTIME_TRACE_CBID_name[5530:5593]:   106,
	_CUPTI_RUNTIME_TRACE_CBID_name[5593:5648]:   107,
	_CUPTI_RUNTIME_TRACE_CBID_name[5648:5705]:   108,
	_CUPTI_RUNTIME_TRACE_CBID_name[5705:5756]:   109,
	_CUPTI_RUNTIME_TRACE_CBID_name[5756:5809]:   110,
	_CUPTI_RUNTIME_TRACE_CBID_name[5809:5867]:   111,
	_CUPTI_RUNTIME_TRACE_CBID_name[5867:5934]:   112,
	_CUPTI_RUNTIME_TRACE_CBID_name[5934:5995]:   113,
	_CUPTI_RUNTIME_TRACE_CBID_name[5995:6058]:   114,
	_CUPTI_RUNTIME_TRACE_CBID_name[6058:6118]:   115,
	_CUPTI_RUNTIME_TRACE_CBID_name[6118:6179]:   116,
	_CUPTI_RUNTIME_TRACE_CBID_name[6179:6223]:   117,
	_CUPTI_RUNTIME_TRACE_CBID_name[6223:6265]:   118,
	_CUPTI_RUNTIME_TRACE_CBID_name[6265:6324]:   119,
	_CUPTI_RUNTIME_TRACE_CBID_name[6324:6385]:   120,
	_CUPTI_RUNTIME_TRACE_CBID_name[6385:6439]:   121,
	_CUPTI_RUNTIME_TRACE_CBID_name[6439:6495]:   122,
	_CUPTI_RUNTIME_TRACE_CBID_name[6495:6540]:   123,
	_CUPTI_RUNTIME_TRACE_CBID_name[6540:6593]:   124,
	_CUPTI_RUNTIME_TRACE_CBID_name[6593:6644]:   125,
	_CUPTI_RUNTIME_TRACE_CBID_name[6644:6696]:   126,
	_CUPTI_RUNTIME_TRACE_CBID_name[6696:6745]:   127,
	_CUPTI_RUNTIME_TRACE_CBID_name[6745:6794]:   128,
	_CUPTI_RUNTIME_TRACE_CBID_name[6794:6841]:   129,
	_CUPTI_RUNTIME_TRACE_CBID_name[6841:6889]:   130,
	_CUPTI_RUNTIME_TRACE_CBID_name[6889:6941]:   131,
	_CUPTI_RUNTIME_TRACE_CBID_name[6941:6987]:   132,
	_CUPTI_RUNTIME_TRACE_CBID_name[6987:7033]:   133,
	_CUPTI_RUNTIME_TRACE_CBID_name[7033:7088]:   134,
	_CUPTI_RUNTIME_TRACE_CBID_name[7088:7134]:   135,
	_CUPTI_RUNTIME_TRACE_CBID_name[7134:7181]:   136,
	_CUPTI_RUNTIME_TRACE_CBID_name[7181:7232]:   137,
	_CUPTI_RUNTIME_TRACE_CBID_name[7232:7277]:   138,
	_CUPTI_RUNTIME_TRACE_CBID_name[7277:7328]:   139,
	_CUPTI_RUNTIME_TRACE_CBID_name[7328:7371]:   140,
	_CUPTI_RUNTIME_TRACE_CBID_name[7371:7419]:   141,
	_CUPTI_RUNTIME_TRACE_CBID_name[7419:7462]:   142,
	_CUPTI_RUNTIME_TRACE_CBID_name[7462:7510]:   143,
	_CUPTI_RUNTIME_TRACE_CBID_name[7510:7553]:   144,
	_CUPTI_RUNTIME_TRACE_CBID_name[7553:7601]:   145,
	_CUPTI_RUNTIME_TRACE_CBID_name[7601:7656]:   146,
	_CUPTI_RUNTIME_TRACE_CBID_name[7656:7706]:   147,
	_CUPTI_RUNTIME_TRACE_CBID_name[7706:7763]:   148,
	_CUPTI_RUNTIME_TRACE_CBID_name[7763:7820]:   149,
	_CUPTI_RUNTIME_TRACE_CBID_name[7820:7875]:   150,
	_CUPTI_RUNTIME_TRACE_CBID_name[7875:7930]:   151,
	_CUPTI_RUNTIME_TRACE_CBID_name[7930:7977]:   152,
	_CUPTI_RUNTIME_TRACE_CBID_name[7977:8026]:   153,
	_CUPTI_RUNTIME_TRACE_CBID_name[8026:8080]:   154,
	_CUPTI_RUNTIME_TRACE_CBID_name[8080:8137]:   155,
	_CUPTI_RUNTIME_TRACE_CBID_name[8137:8195]:   156,
	_CUPTI_RUNTIME_TRACE_CBID_name[8195:8242]:   157,
	_CUPTI_RUNTIME_TRACE_CBID_name[8242:8291]:   158,
	_CUPTI_RUNTIME_TRACE_CBID_name[8291:8346]:   159,
	_CUPTI_RUNTIME_TRACE_CBID_name[8346:8391]:   160,
	_CUPTI_RUNTIME_TRACE_CBID_name[8391:8441]:   161,
	_CUPTI_RUNTIME_TRACE_CBID_name[8441:8488]:   162,
	_CUPTI_RUNTIME_TRACE_CBID_name[8488:8540]:   163,
	_CUPTI_RUNTIME_TRACE_CBID_name[8540:8586]:   164,
	_CUPTI_RUNTIME_TRACE_CBID_name[8586:8638]:   165,
	_CUPTI_RUNTIME_TRACE_CBID_name[8638:8687]:   166,
	_CUPTI_RUNTIME_TRACE_CBID_name[8687:8736]:   167,
	_CUPTI_RUNTIME_TRACE_CBID_name[8736:8791]:   168,
	_CUPTI_RUNTIME_TRACE_CBID_name[8791:8846]:   169,
	_CUPTI_RUNTIME_TRACE_CBID_name[8846:8899]:   170,
	_CUPTI_RUNTIME_TRACE_CBID_name[8899:8947]:   171,
	_CUPTI_RUNTIME_TRACE_CBID_name[8947:8994]:   172,
	_CUPTI_RUNTIME_TRACE_CBID_name[8994:9048]:   173,
	_CUPTI_RUNTIME_TRACE_CBID_name[9048:9100]:   174,
	_CUPTI_RUNTIME_TRACE_CBID_name[9100:9147]:   175,
	_CUPTI_RUNTIME_TRACE_CBID_name[9147:9199]:   176,
	_CUPTI_RUNTIME_TRACE_CBID_name[9199:9252]:   177,
	_CUPTI_RUNTIME_TRACE_CBID_name[9252:9302]:   178,
	_CUPTI_RUNTIME_TRACE_CBID_name[9302:9353]:   179,
	_CUPTI_RUNTIME_TRACE_CBID_name[9353:9405]:   180,
	_CUPTI_RUNTIME_TRACE_CBID_name[9405:9452]:   181,
	_CUPTI_RUNTIME_TRACE_CBID_name[9452:9509]:   182,
	_CUPTI_RUNTIME_TRACE_CBID_name[9509:9568]:   183,
	_CUPTI_RUNTIME_TRACE_CBID_name[9568:9627]:   184,
	_CUPTI_RUNTIME_TRACE_CBID_name[9627:9681]:   185,
	_CUPTI_RUNTIME_TRACE_CBID_name[9681:9736]:   186,
	_CUPTI_RUNTIME_TRACE_CBID_name[9736:9799]:   187,
	_CUPTI_RUNTIME_TRACE_CBID_name[9799:9861]:   188,
	_CUPTI_RUNTIME_TRACE_CBID_name[9861:9915]:   189,
	_CUPTI_RUNTIME_TRACE_CBID_name[9915:9970]:   190,
	_CUPTI_RUNTIME_TRACE_CBID_name[9970:10033]:  191,
	_CUPTI_RUNTIME_TRACE_CBID_name[10033:10088]: 192,
	_CUPTI_RUNTIME_TRACE_CBID_name[10088:10145]: 193,
	_CUPTI_RUNTIME_TRACE_CBID_name[10145:10198]: 194,
	_CUPTI_RUNTIME_TRACE_CBID_name[10198:10260]: 195,
	_CUPTI_RUNTIME_TRACE_CBID_name[10260:10334]: 196,
	_CUPTI_RUNTIME_TRACE_CBID_name[10334:10386]: 197,
	_CUPTI_RUNTIME_TRACE_CBID_name[10386:10442]: 198,
	_CUPTI_RUNTIME_TRACE_CBID_name[10442:10509]: 199,
	_CUPTI_RUNTIME_TRACE_CBID_name[10509:10562]: 200,
	_CUPTI_RUNTIME_TRACE_CBID_name[10562:10610]: 201,
	_CUPTI_RUNTIME_TRACE_CBID_name[10610:10669]: 202,
	_CUPTI_RUNTIME_TRACE_CBID_name[10669:10721]: 203,
	_CUPTI_RUNTIME_TRACE_CBID_name[10721:10770]: 204,
	_CUPTI_RUNTIME_TRACE_CBID_name[10770:10833]: 205,
	_CUPTI_RUNTIME_TRACE_CBID_name[10833:10881]: 206,
	_CUPTI_RUNTIME_TRACE_CBID_name[10881:10957]: 207,
	_CUPTI_RUNTIME_TRACE_CBID_name[10957:11012]: 208,
	_CUPTI_RUNTIME_TRACE_CBID_name[11012:11059]: 209,
	_CUPTI_RUNTIME_TRACE_CBID_name[11059:11135]: 210,
	_CUPTI_RUNTIME_TRACE_CBID_name[11135:11182]: 211,
	_CUPTI_RUNTIME_TRACE_CBID_name[11182:11231]: 212,
	_CUPTI_RUNTIME_TRACE_CBID_name[11231:11277]: 213,
	_CUPTI_RUNTIME_TRACE_CBID_name[11277:11329]: 214,
	_CUPTI_RUNTIME_TRACE_CBID_name[11329:11375]: 215,
	_CUPTI_RUNTIME_TRACE_CBID_name[11375:11423]: 216,
	_CUPTI_RUNTIME_TRACE_CBID_name[11423:11476]: 217,
	_CUPTI_RUNTIME_TRACE_CBID_name[11476:11531]: 218,
	_CUPTI_RUNTIME_TRACE_CBID_name[11531:11586]: 219,
	_CUPTI_RUNTIME_TRACE_CBID_name[11586:11643]: 220,
	_CUPTI_RUNTIME_TRACE_CBID_name[11643:11701]: 221,
	_CUPTI_RUNTIME_TRACE_CBID_name[11701:11761]: 222,
	_CUPTI_RUNTIME_TRACE_CBID_name[11761:11815]: 223,
	_CUPTI_RUNTIME_TRACE_CBID_name[11815:11871]: 224,
	_CUPTI_RUNTIME_TRACE_CBID_name[11871:11922]: 225,
	_CUPTI_RUNTIME_TRACE_CBID_name[11922:11980]: 226,
	_CUPTI_RUNTIME_TRACE_CBID_name[11980:12040]: 227,
	_CUPTI_RUNTIME_TRACE_CBID_name[12040:12093]: 228,
	_CUPTI_RUNTIME_TRACE_CBID_name[12093:12153]: 229,
	_CUPTI_RUNTIME_TRACE_CBID_name[12153:12215]: 230,
	_CUPTI_RUNTIME_TRACE_CBID_name[12215:12274]: 231,
	_CUPTI_RUNTIME_TRACE_CBID_name[12274:12335]: 232,
	_CUPTI_RUNTIME_TRACE_CBID_name[12335:12381]: 233,
	_CUPTI_RUNTIME_TRACE_CBID_name[12381:12429]: 234,
	_CUPTI_RUNTIME_TRACE_CBID_name[12429:12480]: 235,
	_CUPTI_RUNTIME_TRACE_CBID_name[12480:12533]: 236,
	_CUPTI_RUNTIME_TRACE_CBID_name[12533:12590]: 237,
	_CUPTI_RUNTIME_TRACE_CBID_name[12590:12644]: 238,
	_CUPTI_RUNTIME_TRACE_CBID_name[12644:12701]: 239,
	_CUPTI_RUNTIME_TRACE_CBID_name[12701:12752]: 240,
	_CUPTI_RUNTIME_TRACE_CBID_name[12752:12812]: 241,
	_CUPTI_RUNTIME_TRACE_CBID_name[12812:12863]: 242,
	_CUPTI_RUNTIME_TRACE_CBID_name[12863:12911]: 243,
	_CUPTI_RUNTIME_TRACE_CBID_name[12911:12964]: 244,
	_CUPTI_RUNTIME_TRACE_CBID_name[12964:13012]: 245,
	_CUPTI_RUNTIME_TRACE_CBID_name[13012:13065]: 246,
	_CUPTI_RUNTIME_TRACE_CBID_name[13065:13120]: 247,
	_CUPTI_RUNTIME_TRACE_CBID_name[13120:13177]: 248,
	_CUPTI_RUNTIME_TRACE_CBID_name[13177:13229]: 249,
	_CUPTI_RUNTIME_TRACE_CBID_name[13229:13286]: 250,
	_CUPTI_RUNTIME_TRACE_CBID_name[13286:13371]: 251,
	_CUPTI_RUNTIME_TRACE_CBID_name[13371:13422]: 252,
	_CUPTI_RUNTIME_TRACE_CBID_name[13422:13478]: 253,
	_CUPTI_RUNTIME_TRACE_CBID_name[13478:13522]: 254,
	_CUPTI_RUNTIME_TRACE_CBID_name[13522:13578]: 255,
	_CUPTI_RUNTIME_TRACE_CBID_name[13578:13637]: 256,
	_CUPTI_RUNTIME_TRACE_CBID_name[13637:13696]: 257,
	_CUPTI_RUNTIME_TRACE_CBID_name[13696:13758]: 258,
	_CUPTI_RUNTIME_TRACE_CBID_name[13758:13822]: 259,
	_CUPTI_RUNTIME_TRACE_CBID_name[13822:13886]: 260,
	_CUPTI_RUNTIME_TRACE_CBID_name[13886:13945]: 261,
	_CUPTI_RUNTIME_TRACE_CBID_name[13945:14007]: 262,
	_CUPTI_RUNTIME_TRACE_CBID_name[14007:14071]: 263,
	_CUPTI_RUNTIME_TRACE_CBID_name[14071:14134]: 264,
	_CUPTI_RUNTIME_TRACE_CBID_name[14134:14202]: 265,
	_CUPTI_RUNTIME_TRACE_CBID_name[14202:14257]: 266,
	_CUPTI_RUNTIME_TRACE_CBID_name[14257:14313]: 267,
	_CUPTI_RUNTIME_TRACE_CBID_name[14313:14381]: 268,
	_CUPTI_RUNTIME_TRACE_CBID_name[14381:14439]: 269,
	_CUPTI_RUNTIME_TRACE_CBID_name[14439:14502]: 270,
	_CUPTI_RUNTIME_TRACE_CBID_name[14502:14559]: 271,
	_CUPTI_RUNTIME_TRACE_CBID_name[14559:14628]: 272,
	_CUPTI_RUNTIME_TRACE_CBID_name[14628:14679]: 273,
	_CUPTI_RUNTIME_TRACE_CBID_name[14679:14708]: 274,
}

func CUPTI_RUNTIME_TRACE_CBIDString(s string) (CUPTI_RUNTIME_TRACE_CBID, error) {
	if val, ok := _CUPTI_RUNTIME_TRACE_CBIDNameToValue_map[s]; ok {
		return val, nil
	}
	return 0, fmt.Errorf("%s does not belong to CUPTI_RUNTIME_TRACE_CBID values", s)
}

func (i CUPTI_RUNTIME_TRACE_CBID) MarshalJSON() ([]byte, error) {
	return json.Marshal(i.String())
}

func (i *CUPTI_RUNTIME_TRACE_CBID) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err != nil {
		return fmt.Errorf("CUPTI_RUNTIME_TRACE_CBID should be a string, got %s", data)
	}

	var err error
	*i, err = CUPTI_RUNTIME_TRACE_CBIDString(s)
	return err
}
