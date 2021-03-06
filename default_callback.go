package cupti

var (
	DefaultCallbacks = []string{
		// "CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel", // Not needed if enabling the runtime *cudaLaunch*
		"CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2",
		"CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2",
		"CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2",
		"CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2",
		"CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD_v2",
		"CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2",
		"CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSynchronize_v3020",
		"CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_v3020",
		"CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020",
		"CUPTI_RUNTIME_TRACE_CBID_cudaMallocArray_v3020",
		"CUPTI_RUNTIME_TRACE_CBID_cudaMallocHost_v3020",
		"CUPTI_RUNTIME_TRACE_CBID_cudaHostAlloc_v3020",
		"CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020",
		"CUPTI_RUNTIME_TRACE_CBID_cudaFreeHost_v3020",
		"CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020",
		"CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000",
		"CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020",
		"CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020",
		"CUPTI_RUNTIME_TRACE_CBID_cudaSetupArgument_v3020",
		"CUPTI_RUNTIME_TRACE_CBID_cudaMallocManaged_v6000",
		"CUPTI_RUNTIME_TRACE_CBID_cudaMallocPitch_v3020",
		"CUPTI_RUNTIME_TRACE_CBID_cudaThreadSynchronize_v3020",
		"CUPTI_RUNTIME_TRACE_CBID_cudaIpcGetEventHandle_v4010",
		"CUPTI_RUNTIME_TRACE_CBID_cudaIpcOpenEventHandle_v4010",
		"CUPTI_RUNTIME_TRACE_CBID_cudaIpcGetMemHandle_v4010",
		"CUPTI_RUNTIME_TRACE_CBID_cudaIpcOpenMemHandle_v4010",
		"CUPTI_RUNTIME_TRACE_CBID_cudaIpcCloseMemHandle_v4010",
		"CUPTI_CBID_RESOURCE_CONTEXT_CREATED",
		"CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING",
	}
)
