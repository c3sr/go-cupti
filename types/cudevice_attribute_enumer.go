// Code generated by "enumer -type=CUdevice_attribute -json -text -yaml -sql"; DO NOT EDIT.

//
package types

import (
	"database/sql/driver"
	"encoding/json"
	"fmt"
)

const _CUdevice_attributeName = "CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCKCU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_XCU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_YCU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_ZCU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_XCU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_YCU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_ZCU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCKCU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORYCU_DEVICE_ATTRIBUTE_WARP_SIZECU_DEVICE_ATTRIBUTE_MAX_PITCHCU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCKCU_DEVICE_ATTRIBUTE_CLOCK_RATECU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENTCU_DEVICE_ATTRIBUTE_GPU_OVERLAPCU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNTCU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUTCU_DEVICE_ATTRIBUTE_INTEGRATEDCU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORYCU_DEVICE_ATTRIBUTE_COMPUTE_MODECU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTHCU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTHCU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHTCU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTHCU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHTCU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTHCU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTHCU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHTCU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERSCU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENTCU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELSCU_DEVICE_ATTRIBUTE_ECC_ENABLEDCU_DEVICE_ATTRIBUTE_PCI_BUS_IDCU_DEVICE_ATTRIBUTE_PCI_DEVICE_IDCU_DEVICE_ATTRIBUTE_TCC_DRIVERCU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATECU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTHCU_DEVICE_ATTRIBUTE_L2_CACHE_SIZECU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSORCU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNTCU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSINGCU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTHCU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERSCU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHERCU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTHCU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHTCU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATECU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATECU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATECU_DEVICE_ATTRIBUTE_PCI_DOMAIN_IDCU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENTCU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTHCU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTHCU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERSCU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTHCU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTHCU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHTCU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTHCU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHTCU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTHCU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTHCU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERSCU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTHCU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHTCU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERSCU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTHCU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTHCU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERSCU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTHCU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTHCU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHTCU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCHCU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTHCU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHTCU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJORCU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINORCU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTHCU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTEDCU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTEDCU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTEDCU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSORCU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSORCU_DEVICE_ATTRIBUTE_MANAGED_MEMORYCU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARDCU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_IDCU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTEDCU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIOCU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESSCU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESSCU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTEDCU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEMCU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPSCU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPSCU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NORCU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCHCU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCHCU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN"

var _CUdevice_attributeIndex = [...]uint16{0, 41, 76, 111, 146, 180, 214, 248, 295, 336, 365, 394, 437, 467, 504, 535, 575, 614, 644, 683, 715, 758, 801, 845, 888, 932, 975, 1026, 1078, 1130, 1167, 1205, 1236, 1266, 1299, 1329, 1366, 1409, 1442, 1492, 1530, 1568, 1619, 1671, 1707, 1757, 1808, 1861, 1915, 1968, 2001, 2044, 2092, 2148, 2205, 2248, 2291, 2335, 2378, 2422, 2465, 2516, 2568, 2619, 2671, 2723, 2771, 2827, 2884, 2934, 2984, 3035, 3085, 3138, 3192, 3236, 3280, 3333, 3380, 3425, 3469, 3525, 3577, 3611, 3646, 3690, 3738, 3795, 3837, 3882, 3930, 3989, 4031, 4080, 4129, 4167, 4218, 4271}

func (i CUdevice_attribute) String() string {
	i -= 1
	if i < 0 || i >= CUdevice_attribute(len(_CUdevice_attributeIndex)-1) {
		return fmt.Sprintf("CUdevice_attribute(%d)", i+1)
	}
	return _CUdevice_attributeName[_CUdevice_attributeIndex[i]:_CUdevice_attributeIndex[i+1]]
}

var _CUdevice_attributeValues = []CUdevice_attribute{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97}

var _CUdevice_attributeNameToValueMap = map[string]CUdevice_attribute{
	_CUdevice_attributeName[0:41]:      1,
	_CUdevice_attributeName[41:76]:     2,
	_CUdevice_attributeName[76:111]:    3,
	_CUdevice_attributeName[111:146]:   4,
	_CUdevice_attributeName[146:180]:   5,
	_CUdevice_attributeName[180:214]:   6,
	_CUdevice_attributeName[214:248]:   7,
	_CUdevice_attributeName[248:295]:   8,
	_CUdevice_attributeName[295:336]:   9,
	_CUdevice_attributeName[336:365]:   10,
	_CUdevice_attributeName[365:394]:   11,
	_CUdevice_attributeName[394:437]:   12,
	_CUdevice_attributeName[437:467]:   13,
	_CUdevice_attributeName[467:504]:   14,
	_CUdevice_attributeName[504:535]:   15,
	_CUdevice_attributeName[535:575]:   16,
	_CUdevice_attributeName[575:614]:   17,
	_CUdevice_attributeName[614:644]:   18,
	_CUdevice_attributeName[644:683]:   19,
	_CUdevice_attributeName[683:715]:   20,
	_CUdevice_attributeName[715:758]:   21,
	_CUdevice_attributeName[758:801]:   22,
	_CUdevice_attributeName[801:845]:   23,
	_CUdevice_attributeName[845:888]:   24,
	_CUdevice_attributeName[888:932]:   25,
	_CUdevice_attributeName[932:975]:   26,
	_CUdevice_attributeName[975:1026]:  27,
	_CUdevice_attributeName[1026:1078]: 28,
	_CUdevice_attributeName[1078:1130]: 29,
	_CUdevice_attributeName[1130:1167]: 30,
	_CUdevice_attributeName[1167:1205]: 31,
	_CUdevice_attributeName[1205:1236]: 32,
	_CUdevice_attributeName[1236:1266]: 33,
	_CUdevice_attributeName[1266:1299]: 34,
	_CUdevice_attributeName[1299:1329]: 35,
	_CUdevice_attributeName[1329:1366]: 36,
	_CUdevice_attributeName[1366:1409]: 37,
	_CUdevice_attributeName[1409:1442]: 38,
	_CUdevice_attributeName[1442:1492]: 39,
	_CUdevice_attributeName[1492:1530]: 40,
	_CUdevice_attributeName[1530:1568]: 41,
	_CUdevice_attributeName[1568:1619]: 42,
	_CUdevice_attributeName[1619:1671]: 43,
	_CUdevice_attributeName[1671:1707]: 44,
	_CUdevice_attributeName[1707:1757]: 45,
	_CUdevice_attributeName[1757:1808]: 46,
	_CUdevice_attributeName[1808:1861]: 47,
	_CUdevice_attributeName[1861:1915]: 48,
	_CUdevice_attributeName[1915:1968]: 49,
	_CUdevice_attributeName[1968:2001]: 50,
	_CUdevice_attributeName[2001:2044]: 51,
	_CUdevice_attributeName[2044:2092]: 52,
	_CUdevice_attributeName[2092:2148]: 53,
	_CUdevice_attributeName[2148:2205]: 54,
	_CUdevice_attributeName[2205:2248]: 55,
	_CUdevice_attributeName[2248:2291]: 56,
	_CUdevice_attributeName[2291:2335]: 57,
	_CUdevice_attributeName[2335:2378]: 58,
	_CUdevice_attributeName[2378:2422]: 59,
	_CUdevice_attributeName[2422:2465]: 60,
	_CUdevice_attributeName[2465:2516]: 61,
	_CUdevice_attributeName[2516:2568]: 62,
	_CUdevice_attributeName[2568:2619]: 63,
	_CUdevice_attributeName[2619:2671]: 64,
	_CUdevice_attributeName[2671:2723]: 65,
	_CUdevice_attributeName[2723:2771]: 66,
	_CUdevice_attributeName[2771:2827]: 67,
	_CUdevice_attributeName[2827:2884]: 68,
	_CUdevice_attributeName[2884:2934]: 69,
	_CUdevice_attributeName[2934:2984]: 70,
	_CUdevice_attributeName[2984:3035]: 71,
	_CUdevice_attributeName[3035:3085]: 72,
	_CUdevice_attributeName[3085:3138]: 73,
	_CUdevice_attributeName[3138:3192]: 74,
	_CUdevice_attributeName[3192:3236]: 75,
	_CUdevice_attributeName[3236:3280]: 76,
	_CUdevice_attributeName[3280:3333]: 77,
	_CUdevice_attributeName[3333:3380]: 78,
	_CUdevice_attributeName[3380:3425]: 79,
	_CUdevice_attributeName[3425:3469]: 80,
	_CUdevice_attributeName[3469:3525]: 81,
	_CUdevice_attributeName[3525:3577]: 82,
	_CUdevice_attributeName[3577:3611]: 83,
	_CUdevice_attributeName[3611:3646]: 84,
	_CUdevice_attributeName[3646:3690]: 85,
	_CUdevice_attributeName[3690:3738]: 86,
	_CUdevice_attributeName[3738:3795]: 87,
	_CUdevice_attributeName[3795:3837]: 88,
	_CUdevice_attributeName[3837:3882]: 89,
	_CUdevice_attributeName[3882:3930]: 90,
	_CUdevice_attributeName[3930:3989]: 91,
	_CUdevice_attributeName[3989:4031]: 92,
	_CUdevice_attributeName[4031:4080]: 93,
	_CUdevice_attributeName[4080:4129]: 94,
	_CUdevice_attributeName[4129:4167]: 95,
	_CUdevice_attributeName[4167:4218]: 96,
	_CUdevice_attributeName[4218:4271]: 97,
}

// CUdevice_attributeString retrieves an enum value from the enum constants string name.
// Throws an error if the param is not part of the enum.
func CUdevice_attributeString(s string) (CUdevice_attribute, error) {
	if val, ok := _CUdevice_attributeNameToValueMap[s]; ok {
		return val, nil
	}
	return 0, fmt.Errorf("%s does not belong to CUdevice_attribute values", s)
}

// CUdevice_attributeValues returns all values of the enum
func CUdevice_attributeValues() []CUdevice_attribute {
	return _CUdevice_attributeValues
}

// IsACUdevice_attribute returns "true" if the value is listed in the enum definition. "false" otherwise
func (i CUdevice_attribute) IsACUdevice_attribute() bool {
	for _, v := range _CUdevice_attributeValues {
		if i == v {
			return true
		}
	}
	return false
}

// MarshalJSON implements the json.Marshaler interface for CUdevice_attribute
func (i CUdevice_attribute) MarshalJSON() ([]byte, error) {
	return json.Marshal(i.String())
}

// UnmarshalJSON implements the json.Unmarshaler interface for CUdevice_attribute
func (i *CUdevice_attribute) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err != nil {
		return fmt.Errorf("CUdevice_attribute should be a string, got %s", data)
	}

	var err error
	*i, err = CUdevice_attributeString(s)
	return err
}

// MarshalText implements the encoding.TextMarshaler interface for CUdevice_attribute
func (i CUdevice_attribute) MarshalText() ([]byte, error) {
	return []byte(i.String()), nil
}

// UnmarshalText implements the encoding.TextUnmarshaler interface for CUdevice_attribute
func (i *CUdevice_attribute) UnmarshalText(text []byte) error {
	var err error
	*i, err = CUdevice_attributeString(string(text))
	return err
}

// MarshalYAML implements a YAML Marshaler for CUdevice_attribute
func (i CUdevice_attribute) MarshalYAML() (interface{}, error) {
	return i.String(), nil
}

// UnmarshalYAML implements a YAML Unmarshaler for CUdevice_attribute
func (i *CUdevice_attribute) UnmarshalYAML(unmarshal func(interface{}) error) error {
	var s string
	if err := unmarshal(&s); err != nil {
		return err
	}

	var err error
	*i, err = CUdevice_attributeString(s)
	return err
}

func (i CUdevice_attribute) Value() (driver.Value, error) {
	return i.String(), nil
}

func (i *CUdevice_attribute) Scan(value interface{}) error {
	if value == nil {
		return nil
	}

	str, ok := value.(string)
	if !ok {
		bytes, ok := value.([]byte)
		if !ok {
			return fmt.Errorf("value is not a byte slice")
		}

		str = string(bytes[:])
	}

	val, err := CUdevice_attributeString(str)
	if err != nil {
		return err
	}

	*i = val
	return nil
}