// Code generated by "enumer -type=CUpti_CallbackIdResource -json -text -yaml -sql"; DO NOT EDIT.

//
package types

import (
	"database/sql/driver"
	"encoding/json"
	"fmt"
)

const (
	_CUpti_CallbackIdResourceName_0 = "CUPTI_CBID_RESOURCE_INVALIDCUPTI_CBID_RESOURCE_CONTEXT_CREATEDCUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTINGCUPTI_CBID_RESOURCE_STREAM_CREATEDCUPTI_CBID_RESOURCE_STREAM_DESTROY_STARTINGCUPTI_CBID_RESOURCE_CU_INIT_FINISHEDCUPTI_CBID_RESOURCE_MODULE_LOADEDCUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTINGCUPTI_CBID_RESOURCE_MODULE_PROFILEDCUPTI_CBID_RESOURCE_SIZE"
	_CUpti_CallbackIdResourceName_1 = "CUPTI_CBID_RESOURCE_FORCE_INT"
)

var (
	_CUpti_CallbackIdResourceIndex_0 = [...]uint16{0, 27, 62, 106, 140, 183, 219, 252, 294, 329, 353}
	_CUpti_CallbackIdResourceIndex_1 = [...]uint8{0, 29}
)

func (i CUpti_CallbackIdResource) String() string {
	switch {
	case 0 <= i && i <= 9:
		return _CUpti_CallbackIdResourceName_0[_CUpti_CallbackIdResourceIndex_0[i]:_CUpti_CallbackIdResourceIndex_0[i+1]]
	case i == 2147483647:
		return _CUpti_CallbackIdResourceName_1
	default:
		return fmt.Sprintf("CUpti_CallbackIdResource(%d)", i)
	}
}

var _CUpti_CallbackIdResourceValues = []CUpti_CallbackIdResource{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 2147483647}

var _CUpti_CallbackIdResourceNameToValueMap = map[string]CUpti_CallbackIdResource{
	_CUpti_CallbackIdResourceName_0[0:27]:    0,
	_CUpti_CallbackIdResourceName_0[27:62]:   1,
	_CUpti_CallbackIdResourceName_0[62:106]:  2,
	_CUpti_CallbackIdResourceName_0[106:140]: 3,
	_CUpti_CallbackIdResourceName_0[140:183]: 4,
	_CUpti_CallbackIdResourceName_0[183:219]: 5,
	_CUpti_CallbackIdResourceName_0[219:252]: 6,
	_CUpti_CallbackIdResourceName_0[252:294]: 7,
	_CUpti_CallbackIdResourceName_0[294:329]: 8,
	_CUpti_CallbackIdResourceName_0[329:353]: 9,
	_CUpti_CallbackIdResourceName_1[0:29]:    2147483647,
}

// CUpti_CallbackIdResourceString retrieves an enum value from the enum constants string name.
// Throws an error if the param is not part of the enum.
func CUpti_CallbackIdResourceString(s string) (CUpti_CallbackIdResource, error) {
	if val, ok := _CUpti_CallbackIdResourceNameToValueMap[s]; ok {
		return val, nil
	}
	return 0, fmt.Errorf("%s does not belong to CUpti_CallbackIdResource values", s)
}

// CUpti_CallbackIdResourceValues returns all values of the enum
func CUpti_CallbackIdResourceValues() []CUpti_CallbackIdResource {
	return _CUpti_CallbackIdResourceValues
}

// IsACUpti_CallbackIdResource returns "true" if the value is listed in the enum definition. "false" otherwise
func (i CUpti_CallbackIdResource) IsACUpti_CallbackIdResource() bool {
	for _, v := range _CUpti_CallbackIdResourceValues {
		if i == v {
			return true
		}
	}
	return false
}

// MarshalJSON implements the json.Marshaler interface for CUpti_CallbackIdResource
func (i CUpti_CallbackIdResource) MarshalJSON() ([]byte, error) {
	return json.Marshal(i.String())
}

// UnmarshalJSON implements the json.Unmarshaler interface for CUpti_CallbackIdResource
func (i *CUpti_CallbackIdResource) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err != nil {
		return fmt.Errorf("CUpti_CallbackIdResource should be a string, got %s", data)
	}

	var err error
	*i, err = CUpti_CallbackIdResourceString(s)
	return err
}

// MarshalText implements the encoding.TextMarshaler interface for CUpti_CallbackIdResource
func (i CUpti_CallbackIdResource) MarshalText() ([]byte, error) {
	return []byte(i.String()), nil
}

// UnmarshalText implements the encoding.TextUnmarshaler interface for CUpti_CallbackIdResource
func (i *CUpti_CallbackIdResource) UnmarshalText(text []byte) error {
	var err error
	*i, err = CUpti_CallbackIdResourceString(string(text))
	return err
}

// MarshalYAML implements a YAML Marshaler for CUpti_CallbackIdResource
func (i CUpti_CallbackIdResource) MarshalYAML() (interface{}, error) {
	return i.String(), nil
}

// UnmarshalYAML implements a YAML Unmarshaler for CUpti_CallbackIdResource
func (i *CUpti_CallbackIdResource) UnmarshalYAML(unmarshal func(interface{}) error) error {
	var s string
	if err := unmarshal(&s); err != nil {
		return err
	}

	var err error
	*i, err = CUpti_CallbackIdResourceString(s)
	return err
}

func (i CUpti_CallbackIdResource) Value() (driver.Value, error) {
	return i.String(), nil
}

func (i *CUpti_CallbackIdResource) Scan(value interface{}) error {
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

	val, err := CUpti_CallbackIdResourceString(str)
	if err != nil {
		return err
	}

	*i = val
	return nil
}