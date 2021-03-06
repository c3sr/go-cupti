// Code generated by "enumer -type=CUpti_ActivityUnifiedMemoryAccessType -json -text -yaml -sql"; DO NOT EDIT.

//
package types

import (
	"database/sql/driver"
	"encoding/json"
	"fmt"
)

const _CUpti_ActivityUnifiedMemoryAccessTypeName = "CUPTI_ACTIVITY_UNIFIED_MEMORY_ACCESS_TYPE_UNKNOWNCUPTI_ACTIVITY_UNIFIED_MEMORY_ACCESS_TYPE_READCUPTI_ACTIVITY_UNIFIED_MEMORY_ACCESS_TYPE_WRITECUPTI_ACTIVITY_UNIFIED_MEMORY_ACCESS_TYPE_ATOMICCUPTI_ACTIVITY_UNIFIED_MEMORY_ACCESS_TYPE_PREFETCH"

var _CUpti_ActivityUnifiedMemoryAccessTypeIndex = [...]uint8{0, 49, 95, 142, 190, 240}

func (i CUpti_ActivityUnifiedMemoryAccessType) String() string {
	if i < 0 || i >= CUpti_ActivityUnifiedMemoryAccessType(len(_CUpti_ActivityUnifiedMemoryAccessTypeIndex)-1) {
		return fmt.Sprintf("CUpti_ActivityUnifiedMemoryAccessType(%d)", i)
	}
	return _CUpti_ActivityUnifiedMemoryAccessTypeName[_CUpti_ActivityUnifiedMemoryAccessTypeIndex[i]:_CUpti_ActivityUnifiedMemoryAccessTypeIndex[i+1]]
}

var _CUpti_ActivityUnifiedMemoryAccessTypeValues = []CUpti_ActivityUnifiedMemoryAccessType{0, 1, 2, 3, 4}

var _CUpti_ActivityUnifiedMemoryAccessTypeNameToValueMap = map[string]CUpti_ActivityUnifiedMemoryAccessType{
	_CUpti_ActivityUnifiedMemoryAccessTypeName[0:49]:    0,
	_CUpti_ActivityUnifiedMemoryAccessTypeName[49:95]:   1,
	_CUpti_ActivityUnifiedMemoryAccessTypeName[95:142]:  2,
	_CUpti_ActivityUnifiedMemoryAccessTypeName[142:190]: 3,
	_CUpti_ActivityUnifiedMemoryAccessTypeName[190:240]: 4,
}

// CUpti_ActivityUnifiedMemoryAccessTypeString retrieves an enum value from the enum constants string name.
// Throws an error if the param is not part of the enum.
func CUpti_ActivityUnifiedMemoryAccessTypeString(s string) (CUpti_ActivityUnifiedMemoryAccessType, error) {
	if val, ok := _CUpti_ActivityUnifiedMemoryAccessTypeNameToValueMap[s]; ok {
		return val, nil
	}
	return 0, fmt.Errorf("%s does not belong to CUpti_ActivityUnifiedMemoryAccessType values", s)
}

// CUpti_ActivityUnifiedMemoryAccessTypeValues returns all values of the enum
func CUpti_ActivityUnifiedMemoryAccessTypeValues() []CUpti_ActivityUnifiedMemoryAccessType {
	return _CUpti_ActivityUnifiedMemoryAccessTypeValues
}

// IsACUpti_ActivityUnifiedMemoryAccessType returns "true" if the value is listed in the enum definition. "false" otherwise
func (i CUpti_ActivityUnifiedMemoryAccessType) IsACUpti_ActivityUnifiedMemoryAccessType() bool {
	for _, v := range _CUpti_ActivityUnifiedMemoryAccessTypeValues {
		if i == v {
			return true
		}
	}
	return false
}

// MarshalJSON implements the json.Marshaler interface for CUpti_ActivityUnifiedMemoryAccessType
func (i CUpti_ActivityUnifiedMemoryAccessType) MarshalJSON() ([]byte, error) {
	return json.Marshal(i.String())
}

// UnmarshalJSON implements the json.Unmarshaler interface for CUpti_ActivityUnifiedMemoryAccessType
func (i *CUpti_ActivityUnifiedMemoryAccessType) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err != nil {
		return fmt.Errorf("CUpti_ActivityUnifiedMemoryAccessType should be a string, got %s", data)
	}

	var err error
	*i, err = CUpti_ActivityUnifiedMemoryAccessTypeString(s)
	return err
}

// MarshalText implements the encoding.TextMarshaler interface for CUpti_ActivityUnifiedMemoryAccessType
func (i CUpti_ActivityUnifiedMemoryAccessType) MarshalText() ([]byte, error) {
	return []byte(i.String()), nil
}

// UnmarshalText implements the encoding.TextUnmarshaler interface for CUpti_ActivityUnifiedMemoryAccessType
func (i *CUpti_ActivityUnifiedMemoryAccessType) UnmarshalText(text []byte) error {
	var err error
	*i, err = CUpti_ActivityUnifiedMemoryAccessTypeString(string(text))
	return err
}

// MarshalYAML implements a YAML Marshaler for CUpti_ActivityUnifiedMemoryAccessType
func (i CUpti_ActivityUnifiedMemoryAccessType) MarshalYAML() (interface{}, error) {
	return i.String(), nil
}

// UnmarshalYAML implements a YAML Unmarshaler for CUpti_ActivityUnifiedMemoryAccessType
func (i *CUpti_ActivityUnifiedMemoryAccessType) UnmarshalYAML(unmarshal func(interface{}) error) error {
	var s string
	if err := unmarshal(&s); err != nil {
		return err
	}

	var err error
	*i, err = CUpti_ActivityUnifiedMemoryAccessTypeString(s)
	return err
}

func (i CUpti_ActivityUnifiedMemoryAccessType) Value() (driver.Value, error) {
	return i.String(), nil
}

func (i *CUpti_ActivityUnifiedMemoryAccessType) Scan(value interface{}) error {
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

	val, err := CUpti_ActivityUnifiedMemoryAccessTypeString(str)
	if err != nil {
		return err
	}

	*i = val
	return nil
}
