// Code generated by "enumer -type=CUpti_ActivityOverheadKind -json -text -yaml -sql"; DO NOT EDIT.

//
package types

import (
	"database/sql/driver"
	"encoding/json"
	"fmt"
)

const (
	_CUpti_ActivityOverheadKindName_0 = "CUPTI_ACTIVITY_OVERHEAD_UNKNOWNCUPTI_ACTIVITY_OVERHEAD_DRIVER_COMPILER"
	_CUpti_ActivityOverheadKindName_1 = "CUPTI_ACTIVITY_OVERHEAD_CUPTI_BUFFER_FLUSH"
	_CUpti_ActivityOverheadKindName_2 = "CUPTI_ACTIVITY_OVERHEAD_CUPTI_INSTRUMENTATION"
	_CUpti_ActivityOverheadKindName_3 = "CUPTI_ACTIVITY_OVERHEAD_CUPTI_RESOURCE"
	_CUpti_ActivityOverheadKindName_4 = "CUPTI_ACTIVITY_OVERHEAD_FORCE_INT"
)

var (
	_CUpti_ActivityOverheadKindIndex_0 = [...]uint8{0, 31, 70}
	_CUpti_ActivityOverheadKindIndex_1 = [...]uint8{0, 42}
	_CUpti_ActivityOverheadKindIndex_2 = [...]uint8{0, 45}
	_CUpti_ActivityOverheadKindIndex_3 = [...]uint8{0, 38}
	_CUpti_ActivityOverheadKindIndex_4 = [...]uint8{0, 33}
)

func (i CUpti_ActivityOverheadKind) String() string {
	switch {
	case 0 <= i && i <= 1:
		return _CUpti_ActivityOverheadKindName_0[_CUpti_ActivityOverheadKindIndex_0[i]:_CUpti_ActivityOverheadKindIndex_0[i+1]]
	case i == 65536:
		return _CUpti_ActivityOverheadKindName_1
	case i == 131072:
		return _CUpti_ActivityOverheadKindName_2
	case i == 196608:
		return _CUpti_ActivityOverheadKindName_3
	case i == 2147483647:
		return _CUpti_ActivityOverheadKindName_4
	default:
		return fmt.Sprintf("CUpti_ActivityOverheadKind(%d)", i)
	}
}

var _CUpti_ActivityOverheadKindValues = []CUpti_ActivityOverheadKind{0, 1, 65536, 131072, 196608, 2147483647}

var _CUpti_ActivityOverheadKindNameToValueMap = map[string]CUpti_ActivityOverheadKind{
	_CUpti_ActivityOverheadKindName_0[0:31]:  0,
	_CUpti_ActivityOverheadKindName_0[31:70]: 1,
	_CUpti_ActivityOverheadKindName_1[0:42]:  65536,
	_CUpti_ActivityOverheadKindName_2[0:45]:  131072,
	_CUpti_ActivityOverheadKindName_3[0:38]:  196608,
	_CUpti_ActivityOverheadKindName_4[0:33]:  2147483647,
}

// CUpti_ActivityOverheadKindString retrieves an enum value from the enum constants string name.
// Throws an error if the param is not part of the enum.
func CUpti_ActivityOverheadKindString(s string) (CUpti_ActivityOverheadKind, error) {
	if val, ok := _CUpti_ActivityOverheadKindNameToValueMap[s]; ok {
		return val, nil
	}
	return 0, fmt.Errorf("%s does not belong to CUpti_ActivityOverheadKind values", s)
}

// CUpti_ActivityOverheadKindValues returns all values of the enum
func CUpti_ActivityOverheadKindValues() []CUpti_ActivityOverheadKind {
	return _CUpti_ActivityOverheadKindValues
}

// IsACUpti_ActivityOverheadKind returns "true" if the value is listed in the enum definition. "false" otherwise
func (i CUpti_ActivityOverheadKind) IsACUpti_ActivityOverheadKind() bool {
	for _, v := range _CUpti_ActivityOverheadKindValues {
		if i == v {
			return true
		}
	}
	return false
}

// MarshalJSON implements the json.Marshaler interface for CUpti_ActivityOverheadKind
func (i CUpti_ActivityOverheadKind) MarshalJSON() ([]byte, error) {
	return json.Marshal(i.String())
}

// UnmarshalJSON implements the json.Unmarshaler interface for CUpti_ActivityOverheadKind
func (i *CUpti_ActivityOverheadKind) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err != nil {
		return fmt.Errorf("CUpti_ActivityOverheadKind should be a string, got %s", data)
	}

	var err error
	*i, err = CUpti_ActivityOverheadKindString(s)
	return err
}

// MarshalText implements the encoding.TextMarshaler interface for CUpti_ActivityOverheadKind
func (i CUpti_ActivityOverheadKind) MarshalText() ([]byte, error) {
	return []byte(i.String()), nil
}

// UnmarshalText implements the encoding.TextUnmarshaler interface for CUpti_ActivityOverheadKind
func (i *CUpti_ActivityOverheadKind) UnmarshalText(text []byte) error {
	var err error
	*i, err = CUpti_ActivityOverheadKindString(string(text))
	return err
}

// MarshalYAML implements a YAML Marshaler for CUpti_ActivityOverheadKind
func (i CUpti_ActivityOverheadKind) MarshalYAML() (interface{}, error) {
	return i.String(), nil
}

// UnmarshalYAML implements a YAML Unmarshaler for CUpti_ActivityOverheadKind
func (i *CUpti_ActivityOverheadKind) UnmarshalYAML(unmarshal func(interface{}) error) error {
	var s string
	if err := unmarshal(&s); err != nil {
		return err
	}

	var err error
	*i, err = CUpti_ActivityOverheadKindString(s)
	return err
}

func (i CUpti_ActivityOverheadKind) Value() (driver.Value, error) {
	return i.String(), nil
}

func (i *CUpti_ActivityOverheadKind) Scan(value interface{}) error {
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

	val, err := CUpti_ActivityOverheadKindString(str)
	if err != nil {
		return err
	}

	*i = val
	return nil
}
