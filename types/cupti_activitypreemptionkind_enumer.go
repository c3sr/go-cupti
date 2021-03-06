// Code generated by "enumer -type=CUpti_ActivityPreemptionKind -json -text -yaml -sql"; DO NOT EDIT.

//
package types

import (
	"database/sql/driver"
	"encoding/json"
	"fmt"
)

const (
	_CUpti_ActivityPreemptionKindName_0 = "CUPTI_ACTIVITY_PREEMPTION_KIND_UNKNOWNCUPTI_ACTIVITY_PREEMPTION_KIND_SAVECUPTI_ACTIVITY_PREEMPTION_KIND_RESTORE"
	_CUpti_ActivityPreemptionKindName_1 = "CUPTI_ACTIVITY_PREEMPTION_KIND_FORCE_INT"
)

var (
	_CUpti_ActivityPreemptionKindIndex_0 = [...]uint8{0, 38, 73, 111}
	_CUpti_ActivityPreemptionKindIndex_1 = [...]uint8{0, 40}
)

func (i CUpti_ActivityPreemptionKind) String() string {
	switch {
	case 0 <= i && i <= 2:
		return _CUpti_ActivityPreemptionKindName_0[_CUpti_ActivityPreemptionKindIndex_0[i]:_CUpti_ActivityPreemptionKindIndex_0[i+1]]
	case i == 2147483647:
		return _CUpti_ActivityPreemptionKindName_1
	default:
		return fmt.Sprintf("CUpti_ActivityPreemptionKind(%d)", i)
	}
}

var _CUpti_ActivityPreemptionKindValues = []CUpti_ActivityPreemptionKind{0, 1, 2, 2147483647}

var _CUpti_ActivityPreemptionKindNameToValueMap = map[string]CUpti_ActivityPreemptionKind{
	_CUpti_ActivityPreemptionKindName_0[0:38]:   0,
	_CUpti_ActivityPreemptionKindName_0[38:73]:  1,
	_CUpti_ActivityPreemptionKindName_0[73:111]: 2,
	_CUpti_ActivityPreemptionKindName_1[0:40]:   2147483647,
}

// CUpti_ActivityPreemptionKindString retrieves an enum value from the enum constants string name.
// Throws an error if the param is not part of the enum.
func CUpti_ActivityPreemptionKindString(s string) (CUpti_ActivityPreemptionKind, error) {
	if val, ok := _CUpti_ActivityPreemptionKindNameToValueMap[s]; ok {
		return val, nil
	}
	return 0, fmt.Errorf("%s does not belong to CUpti_ActivityPreemptionKind values", s)
}

// CUpti_ActivityPreemptionKindValues returns all values of the enum
func CUpti_ActivityPreemptionKindValues() []CUpti_ActivityPreemptionKind {
	return _CUpti_ActivityPreemptionKindValues
}

// IsACUpti_ActivityPreemptionKind returns "true" if the value is listed in the enum definition. "false" otherwise
func (i CUpti_ActivityPreemptionKind) IsACUpti_ActivityPreemptionKind() bool {
	for _, v := range _CUpti_ActivityPreemptionKindValues {
		if i == v {
			return true
		}
	}
	return false
}

// MarshalJSON implements the json.Marshaler interface for CUpti_ActivityPreemptionKind
func (i CUpti_ActivityPreemptionKind) MarshalJSON() ([]byte, error) {
	return json.Marshal(i.String())
}

// UnmarshalJSON implements the json.Unmarshaler interface for CUpti_ActivityPreemptionKind
func (i *CUpti_ActivityPreemptionKind) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err != nil {
		return fmt.Errorf("CUpti_ActivityPreemptionKind should be a string, got %s", data)
	}

	var err error
	*i, err = CUpti_ActivityPreemptionKindString(s)
	return err
}

// MarshalText implements the encoding.TextMarshaler interface for CUpti_ActivityPreemptionKind
func (i CUpti_ActivityPreemptionKind) MarshalText() ([]byte, error) {
	return []byte(i.String()), nil
}

// UnmarshalText implements the encoding.TextUnmarshaler interface for CUpti_ActivityPreemptionKind
func (i *CUpti_ActivityPreemptionKind) UnmarshalText(text []byte) error {
	var err error
	*i, err = CUpti_ActivityPreemptionKindString(string(text))
	return err
}

// MarshalYAML implements a YAML Marshaler for CUpti_ActivityPreemptionKind
func (i CUpti_ActivityPreemptionKind) MarshalYAML() (interface{}, error) {
	return i.String(), nil
}

// UnmarshalYAML implements a YAML Unmarshaler for CUpti_ActivityPreemptionKind
func (i *CUpti_ActivityPreemptionKind) UnmarshalYAML(unmarshal func(interface{}) error) error {
	var s string
	if err := unmarshal(&s); err != nil {
		return err
	}

	var err error
	*i, err = CUpti_ActivityPreemptionKindString(s)
	return err
}

func (i CUpti_ActivityPreemptionKind) Value() (driver.Value, error) {
	return i.String(), nil
}

func (i *CUpti_ActivityPreemptionKind) Scan(value interface{}) error {
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

	val, err := CUpti_ActivityPreemptionKindString(str)
	if err != nil {
		return err
	}

	*i = val
	return nil
}
