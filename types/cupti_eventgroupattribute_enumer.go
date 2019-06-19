// Code generated by "enumer -type=CUpti_EventGroupAttribute -json -text -yaml -sql"; DO NOT EDIT.

//
package types

import (
	"database/sql/driver"
	"encoding/json"
	"fmt"
)

const (
	_CUpti_EventGroupAttributeName_0 = "CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_IDCUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCESCUPTI_EVENT_GROUP_ATTR_USER_DATACUPTI_EVENT_GROUP_ATTR_NUM_EVENTSCUPTI_EVENT_GROUP_ATTR_EVENTSCUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT"
	_CUpti_EventGroupAttributeName_1 = "CUPTI_EVENT_GROUP_ATTR_FORCE_INT"
)

var (
	_CUpti_EventGroupAttributeIndex_0 = [...]uint8{0, 38, 89, 121, 154, 183, 220}
	_CUpti_EventGroupAttributeIndex_1 = [...]uint8{0, 32}
)

func (i CUpti_EventGroupAttribute) String() string {
	switch {
	case 0 <= i && i <= 5:
		return _CUpti_EventGroupAttributeName_0[_CUpti_EventGroupAttributeIndex_0[i]:_CUpti_EventGroupAttributeIndex_0[i+1]]
	case i == 2147483647:
		return _CUpti_EventGroupAttributeName_1
	default:
		return fmt.Sprintf("CUpti_EventGroupAttribute(%d)", i)
	}
}

var _CUpti_EventGroupAttributeValues = []CUpti_EventGroupAttribute{0, 1, 2, 3, 4, 5, 2147483647}

var _CUpti_EventGroupAttributeNameToValueMap = map[string]CUpti_EventGroupAttribute{
	_CUpti_EventGroupAttributeName_0[0:38]:    0,
	_CUpti_EventGroupAttributeName_0[38:89]:   1,
	_CUpti_EventGroupAttributeName_0[89:121]:  2,
	_CUpti_EventGroupAttributeName_0[121:154]: 3,
	_CUpti_EventGroupAttributeName_0[154:183]: 4,
	_CUpti_EventGroupAttributeName_0[183:220]: 5,
	_CUpti_EventGroupAttributeName_1[0:32]:    2147483647,
}

// CUpti_EventGroupAttributeString retrieves an enum value from the enum constants string name.
// Throws an error if the param is not part of the enum.
func CUpti_EventGroupAttributeString(s string) (CUpti_EventGroupAttribute, error) {
	if val, ok := _CUpti_EventGroupAttributeNameToValueMap[s]; ok {
		return val, nil
	}
	return 0, fmt.Errorf("%s does not belong to CUpti_EventGroupAttribute values", s)
}

// CUpti_EventGroupAttributeValues returns all values of the enum
func CUpti_EventGroupAttributeValues() []CUpti_EventGroupAttribute {
	return _CUpti_EventGroupAttributeValues
}

// IsACUpti_EventGroupAttribute returns "true" if the value is listed in the enum definition. "false" otherwise
func (i CUpti_EventGroupAttribute) IsACUpti_EventGroupAttribute() bool {
	for _, v := range _CUpti_EventGroupAttributeValues {
		if i == v {
			return true
		}
	}
	return false
}

// MarshalJSON implements the json.Marshaler interface for CUpti_EventGroupAttribute
func (i CUpti_EventGroupAttribute) MarshalJSON() ([]byte, error) {
	return json.Marshal(i.String())
}

// UnmarshalJSON implements the json.Unmarshaler interface for CUpti_EventGroupAttribute
func (i *CUpti_EventGroupAttribute) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err != nil {
		return fmt.Errorf("CUpti_EventGroupAttribute should be a string, got %s", data)
	}

	var err error
	*i, err = CUpti_EventGroupAttributeString(s)
	return err
}

// MarshalText implements the encoding.TextMarshaler interface for CUpti_EventGroupAttribute
func (i CUpti_EventGroupAttribute) MarshalText() ([]byte, error) {
	return []byte(i.String()), nil
}

// UnmarshalText implements the encoding.TextUnmarshaler interface for CUpti_EventGroupAttribute
func (i *CUpti_EventGroupAttribute) UnmarshalText(text []byte) error {
	var err error
	*i, err = CUpti_EventGroupAttributeString(string(text))
	return err
}

// MarshalYAML implements a YAML Marshaler for CUpti_EventGroupAttribute
func (i CUpti_EventGroupAttribute) MarshalYAML() (interface{}, error) {
	return i.String(), nil
}

// UnmarshalYAML implements a YAML Unmarshaler for CUpti_EventGroupAttribute
func (i *CUpti_EventGroupAttribute) UnmarshalYAML(unmarshal func(interface{}) error) error {
	var s string
	if err := unmarshal(&s); err != nil {
		return err
	}

	var err error
	*i, err = CUpti_EventGroupAttributeString(s)
	return err
}

func (i CUpti_EventGroupAttribute) Value() (driver.Value, error) {
	return i.String(), nil
}

func (i *CUpti_EventGroupAttribute) Scan(value interface{}) error {
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

	val, err := CUpti_EventGroupAttributeString(str)
	if err != nil {
		return err
	}

	*i = val
	return nil
}