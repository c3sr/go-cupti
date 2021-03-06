// Code generated by "enumer -type=CUpti_MetricAttribute -json -text -yaml -sql"; DO NOT EDIT.

//
package types

import (
	"database/sql/driver"
	"encoding/json"
	"fmt"
)

const (
	_CUpti_MetricAttributeName_0 = "CUPTI_METRIC_ATTR_NAMECUPTI_METRIC_ATTR_SHORT_DESCRIPTIONCUPTI_METRIC_ATTR_LONG_DESCRIPTIONCUPTI_METRIC_ATTR_CATEGORYCUPTI_METRIC_ATTR_VALUE_KINDCUPTI_METRIC_ATTR_EVALUATION_MODE"
	_CUpti_MetricAttributeName_1 = "CUPTI_METRIC_ATTR_FORCE_INT"
)

var (
	_CUpti_MetricAttributeIndex_0 = [...]uint8{0, 22, 57, 91, 117, 145, 178}
	_CUpti_MetricAttributeIndex_1 = [...]uint8{0, 27}
)

func (i CUpti_MetricAttribute) String() string {
	switch {
	case 0 <= i && i <= 5:
		return _CUpti_MetricAttributeName_0[_CUpti_MetricAttributeIndex_0[i]:_CUpti_MetricAttributeIndex_0[i+1]]
	case i == 2147483647:
		return _CUpti_MetricAttributeName_1
	default:
		return fmt.Sprintf("CUpti_MetricAttribute(%d)", i)
	}
}

var _CUpti_MetricAttributeValues = []CUpti_MetricAttribute{0, 1, 2, 3, 4, 5, 2147483647}

var _CUpti_MetricAttributeNameToValueMap = map[string]CUpti_MetricAttribute{
	_CUpti_MetricAttributeName_0[0:22]:    0,
	_CUpti_MetricAttributeName_0[22:57]:   1,
	_CUpti_MetricAttributeName_0[57:91]:   2,
	_CUpti_MetricAttributeName_0[91:117]:  3,
	_CUpti_MetricAttributeName_0[117:145]: 4,
	_CUpti_MetricAttributeName_0[145:178]: 5,
	_CUpti_MetricAttributeName_1[0:27]:    2147483647,
}

// CUpti_MetricAttributeString retrieves an enum value from the enum constants string name.
// Throws an error if the param is not part of the enum.
func CUpti_MetricAttributeString(s string) (CUpti_MetricAttribute, error) {
	if val, ok := _CUpti_MetricAttributeNameToValueMap[s]; ok {
		return val, nil
	}
	return 0, fmt.Errorf("%s does not belong to CUpti_MetricAttribute values", s)
}

// CUpti_MetricAttributeValues returns all values of the enum
func CUpti_MetricAttributeValues() []CUpti_MetricAttribute {
	return _CUpti_MetricAttributeValues
}

// IsACUpti_MetricAttribute returns "true" if the value is listed in the enum definition. "false" otherwise
func (i CUpti_MetricAttribute) IsACUpti_MetricAttribute() bool {
	for _, v := range _CUpti_MetricAttributeValues {
		if i == v {
			return true
		}
	}
	return false
}

// MarshalJSON implements the json.Marshaler interface for CUpti_MetricAttribute
func (i CUpti_MetricAttribute) MarshalJSON() ([]byte, error) {
	return json.Marshal(i.String())
}

// UnmarshalJSON implements the json.Unmarshaler interface for CUpti_MetricAttribute
func (i *CUpti_MetricAttribute) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err != nil {
		return fmt.Errorf("CUpti_MetricAttribute should be a string, got %s", data)
	}

	var err error
	*i, err = CUpti_MetricAttributeString(s)
	return err
}

// MarshalText implements the encoding.TextMarshaler interface for CUpti_MetricAttribute
func (i CUpti_MetricAttribute) MarshalText() ([]byte, error) {
	return []byte(i.String()), nil
}

// UnmarshalText implements the encoding.TextUnmarshaler interface for CUpti_MetricAttribute
func (i *CUpti_MetricAttribute) UnmarshalText(text []byte) error {
	var err error
	*i, err = CUpti_MetricAttributeString(string(text))
	return err
}

// MarshalYAML implements a YAML Marshaler for CUpti_MetricAttribute
func (i CUpti_MetricAttribute) MarshalYAML() (interface{}, error) {
	return i.String(), nil
}

// UnmarshalYAML implements a YAML Unmarshaler for CUpti_MetricAttribute
func (i *CUpti_MetricAttribute) UnmarshalYAML(unmarshal func(interface{}) error) error {
	var s string
	if err := unmarshal(&s); err != nil {
		return err
	}

	var err error
	*i, err = CUpti_MetricAttributeString(s)
	return err
}

func (i CUpti_MetricAttribute) Value() (driver.Value, error) {
	return i.String(), nil
}

func (i *CUpti_MetricAttribute) Scan(value interface{}) error {
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

	val, err := CUpti_MetricAttributeString(str)
	if err != nil {
		return err
	}

	*i = val
	return nil
}
