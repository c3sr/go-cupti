// Code generated by "enumer -type=CUpti_MetricCategory -json -text -yaml -sql"; DO NOT EDIT.

//
package types

import (
	"database/sql/driver"
	"encoding/json"
	"fmt"
)

const (
	_CUpti_MetricCategoryName_0 = "CUPTI_METRIC_CATEGORY_MEMORYCUPTI_METRIC_CATEGORY_INSTRUCTIONCUPTI_METRIC_CATEGORY_MULTIPROCESSORCUPTI_METRIC_CATEGORY_CACHECUPTI_METRIC_CATEGORY_TEXTURECUPTI_METRIC_CATEGORY_NVLINK"
	_CUpti_MetricCategoryName_1 = "CUPTI_METRIC_CATEGORY_FORCE_INT"
)

var (
	_CUpti_MetricCategoryIndex_0 = [...]uint8{0, 28, 61, 97, 124, 153, 181}
	_CUpti_MetricCategoryIndex_1 = [...]uint8{0, 31}
)

func (i CUpti_MetricCategory) String() string {
	switch {
	case 0 <= i && i <= 5:
		return _CUpti_MetricCategoryName_0[_CUpti_MetricCategoryIndex_0[i]:_CUpti_MetricCategoryIndex_0[i+1]]
	case i == 2147483647:
		return _CUpti_MetricCategoryName_1
	default:
		return fmt.Sprintf("CUpti_MetricCategory(%d)", i)
	}
}

var _CUpti_MetricCategoryValues = []CUpti_MetricCategory{0, 1, 2, 3, 4, 5, 2147483647}

var _CUpti_MetricCategoryNameToValueMap = map[string]CUpti_MetricCategory{
	_CUpti_MetricCategoryName_0[0:28]:    0,
	_CUpti_MetricCategoryName_0[28:61]:   1,
	_CUpti_MetricCategoryName_0[61:97]:   2,
	_CUpti_MetricCategoryName_0[97:124]:  3,
	_CUpti_MetricCategoryName_0[124:153]: 4,
	_CUpti_MetricCategoryName_0[153:181]: 5,
	_CUpti_MetricCategoryName_1[0:31]:    2147483647,
}

// CUpti_MetricCategoryString retrieves an enum value from the enum constants string name.
// Throws an error if the param is not part of the enum.
func CUpti_MetricCategoryString(s string) (CUpti_MetricCategory, error) {
	if val, ok := _CUpti_MetricCategoryNameToValueMap[s]; ok {
		return val, nil
	}
	return 0, fmt.Errorf("%s does not belong to CUpti_MetricCategory values", s)
}

// CUpti_MetricCategoryValues returns all values of the enum
func CUpti_MetricCategoryValues() []CUpti_MetricCategory {
	return _CUpti_MetricCategoryValues
}

// IsACUpti_MetricCategory returns "true" if the value is listed in the enum definition. "false" otherwise
func (i CUpti_MetricCategory) IsACUpti_MetricCategory() bool {
	for _, v := range _CUpti_MetricCategoryValues {
		if i == v {
			return true
		}
	}
	return false
}

// MarshalJSON implements the json.Marshaler interface for CUpti_MetricCategory
func (i CUpti_MetricCategory) MarshalJSON() ([]byte, error) {
	return json.Marshal(i.String())
}

// UnmarshalJSON implements the json.Unmarshaler interface for CUpti_MetricCategory
func (i *CUpti_MetricCategory) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err != nil {
		return fmt.Errorf("CUpti_MetricCategory should be a string, got %s", data)
	}

	var err error
	*i, err = CUpti_MetricCategoryString(s)
	return err
}

// MarshalText implements the encoding.TextMarshaler interface for CUpti_MetricCategory
func (i CUpti_MetricCategory) MarshalText() ([]byte, error) {
	return []byte(i.String()), nil
}

// UnmarshalText implements the encoding.TextUnmarshaler interface for CUpti_MetricCategory
func (i *CUpti_MetricCategory) UnmarshalText(text []byte) error {
	var err error
	*i, err = CUpti_MetricCategoryString(string(text))
	return err
}

// MarshalYAML implements a YAML Marshaler for CUpti_MetricCategory
func (i CUpti_MetricCategory) MarshalYAML() (interface{}, error) {
	return i.String(), nil
}

// UnmarshalYAML implements a YAML Unmarshaler for CUpti_MetricCategory
func (i *CUpti_MetricCategory) UnmarshalYAML(unmarshal func(interface{}) error) error {
	var s string
	if err := unmarshal(&s); err != nil {
		return err
	}

	var err error
	*i, err = CUpti_MetricCategoryString(s)
	return err
}

func (i CUpti_MetricCategory) Value() (driver.Value, error) {
	return i.String(), nil
}

func (i *CUpti_MetricCategory) Scan(value interface{}) error {
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

	val, err := CUpti_MetricCategoryString(str)
	if err != nil {
		return err
	}

	*i = val
	return nil
}