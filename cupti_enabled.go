// +build linux,cgo,!nogpu

package cupti

// #include <cupti.h>
// #include "csrc/utils.hpp"
// extern void CUPTIAPI kernelReplayCallback( char*  kernelName,  int numReplaysDone, void*  customData );
import "C"
import (
	"context"
	"fmt"
	"sync"
	"time"
	"unsafe"

	"github.com/c3sr/go-cupti/types"
	nvidiasmi "github.com/c3sr/nvidia-smi"
	"github.com/c3sr/tracer"
	opentracing "github.com/opentracing/opentracing-go"
	spanlog "github.com/opentracing/opentracing-go/log"
	"github.com/pkg/errors"
)

type CUPTI struct {
	*Options
	sync.RWMutex
	cuCtxs          []C.CUcontext
	subscriber      C.CUpti_SubscriberHandle
	deviceResetTime time.Time
	startTimeStamp  uint64
	beginTime       time.Time
	spans           sync.Map
	correlationMap  map[uint64]int
}

var (
	currentCUPTI *CUPTI
)

func New(opts ...Option) (*CUPTI, error) {
	nvidiasmi.Wait()
	if !nvidiasmi.HasGPU {
		return nil, errors.New("no gpu found while trying to initialize cupti")
	}

	runInit()

	options := NewOptions(opts...)
	c := &CUPTI{
		Options: options,
	}

	span, _ := tracer.StartSpanFromContext(options.ctx, tracer.FULL_TRACE, "cupti_new")
	defer span.Finish()

	currentCUPTI = c

	if err := c.Subscribe(); err != nil {
		return nil, c.Close()
	}

	startTimeStamp, err := cuptiGetTimestamp()
	if err != nil {
		return nil, err
	}
	c.startTimeStamp = startTimeStamp
	c.beginTime = time.Now()

	if len(c.metrics) == 0 {
		C.startProfiling(nil)
	} else {
		metric := c.metrics[0]
		for i := 1; i < len(c.metrics); i++ {
			metric += ","
			metric += c.metrics[i]
		}
		cMetric := C.CString(metric)
		C.startProfiling(cMetric)
		C.free(unsafe.Pointer(cMetric))
	}

	c.correlationMap = make(map[uint64]int)

	return c, nil
}

func (c *CUPTI) SetContext(ctx context.Context) {
	c.Lock()
	defer c.Unlock()
	c.ctx = ctx
}

func runInit() {
	if err := checkCUResult(C.cuInit(0)); err != nil {
		log.WithError(err).Error("failed to perform cuInit")
	}
}

func (c *CUPTI) Subscribe() error {
	if err := c.startActivies(); err != nil {
		return err
	}

	if err := c.cuptiSubscribe(); err != nil {
		return err
	}

	if err := c.enableCallbacks(); err != nil {
		return err
	}

	if c.samplingPeriod != 0 {
		for ii, cuCtx := range c.cuCtxs {
			var samplingConfig C.CUpti_ActivityPCSamplingConfig
			samplingConfig.samplingPeriod = C.CUpti_ActivityPCSamplingPeriod(c.samplingPeriod)
			if err := cuptiActivityConfigurePCSampling(cuCtx, samplingConfig); err != nil {
				log.WithError(err).WithField("device_id", ii).Error("failed to set cupti sampling period")
				return err
			}
		}
	}

	return nil
}

func (c *CUPTI) Unsubscribe() error {
	c.Wait()
	if err := c.stopActivies(); err != nil {
		return err
	}
	if err := c.cuptiUnsubscribe(); err != nil {
		return err
	}
	return nil
}

func (c *CUPTI) Close() error {
	if c == nil {
		return nil
	}

	var flattenedLength C.uint64_t

	ptr := C.endProfiling(&flattenedLength)
	if ptr != nil {
		metricData := (*[1 << 30]float64)(unsafe.Pointer(ptr))[:uint64(flattenedLength):uint64(flattenedLength)]
		fmt.Println(flattenedLength)
		c.spans.Range(func(key, value interface{}) bool {
			spKey := key.(spanKey)
			if spKey.tag == "cuda_launch" {
				sp := value.(opentracing.Span)
				st := c.correlationMap[spKey.correlationId] * len(c.metrics)
				for i := 0; i < len(c.metrics); i++ {
					sp.LogFields(spanlog.Float64(c.metrics[i], metricData[st+i]))
				}
			}
			return true
		})
	}

	c.Unsubscribe()

	currentCUPTI = nil
	return nil
}

func (c *CUPTI) startActivies() error {
	for _, activityName := range c.activities {
		activity, err := types.CUpti_ActivityKindString(activityName)
		if err != nil {
			return errors.Wrapf(err, "unable to map %v to activity kind", activityName)
		}
		err = cuptiActivityEnable(activity)
		if err != nil {
			log.WithError(err).
				WithField("activity", activityName).
				WithField("activity_enum", int(activity)).
				Error("unable to enable activity")
			panic(err)
			return errors.Wrapf(err, "unable to enable activitiy %v", activityName)
		}
	}

	err := cuptiActivityRegisterCallbacks()
	if err != nil {
		return errors.Wrap(err, "unable to register activity callbacks")
	}

	return nil
}

func (c *CUPTI) stopActivies() error {
	for _, activityName := range c.activities {
		activity, err := types.CUpti_ActivityKindString(activityName)
		if err != nil {
			return errors.Wrapf(err, "unable to map %v to activity kind", activityName)
		}
		err = cuptiActivityDisable(activity)
		if err != nil {
			return errors.Wrapf(err, "unable to disable activity %v", activityName)
		}
	}
	return nil
}

func (c *CUPTI) enableCallbacks() error {
	for _, callback := range c.callbacks {
		err := c.enableCallback(callback)
		if err != nil {
			return err
		}
	}
	return nil
}
