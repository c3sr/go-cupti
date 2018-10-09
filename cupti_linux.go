// +build linux,cgo,!arm64

package cupti

// #include <cupti.h>
import "C"
import (
	"sync"
	"time"

	context "context"

	"github.com/pkg/errors"
	"github.com/rai-project/go-cupti/types"
	nvidiasmi "github.com/rai-project/nvidia-smi"

	_ "github.com/rai-project/tracer/jaeger"
	_ "github.com/rai-project/tracer/noop"
	_ "github.com/rai-project/tracer/zipkin"
)

type CUPTI struct {
	*Options
	sync.Mutex
	cuCtxs          []C.CUcontext
	subscriber      C.CUpti_SubscriberHandle
	deviceResetTime time.Time
	startTimeStamp  uint64
	beginTime       time.Time
}

var (
	currentCUPTI *CUPTI
)

func New(opts ...Option) (*CUPTI, error) {
	nvidiasmi.Wait()
	if !nvidiasmi.HasGPU {
		return nil, errors.New("no gpu found while trying to initialize cupti")
	}

	options := NewOptions(opts...)
	c := &CUPTI{
		Options: options,
	}

	// if err := c.init(); err != nil {
	// 	panic(err)
	// }

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

	return c, nil
}

func (c *CUPTI) SetContext(ctx context.Context) {
	c.Lock()
	defer c.Unlock()
	c.ctx = ctx
}

func init() {
	if err := checkCUResult(C.cuInit(0)); err != nil {
		log.WithError(err).Error("failed to perform cuInit")
	}
}

var cuInitOnce sync.Once

func (c *CUPTI) init() error {

	var devCount C.int
	C.cudaGetDeviceCount(&devCount)
	c.cuCtxs = make([]C.CUcontext, len(nvidiasmi.Info.GPUS))
	for ii, gpu := range nvidiasmi.Info.GPUS {
		var cuCtx C.CUcontext

		if err := checkCUResult(C.cuCtxCreate(&cuCtx, 0, C.CUdevice(ii))); err != nil {
			log.WithError(err).
				WithField("device_index", ii).
				WithField("device_id", gpu.ID).
				Error("failed to create cuda context")
			return err
		}
		c.cuCtxs[ii] = cuCtx
	}

	if _, err := c.DeviceReset(); err != nil {
		return err
	}

	return nil
}

func (c *CUPTI) Subscribe() error {
	if err := c.cuptiSubscribe(); err != nil {
		return err
	}
	if err := c.startActivies(); err != nil {
		return err
	}

	if err := c.registerCallbacks(); err != nil {
		return err
	}

	for ii, cuCtx := range c.cuCtxs {
		var samplingConfig C.CUpti_ActivityPCSamplingConfig
		samplingConfig.samplingPeriod = C.CUpti_ActivityPCSamplingPeriod(c.samplingPeriod)
		if err := cuptiActivityConfigurePCSampling(cuCtx, samplingConfig); err != nil {
			log.WithError(err).WithField("device_id", ii).Error("failed to set cupti sampling period")
			return err
		}
	}

	return nil
}

func (c *CUPTI) Unsubscribe() error {
	c.Wait()

	// if err := c.stopActivies(); err != nil {
	// 	log.WithError(err).Error("failed to stop activities")
	// }

	if c.subscriber != nil {
		C.cuptiUnsubscribe(c.subscriber)
	}

	return nil
}

func (c *CUPTI) Close() error {
	currentCUPTI = nil
	if c == nil {
		return nil
	}

	c.Unsubscribe()

	return nil
}

func (c *CUPTI) startActivies() error {
	for _, activityName := range c.activities {
		activity, err := types.CUpti_ActivityKindString(activityName)
		if err != nil {
			return errors.Wrap(err, "unable to map activityName to activity kind")
		}
		err = cuptiActivityEnable(activity)
		if err != nil && err != (*Error)(nil) {
			log.WithError(err).
				WithField("activity", activityName).
				WithField("activity_enum", int(activity)).
				Error("unable to enable activity")
			panic(err)
			return errors.Wrap(err, "unable to enable activities")
		}
	}

	// err := cuptiActivityRegisterCallbacks()
	// if err != nil {
	// 	return errors.Wrap(err, "unable to register activity callbacks")
	// }

	return nil
}

func (c *CUPTI) stopActivies() error {
	for _, activityName := range c.activities {
		activity, err := types.CUpti_ActivityKindString(activityName)
		if err != nil {
			return errors.Wrap(err, "unable to stop activities")
		}
		err = cuptiActivityDisable(activity)
		if err != nil {
			return errors.Wrap(err, "unable to disable activities")
		}
	}
	return nil
}

func (c *CUPTI) registerCallbacks() error {
	for _, callback := range c.callbacks {
		err := c.addCallback(callback)
		if err != nil {
			return err
		}
	}
	return nil
}
