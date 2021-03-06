package cuptigrpc

import (
	"context"
	"google.golang.org/grpc"
)

func noopUnaryServer(
	ctx context.Context,
	req interface{},
	info *grpc.UnaryServerInfo,
	handler grpc.UnaryHandler,
) (resp interface{}, err error) {
	return handler(ctx, req)
}
