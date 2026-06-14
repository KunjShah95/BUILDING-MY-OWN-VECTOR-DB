package vectordb

import "fmt"

// VectorDBError is the base error type for all Vector DB client errors.
type VectorDBError struct {
	msg string
}

func (e *VectorDBError) Error() string { return e.msg }

// VectorDBHTTPError is returned when the API responds with a non-2xx status.
type VectorDBHTTPError struct {
	VectorDBError
	StatusCode int
	Detail     interface{}
}

func NewVectorDBHTTPError(statusCode int, detail interface{}) *VectorDBHTTPError {
	return &VectorDBHTTPError{
		VectorDBError: VectorDBError{
			msg: fmt.Sprintf("HTTP %d: %v", statusCode, detail),
		},
		StatusCode: statusCode,
		Detail:     detail,
	}
}
