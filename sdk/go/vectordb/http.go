package vectordb

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/url"
	"strings"
)

// requestOptions holds per-request configuration.
type requestOptions struct {
	Headers map[string]string
	Params  map[string]string
}

// RequestOption is a functional option for HTTP requests.
type RequestOption func(*requestOptions)

// WithHeader sets an additional HTTP header for the request.
func WithHeader(key, value string) RequestOption {
	return func(o *requestOptions) {
		if o.Headers == nil {
			o.Headers = make(map[string]string)
		}
		o.Headers[key] = value
	}
}

// WithParam sets a query string parameter for the request.
func WithParam(key, value string) RequestOption {
	return func(o *requestOptions) {
		if o.Params == nil {
			o.Params = make(map[string]string)
		}
		o.Params[key] = value
	}
}

// rawResponse holds the parsed JSON body and the original raw bytes.
type rawResponse struct {
	parsed map[string]interface{}
	raw    []byte
}

// raiseForStatus parses the response body and returns a VectorDBHTTPError
// on non-2xx status codes. It returns both a parsed map and the raw bytes
// so callers can unmarshal directly into typed structs.
func raiseForStatus(res *http.Response) (rawResponse, error) {
	body, err := io.ReadAll(res.Body)
	if err != nil {
		return rawResponse{}, &VectorDBError{msg: fmt.Sprintf("failed to read response body: %v", err)}
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal(body, &parsed); err != nil {
		parsed = map[string]interface{}{"message": string(body)}
	}

	if res.StatusCode < 200 || res.StatusCode >= 300 {
		detail := parsed["detail"]
		if detail == nil {
			detail = parsed
		}
		return rawResponse{}, NewVectorDBHTTPError(res.StatusCode, detail)
	}
	return rawResponse{parsed: parsed, raw: body}, nil
}

// buildURL constructs a full URL with optional query parameters.
func buildURL(baseURL, path string, opts *requestOptions) string {
	u := strings.TrimRight(baseURL, "/") + "/" + strings.TrimLeft(path, "/")
	if opts != nil && len(opts.Params) > 0 {
		vals := url.Values{}
		for k, v := range opts.Params {
			vals.Set(k, v)
		}
		u = u + "?" + vals.Encode()
	}
	return u
}

// jsonGetRaw performs a GET request and returns the full rawResponse
// so callers can unmarshal directly into typed structs.
func jsonGetRaw(ctx context.Context, client *http.Client, baseURL, path string, opts *requestOptions) (rawResponse, error) {
	u := buildURL(baseURL, path, opts)
	req, err := http.NewRequestWithContext(ctx, "GET", u, nil)
	if err != nil {
		return rawResponse{}, &VectorDBError{msg: fmt.Sprintf("failed to create request: %v", err)}
	}
	req.Header.Set("Content-Type", "application/json")
	if opts != nil {
		for k, v := range opts.Headers {
			req.Header.Set(k, v)
		}
	}
	res, err := client.Do(req)
	if err != nil {
		return rawResponse{}, &VectorDBError{msg: fmt.Sprintf("request failed: %v", err)}
	}
	defer res.Body.Close()
	return raiseForStatus(res)
}

// jsonGet performs a GET request and returns the parsed JSON response.
func jsonGet(ctx context.Context, client *http.Client, baseURL, path string, opts *requestOptions) (map[string]interface{}, error) {
	u := buildURL(baseURL, path, opts)
	req, err := http.NewRequestWithContext(ctx, "GET", u, nil)
	if err != nil {
		return nil, &VectorDBError{msg: fmt.Sprintf("failed to create request: %v", err)}
	}
	req.Header.Set("Content-Type", "application/json")
	if opts != nil {
		for k, v := range opts.Headers {
			req.Header.Set(k, v)
		}
	}
	res, err := client.Do(req)
	if err != nil {
		return nil, &VectorDBError{msg: fmt.Sprintf("request failed: %v", err)}
	}
	defer res.Body.Close()
	rr, err := raiseForStatus(res)
	if err != nil {
		return nil, err
	}
	return rr.parsed, nil
}

// jsonPostRaw performs a POST request and returns the full rawResponse
// so callers can unmarshal directly into typed structs.
func jsonPostRaw(ctx context.Context, client *http.Client, baseURL, path string, body interface{}, opts *requestOptions) (rawResponse, error) {
	u := buildURL(baseURL, path, opts)
	var buf bytes.Buffer
	if body != nil {
		if err := json.NewEncoder(&buf).Encode(body); err != nil {
			return rawResponse{}, &VectorDBError{msg: fmt.Sprintf("failed to encode body: %v", err)}
		}
	}
	req, err := http.NewRequestWithContext(ctx, "POST", u, &buf)
	if err != nil {
		return rawResponse{}, &VectorDBError{msg: fmt.Sprintf("failed to create request: %v", err)}
	}
	req.Header.Set("Content-Type", "application/json")
	if opts != nil {
		for k, v := range opts.Headers {
			req.Header.Set(k, v)
		}
	}
	res, err := client.Do(req)
	if err != nil {
		return rawResponse{}, &VectorDBError{msg: fmt.Sprintf("request failed: %v", err)}
	}
	defer res.Body.Close()
	return raiseForStatus(res)
}

// jsonPost performs a POST request and returns the parsed JSON response.
func jsonPost(ctx context.Context, client *http.Client, baseURL, path string, body interface{}, opts *requestOptions) (map[string]interface{}, error) {
	rr, err := jsonPostRaw(ctx, client, baseURL, path, body, opts)
	if err != nil {
		return nil, err
	}
	return rr.parsed, nil
}

// jsonDeleteRaw performs a DELETE request and returns the full rawResponse.
func jsonDeleteRaw(ctx context.Context, client *http.Client, baseURL, path string, opts *requestOptions) (rawResponse, error) {
	u := buildURL(baseURL, path, opts)
	req, err := http.NewRequestWithContext(ctx, "DELETE", u, nil)
	if err != nil {
		return rawResponse{}, &VectorDBError{msg: fmt.Sprintf("failed to create request: %v", err)}
	}
	if opts != nil {
		for k, v := range opts.Headers {
			req.Header.Set(k, v)
		}
	}
	res, err := client.Do(req)
	if err != nil {
		return rawResponse{}, &VectorDBError{msg: fmt.Sprintf("request failed: %v", err)}
	}
	defer res.Body.Close()
	return raiseForStatus(res)
}

// jsonDelete performs a DELETE request and returns the parsed JSON response.
func jsonDelete(ctx context.Context, client *http.Client, baseURL, path string, opts *requestOptions) (map[string]interface{}, error) {
	rr, err := jsonDeleteRaw(ctx, client, baseURL, path, opts)
	if err != nil {
		return nil, err
	}
	return rr.parsed, nil
}

// multipartPostRaw performs a POST request with multipart/form-data and
// returns the full rawResponse for direct unmarshaling.
func multipartPostRaw(ctx context.Context, client *http.Client, baseURL, path string, fields map[string]io.Reader, fileField string, fileContents io.Reader, fileName string, opts *requestOptions) (rawResponse, error) {
	u := buildURL(baseURL, path, opts)
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)

	// Write file field
	if fileContents != nil {
		part, err := w.CreateFormFile(fileField, fileName)
		if err != nil {
			return rawResponse{}, &VectorDBError{msg: fmt.Sprintf("failed to create form file: %v", err)}
		}
		if _, err := io.Copy(part, fileContents); err != nil {
			return rawResponse{}, &VectorDBError{msg: fmt.Sprintf("failed to copy file data: %v", err)}
		}
	}

	// Write extra form fields
	for key, r := range fields {
		var val bytes.Buffer
		if _, err := io.Copy(&val, r); err != nil {
			return rawResponse{}, &VectorDBError{msg: fmt.Sprintf("failed to read field %s: %v", key, err)}
		}
		if err := w.WriteField(key, val.String()); err != nil {
			return rawResponse{}, &VectorDBError{msg: fmt.Sprintf("failed to write field %s: %v", key, err)}
		}
	}

	if err := w.Close(); err != nil {
		return rawResponse{}, &VectorDBError{msg: fmt.Sprintf("failed to close multipart writer: %v", err)}
	}

	req, err := http.NewRequestWithContext(ctx, "POST", u, &buf)
	if err != nil {
		return rawResponse{}, &VectorDBError{msg: fmt.Sprintf("failed to create request: %v", err)}
	}
	req.Header.Set("Content-Type", w.FormDataContentType())
	if opts != nil {
		for k, v := range opts.Headers {
			req.Header.Set(k, v)
		}
	}
	res, err := client.Do(req)
	if err != nil {
		return rawResponse{}, &VectorDBError{msg: fmt.Sprintf("request failed: %v", err)}
	}
	defer res.Body.Close()
	return raiseForStatus(res)
}

// multipartPost performs a POST request with multipart/form-data and
// returns the parsed JSON response.
func multipartPost(ctx context.Context, client *http.Client, baseURL, path string, fields map[string]io.Reader, fileField string, fileContents io.Reader, fileName string, opts *requestOptions) (map[string]interface{}, error) {
	rr, err := multipartPostRaw(ctx, client, baseURL, path, fields, fileField, fileContents, fileName, opts)
	if err != nil {
		return nil, err
	}
	return rr.parsed, nil
}
