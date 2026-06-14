package vectordb

import (
	"context"
	"net/http"
	"strings"
	"time"
)

// ClientOptions configures the VectorDBClient.
type ClientOptions struct {
	BaseURL string
	Timeout time.Duration
	Headers map[string]string
	// HTTPClient allows providing a custom HTTP client (e.g. with transport
	// configuration). If nil, a default one is created.
	HTTPClient *http.Client
}

// VectorDBClient is the main client for the Vector Database API.
type VectorDBClient struct {
	BaseURL     string
	httpClient  *http.Client
	defaultOpts *requestOptions

	Collections *CollectionsAPI
	Vectors     *VectorsAPI
	Multimodal  *MultimodalAPI
}

// NewClient creates a new VectorDBClient.
func NewClient(options ...ClientOptions) *VectorDBClient {
	var opts ClientOptions
	if len(options) > 0 {
		opts = options[0]
	}
	if opts.BaseURL == "" {
		opts.BaseURL = "http://localhost:8000"
	}
	opts.BaseURL = strings.TrimRight(opts.BaseURL, "/")

	if opts.Timeout == 0 {
		opts.Timeout = 60 * time.Second
	}

	httpClient := opts.HTTPClient
	if httpClient == nil {
		httpClient = &http.Client{
			Timeout: opts.Timeout,
		}
	}

	c := &VectorDBClient{
		BaseURL:    opts.BaseURL,
		httpClient: httpClient,
		defaultOpts: &requestOptions{
			Headers: opts.Headers,
		},
	}

	c.Collections = &CollectionsAPI{client: c}
	c.Vectors = &VectorsAPI{client: c}
	c.Multimodal = &MultimodalAPI{client: c}

	return c
}

// mergeOpts merges per-request options on top of the client's default options.
func (c *VectorDBClient) mergeOpts(opts *requestOptions) *requestOptions {
	merged := &requestOptions{}
	if c.defaultOpts != nil {
		if c.defaultOpts.Headers != nil {
			merged.Headers = make(map[string]string)
			for k, v := range c.defaultOpts.Headers {
				merged.Headers[k] = v
			}
		}
		if c.defaultOpts.Params != nil {
			merged.Params = make(map[string]string)
			for k, v := range c.defaultOpts.Params {
				merged.Params[k] = v
			}
		}
	}
	if opts != nil {
		if merged.Headers == nil {
			merged.Headers = make(map[string]string)
		}
		for k, v := range opts.Headers {
			merged.Headers[k] = v
		}
		if merged.Params == nil {
			merged.Params = make(map[string]string)
		}
		for k, v := range opts.Params {
			merged.Params[k] = v
		}
	}
	return merged
}

// Health performs a health check against the API.
func (c *VectorDBClient) Health(ctx context.Context, opts ...RequestOption) (map[string]interface{}, error) {
	o := &requestOptions{}
	for _, fn := range opts {
		fn(o)
	}
	merged := c.mergeOpts(o)
	return jsonGet(ctx, c.httpClient, c.BaseURL, "/health", merged)
}
