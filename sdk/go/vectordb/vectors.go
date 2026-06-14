package vectordb

import (
	"context"

	"net/url"
)

// CreateVectorParams configures a new vector creation.
type CreateVectorParams struct {
	Vector   []float64
	Metadata map[string]interface{}
	VectorID string
}

// SearchVectorsParams configures a vector search.
type SearchVectorsParams struct {
	QueryVector []float64
	K           int
	Method      string
	Filters     map[string]interface{}
}

// VectorsAPI provides access to vector CRUD and search operations.
type VectorsAPI struct {
	client *VectorDBClient
}

// Create creates a new vector.
func (a *VectorsAPI) Create(ctx context.Context, params CreateVectorParams, opts ...RequestOption) (map[string]interface{}, error) {
	body := map[string]interface{}{
		"vector": params.Vector,
	}
	if params.Metadata != nil {
		body["metadata"] = params.Metadata
	}
	if params.VectorID != "" {
		body["vector_id"] = params.VectorID
	}

	o := &requestOptions{}
	for _, fn := range opts {
		fn(o)
	}
	merged := a.client.mergeOpts(o)
	return jsonPost(ctx, a.client.httpClient, a.client.BaseURL, "/vectors", body, merged)
}

// Get retrieves a vector by ID.
func (a *VectorsAPI) Get(ctx context.Context, vectorID string, opts ...RequestOption) (map[string]interface{}, error) {
	o := &requestOptions{}
	for _, fn := range opts {
		fn(o)
	}
	merged := a.client.mergeOpts(o)
	return jsonGet(ctx, a.client.httpClient, a.client.BaseURL, "/vectors/"+url.PathEscape(vectorID), merged)
}

// Delete removes a vector by ID.
func (a *VectorsAPI) Delete(ctx context.Context, vectorID string, opts ...RequestOption) (map[string]interface{}, error) {
	o := &requestOptions{}
	for _, fn := range opts {
		fn(o)
	}
	merged := a.client.mergeOpts(o)
	return jsonDelete(ctx, a.client.httpClient, a.client.BaseURL, "/vectors/"+url.PathEscape(vectorID), merged)
}

// Search performs a vector similarity search.
func (a *VectorsAPI) Search(ctx context.Context, params SearchVectorsParams, opts ...RequestOption) (map[string]interface{}, error) {
	body := map[string]interface{}{
		"query_vector": params.QueryVector,
		"k":            ifPositive(params.K, 5),
		"method":       orDefault(params.Method, "hnsw"),
	}
	if params.Filters != nil {
		body["filters"] = params.Filters
	}

	o := &requestOptions{}
	for _, fn := range opts {
		fn(o)
	}
	merged := a.client.mergeOpts(o)
	return jsonPost(ctx, a.client.httpClient, a.client.BaseURL, "/search", body, merged)
}
