package vectordb

import (
	"context"
	"fmt"
	"strings"
)

// CreateIndexParams configures building and populating an index.
type CreateIndexParams struct {
	IndexType      string
	Metric         string
	M              int
	M0             int
	EFConstruction int
	NClusters      int
	NProbes        int
}

// CompareSearchParams configures the search comparison.
type CompareSearchParams struct {
	QueryVector []float64
	K           int
}

// AnnAPI provides access to the ANN Index Management API.
type AnnAPI struct {
	client *VectorDBClient
}

// CreateIndex builds and populates an index from all vectors currently in the DB.
func (a *AnnAPI) CreateIndex(ctx context.Context, params CreateIndexParams, opts ...RequestOption) (map[string]interface{}, error) {
	body := map[string]interface{}{
		"index_type": params.IndexType,
	}
	if params.Metric != "" {
		body["metric"] = params.Metric
	}
	if params.M > 0 {
		body["m"] = params.M
	}
	if params.M0 > 0 {
		body["m0"] = params.M0
	}
	if params.EFConstruction > 0 {
		body["ef_construction"] = params.EFConstruction
	}
	if params.NClusters > 0 {
		body["n_clusters"] = params.NClusters
	}
	if params.NProbes > 0 {
		body["n_probes"] = params.NProbes
	}

	o := &requestOptions{}
	for _, fn := range opts {
		fn(o)
	}
	merged := a.client.mergeOpts(o)
	return jsonPost(ctx, a.client.httpClient, a.client.BaseURL, "/api/v1/ann/index", body, merged)
}

// GetIndexInfo returns stats for all loaded indexes, or just one if indexType is given.
func (a *AnnAPI) GetIndexInfo(ctx context.Context, indexType string, opts ...RequestOption) (map[string]interface{}, error) {
	o := &requestOptions{}
	for _, fn := range opts {
		fn(o)
	}
	if indexType != "" {
		if o.Params == nil {
			o.Params = make(map[string]string)
		}
		o.Params["index_type"] = indexType
	}
	merged := a.client.mergeOpts(o)
	return jsonGet(ctx, a.client.httpClient, a.client.BaseURL, "/api/v1/ann/index", merged)
}

// SaveIndex persists an in-memory index to disk.
func (a *AnnAPI) SaveIndex(ctx context.Context, indexType string, opts ...RequestOption) (map[string]interface{}, error) {
	o := &requestOptions{}
	for _, fn := range opts {
		fn(o)
	}
	if o.Params == nil {
		o.Params = make(map[string]string)
	}
	o.Params["index_type"] = indexType
	merged := a.client.mergeOpts(o)
	return jsonPost(ctx, a.client.httpClient, a.client.BaseURL, "/api/v1/ann/index/save", nil, merged)
}

// LoadIndex restores a previously saved index from disk into memory.
func (a *AnnAPI) LoadIndex(ctx context.Context, indexType string, opts ...RequestOption) (map[string]interface{}, error) {
	o := &requestOptions{}
	for _, fn := range opts {
		fn(o)
	}
	if o.Params == nil {
		o.Params = make(map[string]string)
	}
	o.Params["index_type"] = indexType
	merged := a.client.mergeOpts(o)
	return jsonPost(ctx, a.client.httpClient, a.client.BaseURL, "/api/v1/ann/index/load", nil, merged)
}

// CompareSearch runs the same query against every loaded index.
func (a *AnnAPI) CompareSearch(ctx context.Context, params CompareSearchParams, opts ...RequestOption) (map[string]interface{}, error) {
	o := &requestOptions{}
	for _, fn := range opts {
		fn(o)
	}
	if o.Params == nil {
		o.Params = make(map[string]string)
	}

	var vecStrs []string
	for _, val := range params.QueryVector {
		vecStrs = append(vecStrs, fmt.Sprintf("%g", val))
	}
	o.Params["query_vector"] = strings.Join(vecStrs, ",")
	if params.K > 0 {
		o.Params["k"] = intToStr(params.K)
	}

	merged := a.client.mergeOpts(o)
	return jsonGet(ctx, a.client.httpClient, a.client.BaseURL, "/api/v1/ann/search/compare", merged)
}

// GetStatistics returns DB vector count and stats for all loaded indexes.
func (a *AnnAPI) GetStatistics(ctx context.Context, opts ...RequestOption) (map[string]interface{}, error) {
	o := &requestOptions{}
	for _, fn := range opts {
		fn(o)
	}
	merged := a.client.mergeOpts(o)
	return jsonGet(ctx, a.client.httpClient, a.client.BaseURL, "/api/v1/ann/stats", merged)
}
