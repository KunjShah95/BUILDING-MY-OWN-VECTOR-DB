package vectordb

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
)

// CreateCollectionParams configures a new collection creation.
type CreateCollectionParams struct {
	Name           string
	CollectionID   string
	Modality       string
	Dimension      int
	EmbeddingModel string
	Description    string
	DistanceMetric string
}

// CollectionsAPI provides access to collection CRUD and index operations.
type CollectionsAPI struct {
	client *VectorDBClient
}

// resultCollection extracts a Collection from an API response's "collection" field.
func resultCollection(rr rawResponse) (Collection, error) {
	rawColl, ok := rr.parsed["collection"].(map[string]interface{})
	if !ok {
		return Collection{}, &VectorDBError{msg: "unexpected response format: missing 'collection' field"}
	}

	type collectionEnvelope struct {
		Collection CollectionData `json:"collection"`
	}
	var env collectionEnvelope
	if err := json.Unmarshal(rr.raw, &env); err != nil {
		return Collection{}, err
	}

	dm := env.Collection.DistanceMetric
	if dm == "" {
		dm = "cosine"
	}
	return Collection{
		CollectionID:   env.Collection.CollectionID,
		Name:           env.Collection.Name,
		Modality:       env.Collection.Modality,
		Dimension:      env.Collection.Dimension,
		EmbeddingModel: env.Collection.EmbeddingModel,
		DistanceMetric: dm,
		Description:    env.Collection.Description,
		Raw:            rawColl,
	}, nil
}

// Create creates a new collection.
func (a *CollectionsAPI) Create(ctx context.Context, params CreateCollectionParams, opts ...RequestOption) (Collection, error) {
	body := map[string]interface{}{
		"name":            params.Name,
		"modality":        orDefault(params.Modality, "text"),
		"distance_metric": orDefault(params.DistanceMetric, "cosine"),
	}
	if params.CollectionID != "" {
		body["collection_id"] = params.CollectionID
	}
	if params.Dimension > 0 {
		body["dimension"] = params.Dimension
	}
	if params.EmbeddingModel != "" {
		body["embedding_model"] = params.EmbeddingModel
	}
	if params.Description != "" {
		body["description"] = params.Description
	}

	o := &requestOptions{}
	for _, fn := range opts {
		fn(o)
	}
	merged := a.client.mergeOpts(o)

	rr, err := jsonPostRaw(ctx, a.client.httpClient, a.client.BaseURL, "/collections", body, merged)
	if err != nil {
		return Collection{}, err
	}
	return resultCollection(rr)
}

// List returns collections with pagination.
func (a *CollectionsAPI) List(ctx context.Context, limit, offset int, opts ...RequestOption) ([]Collection, error) {
	o := &requestOptions{}
	for _, fn := range opts {
		fn(o)
	}
	merged := a.client.mergeOpts(o)
	if merged.Params == nil {
		merged.Params = make(map[string]string)
	}
	if limit <= 0 {
		limit = 100
	}
	merged.Params["limit"] = intToStr(limit)
	merged.Params["offset"] = intToStr(offset)

	type listResponse struct {
		Collections []CollectionData `json:"collections"`
	}

	rr, err := jsonGetRaw(ctx, a.client.httpClient, a.client.BaseURL, "/collections", merged)
	if err != nil {
		return nil, err
	}

	var resp listResponse
	if err := json.Unmarshal(rr.raw, &resp); err != nil || resp.Collections == nil {
		return []Collection{}, nil
	}

	collections := make([]Collection, len(resp.Collections))
	for i, cd := range resp.Collections {
		dm := cd.DistanceMetric
		if dm == "" {
			dm = "cosine"
		}
		collections[i] = Collection{
			CollectionID:   cd.CollectionID,
			Name:           cd.Name,
			Modality:       cd.Modality,
			Dimension:      cd.Dimension,
			EmbeddingModel: cd.EmbeddingModel,
			DistanceMetric: dm,
			Description:    cd.Description,
			Raw:            rr.parsed,
		}
	}
	return collections, nil
}

// Get retrieves a single collection by ID.
func (a *CollectionsAPI) Get(ctx context.Context, collectionID string, opts ...RequestOption) (Collection, error) {
	o := &requestOptions{}
	for _, fn := range opts {
		fn(o)
	}
	merged := a.client.mergeOpts(o)

	rr, err := jsonGetRaw(ctx, a.client.httpClient, a.client.BaseURL, "/collections/"+url.PathEscape(collectionID), merged)
	if err != nil {
		return Collection{}, err
	}
	return resultCollection(rr)
}

// Delete removes a collection by ID.
func (a *CollectionsAPI) Delete(ctx context.Context, collectionID string, opts ...RequestOption) (map[string]interface{}, error) {
	o := &requestOptions{}
	for _, fn := range opts {
		fn(o)
	}
	merged := a.client.mergeOpts(o)
	return jsonDelete(ctx, a.client.httpClient, a.client.BaseURL, "/collections/"+url.PathEscape(collectionID), merged)
}

// BuildIndexParams configures index building.
type BuildIndexParams struct {
	Method         string
	M              int
	M0             int
	EFConstruction int
	NClusters      int
	NProbes        int
}

// BuildIndex builds an index for a collection.
func (a *CollectionsAPI) BuildIndex(ctx context.Context, collectionID string, params BuildIndexParams, opts ...RequestOption) (map[string]interface{}, error) {
	body := map[string]interface{}{
		"method":          orDefault(params.Method, "hnsw"),
		"m":               ifPositive(params.M, 16),
		"ef_construction": ifPositive(params.EFConstruction, 200),
		"n_clusters":      ifPositive(params.NClusters, 100),
		"n_probes":        ifPositive(params.NProbes, 10),
	}
	if params.M0 > 0 {
		body["m0"] = params.M0
	}

	o := &requestOptions{}
	for _, fn := range opts {
		fn(o)
	}
	merged := a.client.mergeOpts(o)
	return jsonPost(ctx, a.client.httpClient, a.client.BaseURL, "/collections/"+url.PathEscape(collectionID)+"/index", body, merged)
}

// IndexStats returns index statistics for a collection.
func (a *CollectionsAPI) IndexStats(ctx context.Context, collectionID string, opts ...RequestOption) (map[string]interface{}, error) {
	o := &requestOptions{}
	for _, fn := range opts {
		fn(o)
	}
	merged := a.client.mergeOpts(o)
	return jsonGet(ctx, a.client.httpClient, a.client.BaseURL, "/collections/"+url.PathEscape(collectionID)+"/index/stats", merged)
}

// --- Helpers ---

func orDefault(val, def string) string {
	if val == "" {
		return def
	}
	return val
}

func ifPositive(val, def int) int {
	if val > 0 {
		return val
	}
	return def
}

func intToStr(val int) string {
	return fmt.Sprintf("%d", val)
}
