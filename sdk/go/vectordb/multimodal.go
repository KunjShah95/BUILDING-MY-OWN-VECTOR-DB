package vectordb

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"net/url"
)

// IngestTextParams configures text ingestion.
type IngestTextParams struct {
	Text     string
	Metadata map[string]interface{}
	VectorID string
}

// IngestFileParams configures file-based ingestion (image/audio).
type IngestFileParams struct {
	Contents io.Reader
	Filename string
	Metadata map[string]interface{}
	VectorID string
}

// SearchTextParams configures a text-based search.
type SearchTextParams struct {
	Query   string
	K       int
	Method  string
	Filters map[string]interface{}
}

// SearchFileParams configures a file-based search (image/audio).
type SearchFileParams struct {
	Contents io.Reader
	Filename string
	K        int
	Method   string
	Filters  map[string]interface{}
}

// MultimodalAPI provides access to multimodal (text, image, audio) operations.
type MultimodalAPI struct {
	client *VectorDBClient
}

// IngestText ingests text into a collection.
func (a *MultimodalAPI) IngestText(ctx context.Context, collectionID string, params IngestTextParams, opts ...RequestOption) (map[string]interface{}, error) {
	body := map[string]interface{}{
		"text": params.Text,
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
	return jsonPost(ctx, a.client.httpClient, a.client.BaseURL, "/collections/"+url.PathEscape(collectionID)+"/ingest/text", body, merged)
}

// SearchText performs a text-based search on a collection and returns typed results.
func (a *MultimodalAPI) SearchText(ctx context.Context, collectionID string, params SearchTextParams, opts ...RequestOption) (SearchResult, error) {
	body := map[string]interface{}{
		"query":  params.Query,
		"k":      ifPositive(params.K, 5),
		"method": orDefault(params.Method, "brute"),
	}
	if params.Filters != nil {
		body["filters"] = params.Filters
	}

	o := &requestOptions{}
	for _, fn := range opts {
		fn(o)
	}
	merged := a.client.mergeOpts(o)

	rr, err := jsonPostRaw(ctx, a.client.httpClient, a.client.BaseURL, "/collections/"+url.PathEscape(collectionID)+"/search/text", body, merged)
	if err != nil {
		return SearchResult{}, err
	}
	return searchResultFromRaw(rr)
}

// IngestImage ingests an image file into a collection.
func (a *MultimodalAPI) IngestImage(ctx context.Context, collectionID string, params IngestFileParams, opts ...RequestOption) (map[string]interface{}, error) {
	if params.Contents == nil {
		return nil, &VectorDBError{msg: "provide contents for image ingest"}
	}
	return a.multipartUpload(ctx, collectionID, "ingest", "image", params, opts...)
}

// SearchImage performs an image-based search and returns typed results.
func (a *MultimodalAPI) SearchImage(ctx context.Context, collectionID string, params SearchFileParams, opts ...RequestOption) (SearchResult, error) {
	if params.Contents == nil {
		return SearchResult{}, &VectorDBError{msg: "provide contents for image search"}
	}
	return a.multipartSearch(ctx, collectionID, "search", "image", params, opts...)
}

// IngestAudio ingests an audio file into a collection.
func (a *MultimodalAPI) IngestAudio(ctx context.Context, collectionID string, params IngestFileParams, opts ...RequestOption) (map[string]interface{}, error) {
	if params.Contents == nil {
		return nil, &VectorDBError{msg: "provide contents for audio ingest"}
	}
	return a.multipartUpload(ctx, collectionID, "ingest", "audio", params, opts...)
}

// SearchAudio performs an audio-based search and returns typed results.
func (a *MultimodalAPI) SearchAudio(ctx context.Context, collectionID string, params SearchFileParams, opts ...RequestOption) (SearchResult, error) {
	if params.Contents == nil {
		return SearchResult{}, &VectorDBError{msg: "provide contents for audio search"}
	}
	return a.multipartSearch(ctx, collectionID, "search", "audio", params, opts...)
}

// Query performs a RAG query on a collection.
func (a *MultimodalAPI) Query(ctx context.Context, collectionID, query string, k int, opts ...RequestOption) (map[string]interface{}, error) {
	body := map[string]interface{}{
		"query": query,
	}
	if k > 0 {
		body["k"] = k
	}

	o := &requestOptions{}
	for _, fn := range opts {
		fn(o)
	}
	merged := a.client.mergeOpts(o)
	return jsonPost(ctx, a.client.httpClient, a.client.BaseURL, "/collections/"+url.PathEscape(collectionID)+"/query", body, merged)
}

// --- Internal helpers ---

func (a *MultimodalAPI) multipartUpload(ctx context.Context, collectionID, action, mediaType string, params IngestFileParams, opts ...RequestOption) (map[string]interface{}, error) {
	path := fmt.Sprintf("/collections/%s/%s/%s", url.PathEscape(collectionID), action, mediaType)
	filename := params.Filename
	if filename == "" {
		filename = "file"
	}

	fields := make(map[string]io.Reader)
	if params.Metadata != nil {
		metaBytes, _ := json.Marshal(params.Metadata)
		fields["metadata"] = strings.NewReader(string(metaBytes))
	}
	if params.VectorID != "" {
		fields["vector_id"] = strings.NewReader(params.VectorID)
	}

	o := &requestOptions{}
	for _, fn := range opts {
		fn(o)
	}
	merged := a.client.mergeOpts(o)

	return multipartPost(ctx, a.client.httpClient, a.client.BaseURL, path, fields, "file", params.Contents, filename, merged)
}

func (a *MultimodalAPI) multipartSearch(ctx context.Context, collectionID, action, mediaType string, params SearchFileParams, opts ...RequestOption) (SearchResult, error) {
	path := fmt.Sprintf("/collections/%s/%s/%s", url.PathEscape(collectionID), action, mediaType)
	filename := params.Filename
	if filename == "" {
		filename = "file"
	}

	fields := make(map[string]io.Reader)
	k := params.K
	if k <= 0 {
		k = 5
	}
	fields["k"] = strings.NewReader(fmt.Sprintf("%d", k))
	if params.Method != "" {
		fields["method"] = strings.NewReader(params.Method)
	}
	if params.Filters != nil {
		filterBytes, _ := json.Marshal(params.Filters)
		fields["filters"] = bytes.NewReader(filterBytes)
	}

	o := &requestOptions{}
	for _, fn := range opts {
		fn(o)
	}
	merged := a.client.mergeOpts(o)

	rr, err := multipartPostRaw(ctx, a.client.httpClient, a.client.BaseURL, path, fields, "file", params.Contents, filename, merged)
	if err != nil {
		return SearchResult{}, err
	}
	return searchResultFromRaw(rr)
}

// searchResultFromRaw unmarshals a search result directly from raw JSON bytes.
func searchResultFromRaw(rr rawResponse) (SearchResult, error) {
	var data SearchResultData
	if err := json.Unmarshal(rr.raw, &data); err != nil {
		return SearchResult{}, err
	}

	hits := make([]SearchHit, len(data.Results))
	for i, r := range data.Results {
		meta := r.Metadata
		if meta == nil {
			meta = r.MetaData
		}
		hits[i] = SearchHit{
			VectorID: r.VectorID,
			Distance: r.Distance,
			Metadata: meta,
		}
	}

	total := len(hits)
	if data.TotalResults != nil {
		total = *data.TotalResults
	}
	var st float64
	if data.SearchTime != nil {
		st = *data.SearchTime
	}

	return SearchResult{
		Success:      data.Success,
		Results:      hits,
		TotalResults: total,
		SearchTime:   st,
		Method:       data.Method,
		Raw:          rr.parsed,
	}, nil
}
