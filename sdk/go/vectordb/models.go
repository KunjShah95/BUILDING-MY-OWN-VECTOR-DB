package vectordb

// CollectionData represents the raw API response for a collection.
type CollectionData struct {
	CollectionID   string `json:"collection_id"`
	Name           string `json:"name"`
	Modality       string `json:"modality"`
	Dimension      int    `json:"dimension"`
	EmbeddingModel string `json:"embedding_model"`
	DistanceMetric string `json:"distance_metric,omitempty"`
	Description    string `json:"description,omitempty"`
}

// Collection is a typed wrapper around a collection from the API.
type Collection struct {
	CollectionID   string
	Name           string
	Modality       string
	Dimension      int
	EmbeddingModel string
	DistanceMetric string
	Description    string
	Raw            map[string]interface{}
}

// SearchHitData represents a single search result from the API.
type SearchHitData struct {
	VectorID string                 `json:"vector_id"`
	Distance float64                `json:"distance"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
	MetaData map[string]interface{} `json:"meta_data,omitempty"`
}

// SearchHit is a typed wrapper around a single search result.
type SearchHit struct {
	VectorID string
	Distance float64
	Metadata map[string]interface{}
}

// SearchResultData represents the raw API response for a search.
type SearchResultData struct {
	Success      bool            `json:"success"`
	Results      []SearchHitData `json:"results"`
	TotalResults *int            `json:"total_results,omitempty"`
	SearchTime   *float64        `json:"search_time,omitempty"`
	Method       string          `json:"method,omitempty"`
}

// SearchResult is a typed wrapper around a search API response.
type SearchResult struct {
	Success      bool
	Results      []SearchHit
	TotalResults int
	SearchTime   float64
	Method       string
	Raw          map[string]interface{}
}
