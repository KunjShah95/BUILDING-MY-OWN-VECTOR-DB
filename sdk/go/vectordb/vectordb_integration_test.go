package vectordb

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// ---------------------------------------------------------------------------
// Integration tests — real HTTP test server with dynamic routes
// ---------------------------------------------------------------------------

func setupTestServer(t *testing.T) (*httptest.Server, *VectorDBClient, context.Context) {
	t.Helper()

	mux := http.NewServeMux()

	// Health
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"success":true,"status":"ok"}`))
	})

	// Create collection
	mux.HandleFunc("/collections", func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"collections":[{"collection_id":"integ-test-col","name":"Integration","modality":"text","dimension":384,"embedding_model":"text-embedding-3-small"}],"total":1}`))
		case http.MethodPost:
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusCreated)
			var body map[string]interface{}
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				http.Error(w, `{"detail":"bad request"}`, http.StatusBadRequest)
				return
			}
			name, _ := body["name"].(string)
			colID, _ := body["collection_id"].(string)
			dim, _ := body["dimension"].(float64)
			resp := fmt.Sprintf(`{"collection":{"collection_id":"%s","name":"%s","modality":"text","dimension":%d,"embedding_model":"text-embedding-3-small"},"success":true}`, colID, name, int(dim))
			_, _ = w.Write([]byte(resp))
		default:
			http.Error(w, `{"detail":"method not allowed"}`, http.StatusMethodNotAllowed)
		}
	})

	// Single collection
	mux.HandleFunc("/collections/", func(w http.ResponseWriter, r *http.Request) {
		// Extract the collection ID from the path
		path := strings.TrimPrefix(r.URL.Path, "/collections/")
		// Strip sub-paths like /index, /index/stats, /ingest/text, /search/text, /query
		colID := path
		if idx := strings.Index(path, "/"); idx >= 0 {
			colID = path[:idx]
		}

		if strings.HasSuffix(r.URL.Path, "/index") {
			// Build index
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"success":true,"message":"Index built","method":"hnsw"}`))
			return
		}
		if strings.HasSuffix(r.URL.Path, "/index/stats") {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"success":true,"num_vectors":42,"index_type":"hnsw"}`))
			return
		}
		if strings.HasSuffix(r.URL.Path, "/ingest/text") {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"success":true,"vector_id":"test-vec","message":"Text ingested"}`))
			return
		}
		if strings.HasSuffix(r.URL.Path, "/search/text") {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"success":true,"results":[{"vector_id":"match-1","distance":0.15,"metadata":{"text":"hello world"}}],"total_results":1,"search_time":0.023,"method":"brute"}`))
			return
		}
		if strings.HasSuffix(r.URL.Path, "/query") {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"success":true,"answer":"The return policy is 30 days.","sources":[{"vector_id":"policy-returns","distance":0.12}],"search_time":0.045}`))
			return
		}

		switch r.Method {
		case http.MethodGet:
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(fmt.Sprintf(`{"collection":{"collection_id":"%s","name":"%s","modality":"text","dimension":384,"embedding_model":"text-embedding-3-small"},"success":true}`, colID, colID)))
		case http.MethodDelete:
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(fmt.Sprintf(`{"success":true,"message":"Collection %s deleted"}`, colID)))
		default:
			http.Error(w, `{"detail":"method not allowed"}`, http.StatusMethodNotAllowed)
		}
	})

	// Vectors
	mux.HandleFunc("/vectors", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusCreated)
			// Echo back the vector_id from the request body
			var body map[string]interface{}
			if err := json.NewDecoder(r.Body).Decode(&body); err == nil {
				if vid, ok := body["vector_id"].(string); ok && vid != "" {
					_, _ = w.Write([]byte(fmt.Sprintf(`{"success":true,"vector_id":"%s"}`, vid)))
					return
				}
			}
			_, _ = w.Write([]byte(`{"success":true,"vector_id":"vec-123"}`))
		}
	})

	// Single vector
	mux.HandleFunc("/vectors/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		vecID := strings.TrimPrefix(r.URL.Path, "/vectors/")
		switch r.Method {
		case http.MethodGet:
			_, _ = w.Write([]byte(fmt.Sprintf(`{"success":true,"vector_id":"%s","vector":[0.1,0.2,0.3],"metadata":{"source":"test"}}`, vecID)))
		case http.MethodDelete:
			_, _ = w.Write([]byte(fmt.Sprintf(`{"success":true,"message":"Vector %s deleted"}`, vecID)))
		default:
			http.Error(w, `{"detail":"method not allowed"}`, http.StatusMethodNotAllowed)
		}
	})

	// Search
	mux.HandleFunc("/search", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"success":true,"results":[{"vector_id":"match-1","distance":0.12,"metadata":{"tag":"test"}}],"total_results":1,"search_time":0.015,"method":"hnsw"}`))
	})

	// Error routes
	mux.HandleFunc("/error/401", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"detail":"Invalid API key"}`))
	})
	mux.HandleFunc("/error/403", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusForbidden)
		_, _ = w.Write([]byte(`{"detail":"Forbidden"}`))
	})
	mux.HandleFunc("/error/404", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte(`{"detail":"Not Found"}`))
	})
	mux.HandleFunc("/error/500", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"detail":"Internal Server Error"}`))
	})

	// Custom header echo
	mux.HandleFunc("/echo-headers", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		auth := r.Header.Get("Authorization")
		xKey := r.Header.Get("X-API-Key")
		_, _ = w.Write([]byte(fmt.Sprintf(`{"auth":"%s","x-api-key":"%s"}`, auth, xKey)))
	})

	// URL encoding test route - matches any GET to /collections/%3Awith%20specials
	mux.HandleFunc("/collections/:with specials", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"collection":{"collection_id":":with specials","name":"Special Collection","modality":"text","dimension":384},"success":true}`))
	})

	server := httptest.NewServer(mux)

	client := NewClient(ClientOptions{
		BaseURL: server.URL,
		Timeout: 5 * time.Second,
		Headers: map[string]string{
			"Authorization": "Bearer test-token",
		},
	})

	return server, client, context.Background()
}

func TestIntegrationHealthCheck(t *testing.T) {
	server, client, ctx := setupTestServer(t)
	defer server.Close()

	result, err := client.Health(ctx)
	if err != nil {
		t.Fatalf("Health() returned error: %v", err)
	}
	if result["status"] != "ok" {
		t.Errorf("expected status 'ok', got %v", result["status"])
	}
}

func TestIntegrationCreateCollection(t *testing.T) {
	server, client, ctx := setupTestServer(t)
	defer server.Close()

	col, err := client.Collections.Create(ctx, CreateCollectionParams{
		Name:         "Test Collection",
		CollectionID: "test-col",
		Dimension:    384,
		Modality:     "text",
	})
	if err != nil {
		t.Fatalf("Create() returned error: %v", err)
	}
	if col.CollectionID != "test-col" {
		t.Errorf("expected collection_id 'test-col', got %q", col.CollectionID)
	}
	if col.Name != "Test Collection" {
		t.Errorf("expected name 'Test Collection', got %q", col.Name)
	}
}

func TestIntegrationListCollections(t *testing.T) {
	server, client, ctx := setupTestServer(t)
	defer server.Close()

	cols, err := client.Collections.List(ctx, 100, 0)
	if err != nil {
		t.Fatalf("List() returned error: %v", err)
	}
	if len(cols) == 0 {
		t.Fatal("expected at least one collection")
	}
	if cols[0].CollectionID != "integ-test-col" {
		t.Errorf("expected collection_id 'integ-test-col', got %q", cols[0].CollectionID)
	}
}

func TestIntegrationGetCollection(t *testing.T) {
	server, client, ctx := setupTestServer(t)
	defer server.Close()

	col, err := client.Collections.Get(ctx, "test-col")
	if err != nil {
		t.Fatalf("Get() returned error: %v", err)
	}
	if col.CollectionID != "test-col" {
		t.Errorf("expected collection_id 'test-col', got %q", col.CollectionID)
	}
}

func TestIntegrationDeleteCollection(t *testing.T) {
	server, client, ctx := setupTestServer(t)
	defer server.Close()

	result, err := client.Collections.Delete(ctx, "test-col")
	if err != nil {
		t.Fatalf("Delete() returned error: %v", err)
	}
	if msg, ok := result["message"].(string); !ok || !strings.Contains(msg, "test-col") {
		t.Errorf("unexpected delete message: %v", result["message"])
	}
}

func TestIntegrationBuildIndex(t *testing.T) {
	server, client, ctx := setupTestServer(t)
	defer server.Close()

	result, err := client.Collections.BuildIndex(ctx, "test-col", BuildIndexParams{
		Method: "hnsw",
	})
	if err != nil {
		t.Fatalf("BuildIndex() returned error: %v", err)
	}
	if result["method"] != "hnsw" {
		t.Errorf("expected method 'hnsw', got %v", result["method"])
	}
}

func TestIntegrationIndexStats(t *testing.T) {
	server, client, ctx := setupTestServer(t)
	defer server.Close()

	stats, err := client.Collections.IndexStats(ctx, "test-col")
	if err != nil {
		t.Fatalf("IndexStats() returned error: %v", err)
	}
	if n, ok := stats["num_vectors"].(float64); !ok || n != 42 {
		t.Errorf("expected num_vectors 42, got %v", stats["num_vectors"])
	}
}

func TestIntegrationCreateVector(t *testing.T) {
	server, client, ctx := setupTestServer(t)
	defer server.Close()

	result, err := client.Vectors.Create(ctx, CreateVectorParams{
		Vector:   []float64{0.1, 0.2, 0.3},
		VectorID: "vec-123",
	})
	if err != nil {
		t.Fatalf("Create() returned error: %v", err)
	}
	if result["vector_id"] != "vec-123" {
		t.Errorf("expected vector_id 'vec-123', got %v", result["vector_id"])
	}
}

func TestIntegrationGetVector(t *testing.T) {
	server, client, ctx := setupTestServer(t)
	defer server.Close()

	result, err := client.Vectors.Get(ctx, "vec-123")
	if err != nil {
		t.Fatalf("Get() returned error: %v", err)
	}
	if result["vector_id"] != "vec-123" {
		t.Errorf("expected vector_id 'vec-123', got %v", result["vector_id"])
	}
}

func TestIntegrationDeleteVector(t *testing.T) {
	server, client, ctx := setupTestServer(t)
	defer server.Close()

	result, err := client.Vectors.Delete(ctx, "vec-123")
	if err != nil {
		t.Fatalf("Delete() returned error: %v", err)
	}
	if msg, ok := result["message"].(string); !ok || !strings.Contains(msg, "vec-123") {
		t.Errorf("unexpected delete message: %v", result["message"])
	}
}

func TestIntegrationSearchVectors(t *testing.T) {
	server, client, ctx := setupTestServer(t)
	defer server.Close()

	result, err := client.Vectors.Search(ctx, SearchVectorsParams{
		QueryVector: []float64{0.1, 0.2, 0.3},
		K:           5,
	})
	if err != nil {
		t.Fatalf("Search() returned error: %v", err)
	}
	results, ok := result["results"].([]interface{})
	if !ok || len(results) == 0 {
		t.Fatal("expected at least one search result")
	}
}

func TestIntegrationMultimodalTextIngest(t *testing.T) {
	server, client, ctx := setupTestServer(t)
	defer server.Close()

	result, err := client.Multimodal.IngestText(ctx, "test-col", IngestTextParams{
		Text:     "Hello world",
		VectorID: "test-vec",
	})
	if err != nil {
		t.Fatalf("IngestText() returned error: %v", err)
	}
	if result["vector_id"] != "test-vec" {
		t.Errorf("expected vector_id 'test-vec', got %v", result["vector_id"])
	}
}

func TestIntegrationMultimodalSearchText(t *testing.T) {
	server, client, ctx := setupTestServer(t)
	defer server.Close()

	result, err := client.Multimodal.SearchText(ctx, "test-col", SearchTextParams{
		Query: "hello",
		K:     5,
	})
	if err != nil {
		t.Fatalf("SearchText() returned error: %v", err)
	}
	if !result.Success {
		t.Error("expected success=true")
	}
	if len(result.Results) == 0 {
		t.Fatal("expected at least one search result")
	}
	if result.Results[0].VectorID != "match-1" {
		t.Errorf("expected vector_id 'match-1', got %q", result.Results[0].VectorID)
	}
	if result.Results[0].Distance != 0.15 {
		t.Errorf("expected distance 0.15, got %f", result.Results[0].Distance)
	}
}

func TestIntegrationMultimodalQuery(t *testing.T) {
	server, client, ctx := setupTestServer(t)
	defer server.Close()

	result, err := client.Multimodal.Query(ctx, "test-col", "What is the return policy?", 5)
	if err != nil {
		t.Fatalf("Query() returned error: %v", err)
	}
	if answer, ok := result["answer"].(string); !ok || answer == "" {
		t.Errorf("expected non-empty answer, got %v", result["answer"])
	}
}

func TestIntegrationHTTPError401(t *testing.T) {
	server, client, ctx := setupTestServer(t)
	defer server.Close()

	_, err := jsonGet(ctx, client.httpClient, client.BaseURL, "/error/401", nil)
	if err == nil {
		t.Fatal("expected error for 401")
	}
	httpErr, ok := err.(*VectorDBHTTPError)
	if !ok {
		t.Fatalf("expected *VectorDBHTTPError, got %T: %v", err, err)
	}
	if httpErr.StatusCode != 401 {
		t.Errorf("expected status 401, got %d", httpErr.StatusCode)
	}
}

func TestIntegrationHTTPError403(t *testing.T) {
	server, client, ctx := setupTestServer(t)
	defer server.Close()

	_, err := jsonGet(ctx, client.httpClient, client.BaseURL, "/error/403", nil)
	httpErr, ok := err.(*VectorDBHTTPError)
	if !ok {
		t.Fatalf("expected *VectorDBHTTPError, got %T: %v", err, err)
	}
	if httpErr.StatusCode != 403 {
		t.Errorf("expected status 403, got %d", httpErr.StatusCode)
	}
}

func TestIntegrationHTTPError404(t *testing.T) {
	server, client, ctx := setupTestServer(t)
	defer server.Close()

	_, err := jsonGet(ctx, client.httpClient, client.BaseURL, "/error/404", nil)
	httpErr, ok := err.(*VectorDBHTTPError)
	if !ok {
		t.Fatalf("expected *VectorDBHTTPError, got %T: %v", err, err)
	}
	if httpErr.StatusCode != 404 {
		t.Errorf("expected status 404, got %d", httpErr.StatusCode)
	}
}

func TestIntegrationHTTPError500(t *testing.T) {
	server, client, ctx := setupTestServer(t)
	defer server.Close()

	_, err := jsonGet(ctx, client.httpClient, client.BaseURL, "/error/500", nil)
	httpErr, ok := err.(*VectorDBHTTPError)
	if !ok {
		t.Fatalf("expected *VectorDBHTTPError, got %T: %v", err, err)
	}
	if httpErr.StatusCode != 500 {
		t.Errorf("expected status 500, got %d", httpErr.StatusCode)
	}
}

func TestIntegrationCustomHeaders(t *testing.T) {
	// This test is covered by TestEdgeClientHeadersPropagated in the edge case file
	// which properly verifies client-level headers propagate via the recorded-headers pattern.
}

func TestIntegrationPerRequestHeaderOverride(t *testing.T) {
	// This test is covered by TestEdgePerRequestHeadersOverrideClientHeaders in the edge case file
	// which properly verifies per-request headers override client-level headers via the recorded-headers pattern.
}

func TestIntegrationURLEncodedCollectionID(t *testing.T) {
	server, client, ctx := setupTestServer(t)
	defer server.Close()

	// Collection ID with special characters (colon and space)
	col, err := client.Collections.Get(ctx, ":with specials")
	if err != nil {
		t.Fatalf("Get() returned error: %v", err)
	}
	if col.CollectionID != ":with specials" {
		t.Errorf("expected collection_id ':with specials', got %q", col.CollectionID)
	}
}

func TestIntegrationTypedSearchResult(t *testing.T) {
	server, client, ctx := setupTestServer(t)
	defer server.Close()

	// Create collection returns typed Collection model
	col, err := client.Collections.Create(ctx, CreateCollectionParams{
		Name:         "Typed Test",
		CollectionID: "typed-test",
		Dimension:    768,
		Modality:     "text",
	})
	if err != nil {
		t.Fatalf("Create() returned error: %v", err)
	}
	if col.CollectionID != "typed-test" {
		t.Errorf("expected CollectionID 'typed-test', got %q", col.CollectionID)
	}
	if col.Dimension != 768 {
		t.Errorf("expected Dimension 768, got %d", col.Dimension)
	}

	// Search returns typed SearchResult
	result, err := client.Multimodal.SearchText(ctx, "typed-test", SearchTextParams{
		Query: "test",
		K:     5,
	})
	if err != nil {
		t.Fatalf("SearchText() returned error: %v", err)
	}
	if !result.Success {
		t.Error("expected Success=true")
	}
	if len(result.Results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(result.Results))
	}
	if result.Results[0].Distance != 0.15 {
		t.Errorf("expected Distance 0.15, got %f", result.Results[0].Distance)
	}
	if result.TotalResults != 1 {
		t.Errorf("expected TotalResults 1, got %d", result.TotalResults)
	}
	if result.SearchTime != 0.023 {
		t.Errorf("expected SearchTime 0.023, got %f", result.SearchTime)
	}
	if result.Method != "brute" {
		t.Errorf("expected Method 'brute', got %q", result.Method)
	}
}

func TestIntegrationFullLifecycle(t *testing.T) {
	server, client, ctx := setupTestServer(t)
	defer server.Close()

	// 1. Create collection
	col, err := client.Collections.Create(ctx, CreateCollectionParams{
		Name:         "Lifecycle",
		CollectionID: "lifecycle-test",
		Dimension:    384,
	})
	if err != nil {
		t.Fatalf("step 1 - Create() returned error: %v", err)
	}
	if col.CollectionID != "lifecycle-test" {
		t.Fatalf("expected lifecycle-test, got %q", col.CollectionID)
	}

	// 2. Create vector
	vecResult, err := client.Vectors.Create(ctx, CreateVectorParams{
		Vector:   []float64{0.1, 0.2, 0.3},
		VectorID: "lifecycle-vec",
	})
	if err != nil {
		t.Fatalf("step 2 - Create() returned error: %v", err)
	}
	if vecResult["vector_id"] != "lifecycle-vec" {
		t.Errorf("expected lifecycle-vec, got %v", vecResult["vector_id"])
	}

	// 3. Get vector
	getResult, err := client.Vectors.Get(ctx, "lifecycle-vec")
	if err != nil {
		t.Fatalf("step 3 - Get() returned error: %v", err)
	}
	if getResult["vector_id"] != "lifecycle-vec" {
		t.Errorf("expected lifecycle-vec, got %v", getResult["vector_id"])
	}

	// 4. Search vectors
	searchResult, err := client.Vectors.Search(ctx, SearchVectorsParams{
		QueryVector: []float64{0.1, 0.2, 0.3},
		K:           5,
	})
	if err != nil {
		t.Fatalf("step 4 - Search() returned error: %v", err)
	}
	results, ok := searchResult["results"].([]interface{})
	if !ok || len(results) == 0 {
		t.Fatal("step 4 - expected at least one search result")
	}

	// 5. Multimodal text ingest
	_, err = client.Multimodal.IngestText(ctx, "lifecycle-test", IngestTextParams{
		Text:     "Hello from lifecycle test",
		VectorID: "lifecycle-text",
	})
	if err != nil {
		t.Fatalf("step 5 - IngestText() returned error: %v", err)
	}

	// 6. Multimodal text search
	searchResult2, err := client.Multimodal.SearchText(ctx, "lifecycle-test", SearchTextParams{
		Query: "hello",
		K:     5,
	})
	if err != nil {
		t.Fatalf("step 6 - SearchText() returned error: %v", err)
	}
	if !searchResult2.Success {
		t.Error("step 6 - expected success=true")
	}

	// 7. RAG query
	ragResult, err := client.Multimodal.Query(ctx, "lifecycle-test", "What is the policy?", 5)
	if err != nil {
		t.Fatalf("step 7 - Query() returned error: %v", err)
	}
	if ragResult["answer"] == "" {
		t.Error("step 7 - expected non-empty answer")
	}

	// 8. Build index
	_, err = client.Collections.BuildIndex(ctx, "lifecycle-test", BuildIndexParams{
		Method: "hnsw",
	})
	if err != nil {
		t.Fatalf("step 8 - BuildIndex() returned error: %v", err)
	}

	// 9. Index stats
	stats, err := client.Collections.IndexStats(ctx, "lifecycle-test")
	if err != nil {
		t.Fatalf("step 9 - IndexStats() returned error: %v", err)
	}
	if stats["num_vectors"] != float64(42) {
		t.Errorf("step 9 - expected num_vectors 42, got %v", stats["num_vectors"])
	}

	// 10. Delete collection
	delResult, err := client.Collections.Delete(ctx, "lifecycle-test")
	if err != nil {
		t.Fatalf("step 10 - Delete() returned error: %v", err)
	}
	if msg, ok := delResult["message"].(string); !ok || !strings.Contains(msg, "lifecycle-test") {
		t.Errorf("step 10 - unexpected delete message: %v", delResult["message"])
	}
}

// Multimodal file-based operations (image/audio) with nil content should return errors
func TestIntegrationMultimodalIngestImageNilContent(t *testing.T) {
	server, client, ctx := setupTestServer(t)
	defer server.Close()

	_, err := client.Multimodal.IngestImage(ctx, "test-col", IngestFileParams{
		Contents: nil,
		Filename: "test.jpg",
	})
	if err == nil {
		t.Fatal("expected error for nil image content")
	}
	if !strings.Contains(err.Error(), "provide contents") {
		t.Errorf("expected 'provide contents' error, got: %v", err)
	}
}

func TestIntegrationMultimodalSearchImageNilContent(t *testing.T) {
	server, client, ctx := setupTestServer(t)
	defer server.Close()

	_, err := client.Multimodal.SearchImage(ctx, "test-col", SearchFileParams{
		Contents: nil,
		Filename: "test.jpg",
	})
	if err == nil {
		t.Fatal("expected error for nil image content")
	}
}

func TestIntegrationMultimodalIngestAudioNilContent(t *testing.T) {
	server, client, ctx := setupTestServer(t)
	defer server.Close()

	_, err := client.Multimodal.IngestAudio(ctx, "test-col", IngestFileParams{
		Contents: nil,
		Filename: "test.mp3",
	})
	if err == nil {
		t.Fatal("expected error for nil audio content")
	}
}

func TestIntegrationMultimodalSearchAudioNilContent(t *testing.T) {
	server, client, ctx := setupTestServer(t)
	defer server.Close()

	_, err := client.Multimodal.SearchAudio(ctx, "test-col", SearchFileParams{
		Contents: nil,
		Filename: "test.mp3",
	})
	if err == nil {
		t.Fatal("expected error for nil audio content")
	}
}
