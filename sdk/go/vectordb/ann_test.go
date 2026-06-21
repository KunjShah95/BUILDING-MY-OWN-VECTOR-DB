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

func setupAnnTestServer(t *testing.T) (*httptest.Server, *VectorDBClient, context.Context) {
	t.Helper()

	mux := http.NewServeMux()

	// create_index POST /api/v1/ann/index
	mux.HandleFunc("/api/v1/ann/index", func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			w.Header().Set("Content-Type", "application/json")
			idxType := r.URL.Query().Get("index_type")
			if idxType != "" {
				_, _ = w.Write([]byte(`{"success":true,"index_type":"hnsw"}`))
			} else {
				_, _ = w.Write([]byte(`{"success":true,"indexes":{"hnsw":{}}}`))
			}
		case http.MethodPost:
			w.Header().Set("Content-Type", "application/json")
			var body map[string]interface{}
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				http.Error(w, `{"detail":"bad request"}`, http.StatusBadRequest)
				return
			}
			idxType, _ := body["index_type"].(string)
			_, _ = w.Write([]byte(fmt.Sprintf(`{"success":true,"message":"Created %s index"}`, idxType)))
		default:
			http.Error(w, `{"detail":"method not allowed"}`, http.StatusMethodNotAllowed)
		}
	})

	// save_index POST /api/v1/ann/index/save
	mux.HandleFunc("/api/v1/ann/index/save", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		idxType := r.URL.Query().Get("index_type")
		_, _ = w.Write([]byte(fmt.Sprintf(`{"success":true,"message":"Saved %s index"}`, idxType)))
	})

	// load_index POST /api/v1/ann/index/load
	mux.HandleFunc("/api/v1/ann/index/load", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		idxType := r.URL.Query().Get("index_type")
		_, _ = w.Write([]byte(fmt.Sprintf(`{"success":true,"message":"Loaded %s index"}`, idxType)))
	})

	// compare_search GET /api/v1/ann/search/compare
	mux.HandleFunc("/api/v1/ann/search/compare", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		query := r.URL.Query().Get("query_vector")
		k := r.URL.Query().Get("k")
		_, _ = w.Write([]byte(fmt.Sprintf(`{"success":true,"query_vector":"%s","k":%s}`, query, k)))
	})

	// stats GET /api/v1/ann/stats
	mux.HandleFunc("/api/v1/ann/stats", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"success":true,"total_vectors_in_db":100}`))
	})

	server := httptest.NewServer(mux)

	client := NewClient(ClientOptions{
		BaseURL: server.URL,
		Timeout: 5 * time.Second,
	})

	return server, client, context.Background()
}

func TestAnnCreateIndex(t *testing.T) {
	server, client, ctx := setupAnnTestServer(t)
	defer server.Close()

	res, err := client.Ann.CreateIndex(ctx, CreateIndexParams{
		IndexType: "hnsw",
		M:         16,
	})
	if err != nil {
		t.Fatalf("CreateIndex() failed: %v", err)
	}
	if res["success"] != true {
		t.Errorf("expected success true, got %v", res["success"])
	}
	if !strings.Contains(res["message"].(string), "hnsw") {
		t.Errorf("expected message to contain hnsw, got %v", res["message"])
	}
}

func TestAnnGetIndexInfo(t *testing.T) {
	server, client, ctx := setupAnnTestServer(t)
	defer server.Close()

	res, err := client.Ann.GetIndexInfo(ctx, "hnsw")
	if err != nil {
		t.Fatalf("GetIndexInfo() failed: %v", err)
	}
	if res["index_type"] != "hnsw" {
		t.Errorf("expected index_type hnsw, got %v", res["index_type"])
	}
}

func TestAnnSaveIndex(t *testing.T) {
	server, client, ctx := setupAnnTestServer(t)
	defer server.Close()

	res, err := client.Ann.SaveIndex(ctx, "ivf")
	if err != nil {
		t.Fatalf("SaveIndex() failed: %v", err)
	}
	if !strings.Contains(res["message"].(string), "ivf") {
		t.Errorf("expected message to contain ivf, got %v", res["message"])
	}
}

func TestAnnLoadIndex(t *testing.T) {
	server, client, ctx := setupAnnTestServer(t)
	defer server.Close()

	res, err := client.Ann.LoadIndex(ctx, "brute")
	if err != nil {
		t.Fatalf("LoadIndex() failed: %v", err)
	}
	if !strings.Contains(res["message"].(string), "brute") {
		t.Errorf("expected message to contain brute, got %v", res["message"])
	}
}

func TestAnnCompareSearch(t *testing.T) {
	server, client, ctx := setupAnnTestServer(t)
	defer server.Close()

	res, err := client.Ann.CompareSearch(ctx, CompareSearchParams{
		QueryVector: []float64{0.1, 0.2, 0.3},
		K:           5,
	})
	if err != nil {
		t.Fatalf("CompareSearch() failed: %v", err)
	}
	if res["query_vector"] != "0.1,0.2,0.3" {
		t.Errorf("expected query_vector 0.1,0.2,0.3, got %v", res["query_vector"])
	}
}

func TestAnnGetStatistics(t *testing.T) {
	server, client, ctx := setupAnnTestServer(t)
	defer server.Close()

	res, err := client.Ann.GetStatistics(ctx)
	if err != nil {
		t.Fatalf("GetStatistics() failed: %v", err)
	}
	if res["total_vectors_in_db"] != float64(100) {
		t.Errorf("expected total_vectors_in_db 100, got %v", res["total_vectors_in_db"])
	}
}
