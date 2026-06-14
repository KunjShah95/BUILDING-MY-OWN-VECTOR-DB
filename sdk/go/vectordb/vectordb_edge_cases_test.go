package vectordb

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// ---------------------------------------------------------------------------
// Edge case tests — network errors, malformed responses, timeouts, etc.
// ---------------------------------------------------------------------------

// --- Network errors --------------------------------------------------------

func TestEdgeNetworkErrorConnectionRefused(t *testing.T) {
	// Connect to a port where nothing is listening
	client := NewClient(ClientOptions{
		BaseURL: "http://127.0.0.1:1",
		Timeout: time.Second,
	})
	_, err := client.Health(context.Background())
	if err == nil {
		t.Fatal("expected error for connection refused")
	}
	var vdbErr *VectorDBError
	if !asVDBError(err, &vdbErr) {
		t.Fatalf("expected *VectorDBError, got %T: %v", err, err)
	}
	if !strings.Contains(err.Error(), "request failed") {
		t.Errorf("expected error mentioning 'request failed', got: %v", err)
	}
}

func TestEdgeNetworkErrorTimeout(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(2 * time.Second)
		_, _ = w.Write([]byte(`{"success":true}`))
	}))
	defer server.Close()

	client := NewClient(ClientOptions{
		BaseURL: server.URL,
		Timeout: 100 * time.Millisecond,
	})
	_, err := client.Health(context.Background())
	if err == nil {
		t.Fatal("expected timeout error")
	}
}

// --- Context cancellation --------------------------------------------------

func TestEdgeContextCancellation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
			// Request was cancelled, this is expected
			return
		case <-time.After(5 * time.Second):
			_, _ = w.Write([]byte(`{"success":true}`))
		}
	}))
	defer server.Close()

	client := NewClient(ClientOptions{
		BaseURL: server.URL,
		Timeout: 10 * time.Second,
	})

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	_, err := client.Health(ctx)
	if err == nil {
		t.Fatal("expected error for cancelled context")
	}
}

func TestEdgeContextDeadlineExceeded(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(2 * time.Second)
		_, _ = w.Write([]byte(`{"success":true}`))
	}))
	defer server.Close()

	client := NewClient(ClientOptions{
		BaseURL: server.URL,
		Timeout: 10 * time.Second,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	_, err := client.Health(ctx)
	if err == nil {
		t.Fatal("expected deadline exceeded error")
	}
}

// --- Malformed responses ---------------------------------------------------

func TestEdgeNonJSONResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("this is not json"))
	}))
	defer server.Close()

	client := NewClient(ClientOptions{BaseURL: server.URL})
	result, err := client.Health(context.Background())
	if err != nil {
		t.Fatalf("Health() should not error on non-JSON response: %v", err)
	}
	// Non-JSON body gets parsed as {"message": rawBody} by raiseForStatus
	if msg, ok := result["message"].(string); ok {
		if msg != "this is not json" {
			t.Errorf("expected message 'this is not json', got %q", msg)
		}
	}
}

func TestEdgeNonJSONErrorResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte("Service Unavailable"))
	}))
	defer server.Close()

	client := NewClient(ClientOptions{BaseURL: server.URL})
	_, err := client.Health(context.Background())
	if err == nil {
		t.Fatal("expected error for 503")
	}
	httpErr, ok := err.(*VectorDBHTTPError)
	if !ok {
		t.Fatalf("expected *VectorDBHTTPError, got %T: %v", err, err)
	}
	if httpErr.StatusCode != 503 {
		t.Errorf("expected 503, got %d", httpErr.StatusCode)
	}
	// The detail should be a map with message key since raiseForStatus sets {message: rawBody} for non-JSON
	detailMap, ok := httpErr.Detail.(map[string]interface{})
	if !ok {
		t.Fatalf("expected detail map, got %T: %v", httpErr.Detail, httpErr.Detail)
	}
	if detailMap["message"] != "Service Unavailable" {
		t.Errorf("expected detail.message 'Service Unavailable', got %v", detailMap["message"])
	}
}

func TestEdgeEmptyResponseBody(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		// No body
	}))
	defer server.Close()

	client := NewClient(ClientOptions{BaseURL: server.URL})
	result, err := client.Health(context.Background())
	if err != nil {
		t.Fatalf("Health() should not error on empty body: %v", err)
	}
	// Empty body -> json.Unmarshal fails, raiseForStatus sets {message: ""}
	// This is fine — the response still succeeds (status 200) and returns a valid map
	if result == nil {
		t.Fatal("expected non-nil result for empty body")
	}
	// The parsed map has fallback "message" key from non-JSON handling
	msg, ok := result["message"]
	if ok && msg != "" {
		t.Errorf("expected empty message, got %q", msg)
	}
}

func TestEdgeEmptyErrorBody(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		// Empty body with error status
	}))
	defer server.Close()

	client := NewClient(ClientOptions{BaseURL: server.URL})
	_, err := client.Health(context.Background())
	if err == nil {
		t.Fatal("expected error for 500")
	}
	httpErr, ok := err.(*VectorDBHTTPError)
	if !ok {
		t.Fatalf("expected *VectorDBHTTPError, got %T: %v", err, err)
	}
	if httpErr.StatusCode != 500 {
		t.Errorf("expected 500, got %d", httpErr.StatusCode)
	}
}

func TestEdgeResponseWithExtraFields(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"success":true,"status":"ok","extra_field":"unexpected","nested":{"a":1}}`))
	}))
	defer server.Close()

	client := NewClient(ClientOptions{BaseURL: server.URL})
	result, err := client.Health(context.Background())
	if err != nil {
		t.Fatalf("Health() returned error: %v", err)
	}
	if result["status"] != "ok" {
		t.Errorf("expected status 'ok', got %v", result["status"])
	}
	if result["extra_field"] != "unexpected" {
		t.Errorf("expected extra_field 'unexpected', got %v", result["extra_field"])
	}
}

func TestEdgeResponseMissingFields(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{}`))
	}))
	defer server.Close()

	client := NewClient(ClientOptions{BaseURL: server.URL})
	result, err := client.Health(context.Background())
	if err != nil {
		t.Fatalf("Health() returned error: %v", err)
	}
	if result == nil {
		t.Fatal("expected non-nil result")
	}
}

// --- HTTP error codes ------------------------------------------------------

func TestEdgeHTTPError400(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"detail":"Bad request"}`))
	}))
	defer server.Close()

	client := NewClient(ClientOptions{BaseURL: server.URL})
	_, err := client.Health(context.Background())
	expectHTTPError(t, err, 400)
}

func TestEdgeHTTPError401(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"detail":"Invalid API key"}`))
	}))
	defer server.Close()

	client := NewClient(ClientOptions{BaseURL: server.URL})
	_, err := client.Health(context.Background())
	expectHTTPError(t, err, 401)
}

func TestEdgeHTTPError403(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusForbidden)
		_, _ = w.Write([]byte(`{"detail":"Forbidden"}`))
	}))
	defer server.Close()

	client := NewClient(ClientOptions{BaseURL: server.URL})
	_, err := client.Health(context.Background())
	expectHTTPError(t, err, 403)
}

func TestEdgeHTTPError404(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte(`{"detail":"Not Found"}`))
	}))
	defer server.Close()

	client := NewClient(ClientOptions{BaseURL: server.URL})
	_, err := client.Health(context.Background())
	expectHTTPError(t, err, 404)
}

func TestEdgeHTTPError405(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusMethodNotAllowed)
		_, _ = w.Write([]byte(`{"detail":"Method not allowed"}`))
	}))
	defer server.Close()

	client := NewClient(ClientOptions{BaseURL: server.URL})
	_, err := client.Health(context.Background())
	expectHTTPError(t, err, 405)
}

func TestEdgeHTTPError409(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusConflict)
		_, _ = w.Write([]byte(`{"detail":"Conflict"}`))
	}))
	defer server.Close()

	client := NewClient(ClientOptions{BaseURL: server.URL})
	_, err := client.Health(context.Background())
	expectHTTPError(t, err, 409)
}

func TestEdgeHTTPError422(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnprocessableEntity)
		_, _ = w.Write([]byte(`{"detail":[{"loc":["body","vector"],"msg":"field required"}]}`))
	}))
	defer server.Close()

	client := NewClient(ClientOptions{BaseURL: server.URL})
	_, err := client.Health(context.Background())
	expectHTTPError(t, err, 422)
}

func TestEdgeHTTPError429(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"detail":"Rate limit exceeded"}`))
	}))
	defer server.Close()

	client := NewClient(ClientOptions{BaseURL: server.URL})
	_, err := client.Health(context.Background())
	expectHTTPError(t, err, 429)
}

func TestEdgeHTTPError502(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadGateway)
		_, _ = w.Write([]byte(`{"detail":"Bad Gateway"}`))
	}))
	defer server.Close()

	client := NewClient(ClientOptions{BaseURL: server.URL})
	_, err := client.Health(context.Background())
	expectHTTPError(t, err, 502)
}

func TestEdgeHTTPError503(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`{"detail":"Service Unavailable"}`))
	}))
	defer server.Close()

	client := NewClient(ClientOptions{BaseURL: server.URL})
	_, err := client.Health(context.Background())
	expectHTTPError(t, err, 503)
}

func TestEdgeHTTPErrorDetail(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"detail":{"reason":"validation_error","field":"dimension","message":"must be positive"}}`))
	}))
	defer server.Close()

	client := NewClient(ClientOptions{BaseURL: server.URL})
	_, err := client.Health(context.Background())
	if err == nil {
		t.Fatal("expected error")
	}
	httpErr, ok := err.(*VectorDBHTTPError)
	if !ok {
		t.Fatalf("expected *VectorDBHTTPError, got %T: %v", err, err)
	}
	if httpErr.StatusCode != 400 {
		t.Errorf("expected 400, got %d", httpErr.StatusCode)
	}
	detail, ok := httpErr.Detail.(map[string]interface{})
	if !ok {
		t.Fatalf("expected detail map, got %T: %v", httpErr.Detail, httpErr.Detail)
	}
	if detail["reason"] != "validation_error" {
		t.Errorf("expected reason 'validation_error', got %v", detail["reason"])
	}
}

// --- Header propagation ----------------------------------------------------

func TestEdgeClientHeadersPropagated(t *testing.T) {
	var recordedHeaders http.Header
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		recordedHeaders = r.Header
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"success":true}`))
	}))
	defer server.Close()

	client := NewClient(ClientOptions{
		BaseURL: server.URL,
		Headers: map[string]string{
			"Authorization": "Bearer client-token",
			"X-API-Key":     "client-key",
		},
	})

	_, err := client.Health(context.Background())
	if err != nil {
		t.Fatalf("Health() returned error: %v", err)
	}

	if recordedHeaders.Get("Authorization") != "Bearer client-token" {
		t.Errorf("expected Authorization 'Bearer client-token', got %q", recordedHeaders.Get("Authorization"))
	}
	if recordedHeaders.Get("X-API-Key") != "client-key" {
		t.Errorf("expected X-API-Key 'client-key', got %q", recordedHeaders.Get("X-API-Key"))
	}
}

func TestEdgePerRequestHeadersOverrideClientHeaders(t *testing.T) {
	var recordedHeaders http.Header
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		recordedHeaders = r.Header
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"success":true}`))
	}))
	defer server.Close()

	client := NewClient(ClientOptions{
		BaseURL: server.URL,
		Headers: map[string]string{
			"Authorization": "Bearer old-token",
			"X-API-Key":     "old-key",
		},
	})

	// Per-request override
	_, err := client.Health(context.Background(),
		WithHeader("Authorization", "Bearer new-token"),
	)
	if err != nil {
		t.Fatalf("Health() returned error: %v", err)
	}

	if recordedHeaders.Get("Authorization") != "Bearer new-token" {
		t.Errorf("expected Authorization 'Bearer new-token', got %q", recordedHeaders.Get("Authorization"))
	}
	if recordedHeaders.Get("X-API-Key") != "old-key" {
		t.Errorf("expected X-API-Key 'old-key' (should not be overridden), got %q", recordedHeaders.Get("X-API-Key"))
	}
}

func TestEdgeNoClientHeaders(t *testing.T) {
	var recordedHeaders http.Header
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		recordedHeaders = r.Header
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"success":true}`))
	}))
	defer server.Close()

	client := NewClient(ClientOptions{
		BaseURL: server.URL,
	})

	_, err := client.Health(context.Background())
	if err != nil {
		t.Fatalf("Health() returned error: %v", err)
	}
	// Should have Content-Type but no custom auth
	_ = recordedHeaders.Get("Content-Type")
	if recordedHeaders.Get("Authorization") != "" {
		t.Errorf("expected no Authorization header, got %q", recordedHeaders.Get("Authorization"))
	}
}

// --- Query parameters ------------------------------------------------------

func TestEdgeQueryParametersPropagated(t *testing.T) {
	var recordedQuery string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		recordedQuery = r.URL.RawQuery
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"success":true}`))
	}))
	defer server.Close()

	client := NewClient(ClientOptions{BaseURL: server.URL})
	_, err := client.Collections.List(context.Background(), 50, 10)
	if err != nil {
		t.Fatalf("List() returned error: %v", err)
	}
	if !strings.Contains(recordedQuery, "limit=50") {
		t.Errorf("expected query to contain 'limit=50', got %q", recordedQuery)
	}
	if !strings.Contains(recordedQuery, "offset=10") {
		t.Errorf("expected query to contain 'offset=10', got %q", recordedQuery)
	}
}

func TestEdgeCustomQueryParams(t *testing.T) {
	var recordedQuery string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		recordedQuery = r.URL.RawQuery
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"success":true}`))
	}))
	defer server.Close()

	client := NewClient(ClientOptions{BaseURL: server.URL})
	_, err := jsonGet(context.Background(), client.httpClient, client.BaseURL, "/health", &requestOptions{
		Params: map[string]string{
			"custom": "value",
		},
	})
	if err != nil {
		t.Fatalf("jsonGet() returned error: %v", err)
	}
	if !strings.Contains(recordedQuery, "custom=value") {
		t.Errorf("expected query to contain 'custom=value', got %q", recordedQuery)
	}
}

// --- Client initialization -------------------------------------------------

func TestEdgeDefaultClientOptions(t *testing.T) {
	client := NewClient()
	if client.BaseURL != "http://localhost:8000" {
		t.Errorf("expected default BaseURL 'http://localhost:8000', got %q", client.BaseURL)
	}
	if client.Collections == nil {
		t.Error("expected CollectionsAPI to be initialized")
	}
	if client.Vectors == nil {
		t.Error("expected VectorsAPI to be initialized")
	}
	if client.Multimodal == nil {
		t.Error("expected MultimodalAPI to be initialized")
	}
}

func TestEdgeCustomBaseURL(t *testing.T) {
	client := NewClient(ClientOptions{
		BaseURL: "https://api.example.com/vectordb",
	})
	if client.BaseURL != "https://api.example.com/vectordb" {
		t.Errorf("expected BaseURL 'https://api.example.com/vectordb', got %q", client.BaseURL)
	}
}

func TestEdgeTrailingSlashRemoved(t *testing.T) {
	client := NewClient(ClientOptions{
		BaseURL: "http://localhost:8000/",
	})
	if client.BaseURL != "http://localhost:8000" {
		t.Errorf("expected BaseURL without trailing slash, got %q", client.BaseURL)
	}
}

// --- raiseForStatus unit tests ---------------------------------------------

func TestEdgeRaiseForStatusSuccess(t *testing.T) {
	res := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(`{"success":true}`)),
		Header:     http.Header{"Content-Type": []string{"application/json"}},
	}
	rr, err := raiseForStatus(res)
	if err != nil {
		t.Fatalf("raiseForStatus() returned error: %v", err)
	}
	if rr.parsed["success"] != true {
		t.Errorf("expected success=true, got %v", rr.parsed["success"])
	}
}

func TestEdgeRaiseForStatusError(t *testing.T) {
	res := &http.Response{
		StatusCode: http.StatusBadRequest,
		Body:       io.NopCloser(strings.NewReader(`{"detail":"bad data"}`)),
		Header:     http.Header{"Content-Type": []string{"application/json"}},
	}
	_, err := raiseForStatus(res)
	if err == nil {
		t.Fatal("expected error")
	}
	httpErr, ok := err.(*VectorDBHTTPError)
	if !ok {
		t.Fatalf("expected *VectorDBHTTPError, got %T: %v", err, err)
	}
	if httpErr.StatusCode != 400 {
		t.Errorf("expected 400, got %d", httpErr.StatusCode)
	}
}

func TestEdgeRaiseForStatusNonJSONError(t *testing.T) {
	res := &http.Response{
		StatusCode: http.StatusInternalServerError,
		Body:       io.NopCloser(strings.NewReader("Internal Error")),
		Header:     http.Header{"Content-Type": []string{"text/plain"}},
	}
	_, err := raiseForStatus(res)
	if err == nil {
		t.Fatal("expected error")
	}
	httpErr, ok := err.(*VectorDBHTTPError)
	if !ok {
		t.Fatalf("expected *VectorDBHTTPError, got %T: %v", err, err)
	}
	if httpErr.StatusCode != 500 {
		t.Errorf("expected 500, got %d", httpErr.StatusCode)
	}
}

func TestEdgeRaiseForStatusNoDetail(t *testing.T) {
	res := &http.Response{
		StatusCode: http.StatusNotFound,
		Body:       io.NopCloser(strings.NewReader(`{"error":"not found"}`)),
		Header:     http.Header{"Content-Type": []string{"application/json"}},
	}
	_, err := raiseForStatus(res)
	if err == nil {
		t.Fatal("expected error")
	}
	httpErr, ok := err.(*VectorDBHTTPError)
	if !ok {
		t.Fatalf("expected *VectorDBHTTPError, got %T: %v", err, err)
	}
	// Since there's no "detail" key, the whole parsed body is used
	if httpErr.Detail == nil {
		t.Fatal("expected non-nil detail")
	}
}

// --- buildURL unit tests ---------------------------------------------------

func TestEdgeBuildURLBasic(t *testing.T) {
	u := buildURL("http://localhost:8000", "/health", nil)
	if u != "http://localhost:8000/health" {
		t.Errorf("expected 'http://localhost:8000/health', got %q", u)
	}
}

func TestEdgeBuildURLWithParams(t *testing.T) {
	u := buildURL("http://localhost:8000", "/search", &requestOptions{
		Params: map[string]string{
			"k":  "10",
			"q":  "hello",
		},
	})
	if !strings.Contains(u, "k=10") {
		t.Errorf("expected query to contain 'k=10', got %q", u)
	}
	if !strings.Contains(u, "q=hello") {
		t.Errorf("expected query to contain 'q=hello', got %q", u)
	}
}

func TestEdgeBuildURLNoLeadingSlash(t *testing.T) {
	u := buildURL("http://localhost:8000", "health", nil)
	if u != "http://localhost:8000/health" {
		t.Errorf("expected 'http://localhost:8000/health', got %q", u)
	}
}

func TestEdgeBuildURLTrailingBase(t *testing.T) {
	u := buildURL("http://localhost:8000/api/", "/health", nil)
	if u != "http://localhost:8000/api/health" {
		t.Errorf("expected 'http://localhost:8000/api/health', got %q", u)
	}
}

// --- Error type helpers ----------------------------------------------------

func TestEdgeErrorTypeAssertion(t *testing.T) {
	err := &VectorDBHTTPError{
		VectorDBError: VectorDBError{msg: "test error"},
		StatusCode:    500,
		Detail:        "test detail",
	}

	var vdbErr *VectorDBError
	if !asVDBError(err, &vdbErr) {
		t.Fatal("expected VectorDBHTTPError to match *VectorDBError")
	}
	if vdbErr.Error() != "test error" {
		t.Errorf("expected 'test error', got %q", vdbErr.Error())
	}

	var httpErr *VectorDBHTTPError
	if !asVDBError(err, &httpErr) {
		t.Fatal("expected to match *VectorDBHTTPError")
	}
	if httpErr.StatusCode != 500 {
		t.Errorf("expected 500, got %d", httpErr.StatusCode)
	}
}

// Test that all request methods propagate signal/context cancellation
func TestEdgeAllMethodsPropagateCancellation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
		case <-time.After(5 * time.Second):
			_, _ = w.Write([]byte(`{"success":true}`))
		}
	}))
	defer server.Close()

	client := NewClient(ClientOptions{BaseURL: server.URL})

	cancelCtx, cancel := context.WithCancel(context.Background())
	cancel()

	// Test GET
	_, err := jsonGet(cancelCtx, client.httpClient, client.BaseURL, "/health", nil)
	if err == nil {
		t.Error("expected error for GET with cancelled context")
	}

	// Test POST
	_, err = jsonPost(cancelCtx, client.httpClient, client.BaseURL, "/health", map[string]interface{}{"test": true}, nil)
	if err == nil {
		t.Error("expected error for POST with cancelled context")
	}

	// Test DELETE
	_, err = jsonDelete(cancelCtx, client.httpClient, client.BaseURL, "/health", nil)
	if err == nil {
		t.Error("expected error for DELETE with cancelled context")
	}
}

// --- Helpers ---------------------------------------------------------------

func expectHTTPError(t *testing.T, err error, expectedCode int) {
	t.Helper()
	if err == nil {
		t.Fatalf("expected HTTP %d error", expectedCode)
	}
	httpErr, ok := err.(*VectorDBHTTPError)
	if !ok {
		t.Fatalf("expected *VectorDBHTTPError, got %T: %v", err, err)
	}
	if httpErr.StatusCode != expectedCode {
		t.Errorf("expected status %d, got %d", expectedCode, httpErr.StatusCode)
	}
}

// asVDBError attempts to unwrap err into the given target pointer
// using a type switch. Returns true on success.
func asVDBError(err error, target interface{}) bool {
	switch e := err.(type) {
	case *VectorDBHTTPError:
		switch t := target.(type) {
		case **VectorDBHTTPError:
			*t = e
			return true
		case **VectorDBError:
			*t = &e.VectorDBError
			return true
		}
	case *VectorDBError:
		switch t := target.(type) {
		case **VectorDBError:
			*t = e
			return true
		}
	}
	return false
}

