use reqwest::blocking::{Client, multipart};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION};
use std::fmt;
use std::sync::Arc;
use serde_json::json;

use crate::models::*;

// =========================================================================
// Error Definitions
// =========================================================================

#[derive(Debug, Clone)]
pub struct VectorDBHTTPError {
    pub status_code: u16,
    pub detail: serde_json::Value,
}

#[derive(Debug, Clone)]
pub enum VectorDBError {
    HTTPError(VectorDBHTTPError),
    NetworkError(String),
    SerializationError(String),
    ValidationError(String),
}

impl fmt::Display for VectorDBHTTPError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HTTP {}: {}", self.status_code, self.detail)
    }
}

impl fmt::Display for VectorDBError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VectorDBError::HTTPError(err) => write!(f, "HTTP Error: {}", err),
            VectorDBError::NetworkError(msg) => write!(f, "Network Error: {}", msg),
            VectorDBError::SerializationError(msg) => write!(f, "Serialization Error: {}", msg),
            VectorDBError::ValidationError(msg) => write!(f, "Validation Error: {}", msg),
        }
    }
}

impl std::error::Error for VectorDBError {}

pub type Result<T> = std::result::Result<T, VectorDBError>;

// =========================================================================
// Inner Client Info
// =========================================================================

pub struct ClientInfo {
    pub client: Client,
    pub base_url: String,
}

// Helper function to handle standard HTTP calls
fn request<T: serde::de::DeserializeOwned>(
    client: &Client,
    method: reqwest::Method,
    url: &str,
    body: Option<serde_json::Value>,
    query: Option<&[(&str, &str)]>,
) -> Result<T> {
    let mut req = client.request(method.clone(), url);
    if let Some(b) = body {
        req = req.json(&b);
    }
    if let Some(q) = query {
        req = req.query(q);
    }

    let res = req.send().map_err(|e| VectorDBError::NetworkError(e.to_string()))?;
    let status = res.status();
    let body_text = res.text().map_err(|e| VectorDBError::NetworkError(e.to_string()))?;

    if !status.is_success() {
        let detail: serde_json::Value = serde_json::from_str(&body_text)
            .unwrap_or_else(|_| json!({ "message": body_text }));
        return Err(VectorDBError::HTTPError(VectorDBHTTPError {
            status_code: status.as_u16(),
            detail: detail.get("detail").cloned().unwrap_or(detail),
        }));
    }

    serde_json::from_str(&body_text).map_err(|e| VectorDBError::SerializationError(e.to_string()))
}

// =========================================================================
// Main Client
// =========================================================================

pub struct VectorDBClient {
    pub collections: CollectionsAPI,
    pub vectors: VectorsAPI,
    pub multimodal: MultimodalAPI,
    pub ann: AnnAPI,
    info: Arc<ClientInfo>,
}

impl VectorDBClient {
    pub fn new(base_url: &str, api_key: Option<&str>, tenant_id: Option<&str>) -> Self {
        let mut headers = HeaderMap::new();
        if let Some(key) = api_key {
            let val = HeaderValue::from_str(&format!("Bearer {}", key)).unwrap();
            headers.insert(AUTHORIZATION, val);
        }
        if let Some(t_id) = tenant_id {
            let val = HeaderValue::from_str(t_id).unwrap();
            headers.insert("X-Tenant-ID", val);
        }

        let client = Client::builder()
            .default_headers(headers)
            .build()
            .unwrap();

        let info = Arc::new(ClientInfo {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
        });

        VectorDBClient {
            collections: CollectionsAPI::new(info.clone()),
            vectors: VectorsAPI::new(info.clone()),
            multimodal: MultimodalAPI::new(info.clone()),
            ann: AnnAPI::new(info.clone()),
            info,
        }
    }

    pub fn health(&self) -> Result<serde_json::Value> {
        let url = format!("{}/health", self.info.base_url);
        request(&self.info.client, reqwest::Method::GET, &url, None, None)
    }
}

// =========================================================================
// Collections API
// =========================================================================

pub struct CollectionsAPI {
    info: Arc<ClientInfo>,
}

impl CollectionsAPI {
    pub fn new(info: Arc<ClientInfo>) -> Self {
        CollectionsAPI { info }
    }

    pub fn create(
        &self,
        name: &str,
        collection_id: Option<&str>,
        modality: Option<&str>,
        dimension: Option<i32>,
        embedding_model: Option<&str>,
        distance_metric: Option<&str>,
        description: Option<&str>,
    ) -> Result<Collection> {
        let url = format!("{}/collections", self.info.base_url);
        let mut body = json!({
            "name": name,
            "modality": modality.unwrap_or("text"),
            "distance_metric": distance_metric.unwrap_or("cosine"),
        });
        if let Some(id) = collection_id {
            body["collection_id"] = json!(id);
        }
        if let Some(dim) = dimension {
            body["dimension"] = json!(dim);
        }
        if let Some(model) = embedding_model {
            body["embedding_model"] = json!(model);
        }
        if let Some(desc) = description {
            body["description"] = json!(desc);
        }

        let envelope: CollectionEnvelope = request(
            &self.info.client,
            reqwest::Method::POST,
            &url,
            Some(body),
            None,
        )?;
        Ok(envelope.collection)
    }

    pub fn list(&self, limit: Option<i32>, offset: Option<i32>) -> Result<Vec<Collection>> {
        let url = format!("{}/collections", self.info.base_url);
        let limit_str = limit.unwrap_or(100).to_string();
        let offset_str = offset.unwrap_or(0).to_string();
        let query = [
            ("limit", limit_str.as_str()),
            ("offset", offset_str.as_str()),
        ];
        let envelope: CollectionsListEnvelope = request(
            &self.info.client,
            reqwest::Method::GET,
            &url,
            None,
            Some(&query),
        )?;
        Ok(envelope.collections)
    }

    pub fn get(&self, collection_id: &str) -> Result<Collection> {
        let url = format!("{}/collections/{}", self.info.base_url, percent_encoding::utf8_percent_encode(collection_id, percent_encoding::NON_ALPHANUMERIC));
        let envelope: CollectionEnvelope = request(
            &self.info.client,
            reqwest::Method::GET,
            &url,
            None,
            None,
        )?;
        Ok(envelope.collection)
    }

    pub fn delete(&self, collection_id: &str) -> Result<serde_json::Value> {
        let url = format!("{}/collections/{}", self.info.base_url, percent_encoding::utf8_percent_encode(collection_id, percent_encoding::NON_ALPHANUMERIC));
        request(&self.info.client, reqwest::Method::DELETE, &url, None, None)
    }

    pub fn build_index(
        &self,
        collection_id: &str,
        method: Option<&str>,
        m: Option<i32>,
        m0: Option<i32>,
        ef_construction: Option<i32>,
        n_clusters: Option<i32>,
        n_probes: Option<i32>,
    ) -> Result<serde_json::Value> {
        let url = format!("{}/collections/{}/index", self.info.base_url, percent_encoding::utf8_percent_encode(collection_id, percent_encoding::NON_ALPHANUMERIC));
        let mut body = json!({
            "method": method.unwrap_or("hnsw"),
            "m": m.unwrap_or(16),
            "ef_construction": ef_construction.unwrap_or(200),
            "n_clusters": n_clusters.unwrap_or(100),
            "n_probes": n_probes.unwrap_or(10),
        });
        if let Some(val) = m0 {
            body["m0"] = json!(val);
        }

        request(&self.info.client, reqwest::Method::POST, &url, Some(body), None)
    }

    pub fn index_stats(&self, collection_id: &str) -> Result<serde_json::Value> {
        let url = format!("{}/collections/{}/index/stats", self.info.base_url, percent_encoding::utf8_percent_encode(collection_id, percent_encoding::NON_ALPHANUMERIC));
        request(&self.info.client, reqwest::Method::GET, &url, None, None)
    }
}

// =========================================================================
// Vectors API
// =========================================================================

pub struct VectorsAPI {
    info: Arc<ClientInfo>,
}

impl VectorsAPI {
    pub fn new(info: Arc<ClientInfo>) -> Self {
        VectorsAPI { info }
    }

    pub fn create(
        &self,
        vector: Vec<f64>,
        metadata: Option<serde_json::Value>,
        vector_id: Option<&str>,
    ) -> Result<serde_json::Value> {
        let url = format!("{}/vectors", self.info.base_url);
        let mut body = json!({
            "vector": vector,
        });
        if let Some(meta) = metadata {
            body["metadata"] = meta;
        }
        if let Some(id) = vector_id {
            body["vector_id"] = json!(id);
        }

        request(&self.info.client, reqwest::Method::POST, &url, Some(body), None)
    }

    pub fn get(&self, vector_id: &str) -> Result<serde_json::Value> {
        let url = format!("{}/vectors/{}", self.info.base_url, percent_encoding::utf8_percent_encode(vector_id, percent_encoding::NON_ALPHANUMERIC));
        request(&self.info.client, reqwest::Method::GET, &url, None, None)
    }

    pub fn delete(&self, vector_id: &str) -> Result<serde_json::Value> {
        let url = format!("{}/vectors/{}", self.info.base_url, percent_encoding::utf8_percent_encode(vector_id, percent_encoding::NON_ALPHANUMERIC));
        request(&self.info.client, reqwest::Method::DELETE, &url, None, None)
    }

    pub fn search(
        &self,
        query_vector: Vec<f64>,
        k: Option<i32>,
        method: Option<&str>,
        filters: Option<serde_json::Value>,
    ) -> Result<serde_json::Value> {
        let url = format!("{}/search", self.info.base_url);
        let mut body = json!({
            "query_vector": query_vector,
            "k": k.unwrap_or(5),
            "method": method.unwrap_or("hnsw"),
        });
        if let Some(f) = filters {
            body["filters"] = f;
        }

        request(&self.info.client, reqwest::Method::POST, &url, Some(body), None)
    }
}

// =========================================================================
// Multimodal API
// =========================================================================

pub struct MultimodalAPI {
    info: Arc<ClientInfo>,
}

impl MultimodalAPI {
    pub fn new(info: Arc<ClientInfo>) -> Self {
        MultimodalAPI { info }
    }

    pub fn ingest_text(
        &self,
        collection_id: &str,
        text: &str,
        metadata: Option<serde_json::Value>,
        vector_id: Option<&str>,
    ) -> Result<serde_json::Value> {
        let url = format!("{}/collections/{}/ingest/text", self.info.base_url, percent_encoding::utf8_percent_encode(collection_id, percent_encoding::NON_ALPHANUMERIC));
        let mut body = json!({
            "text": text,
        });
        if let Some(meta) = metadata {
            body["metadata"] = meta;
        }
        if let Some(id) = vector_id {
            body["vector_id"] = json!(id);
        }

        request(&self.info.client, reqwest::Method::POST, &url, Some(body), None)
    }

    pub fn search_text(
        &self,
        collection_id: &str,
        query: &str,
        k: Option<i32>,
        method: Option<&str>,
        filters: Option<serde_json::Value>,
    ) -> Result<SearchResult> {
        let url = format!("{}/collections/{}/search/text", self.info.base_url, percent_encoding::utf8_percent_encode(collection_id, percent_encoding::NON_ALPHANUMERIC));
        let mut body = json!({
            "query": query,
            "k": k.unwrap_or(5),
            "method": method.unwrap_or("brute"),
        });
        if let Some(f) = filters {
            body["filters"] = f;
        }

        request(&self.info.client, reqwest::Method::POST, &url, Some(body), None)
    }

    fn multipart_file_request(
        &self,
        url: &str,
        file_bytes: Vec<u8>,
        filename: &str,
        mime_type: &str,
        metadata: Option<serde_json::Value>,
        vector_id: Option<&str>,
        k: Option<i32>,
        method: Option<&str>,
        filters: Option<serde_json::Value>,
    ) -> Result<serde_json::Value> {
        let mut form = multipart::Form::new()
            .part("file", multipart::Part::bytes(file_bytes)
                .file_name(filename.to_string())
                .mime_str(mime_type)
                .unwrap());

        if let Some(m) = metadata {
            form = form.text("metadata", m.to_string());
        }
        if let Some(vid) = vector_id {
            form = form.text("vector_id", vid.to_string());
        }
        if let Some(val) = k {
            form = form.text("k", val.to_string());
        }
        if let Some(mth) = method {
            form = form.text("method", mth.to_string());
        }
        if let Some(flt) = filters {
            form = form.text("filters", flt.to_string());
        }

        let res = self.info.client.post(url)
            .multipart(form)
            .send()
            .map_err(|e| VectorDBError::NetworkError(e.to_string()))?;

        let status = res.status();
        let body_text = res.text().map_err(|e| VectorDBError::NetworkError(e.to_string()))?;

        if !status.is_success() {
            let detail: serde_json::Value = serde_json::from_str(&body_text)
                .unwrap_or_else(|_| json!({ "message": body_text }));
            return Err(VectorDBError::HTTPError(VectorDBHTTPError {
                status_code: status.as_u16(),
                detail: detail.get("detail").cloned().unwrap_or(detail),
            }));
        }

        serde_json::from_str(&body_text).map_err(|e| VectorDBError::SerializationError(e.to_string()))
    }

    pub fn ingest_image(
        &self,
        collection_id: &str,
        image_bytes: Vec<u8>,
        filename: &str,
        metadata: Option<serde_json::Value>,
        vector_id: Option<&str>,
    ) -> Result<serde_json::Value> {
        let url = format!("{}/collections/{}/ingest/image", self.info.base_url, percent_encoding::utf8_percent_encode(collection_id, percent_encoding::NON_ALPHANUMERIC));
        self.multipart_file_request(&url, image_bytes, filename, "image/jpeg", metadata, vector_id, None, None, None)
    }

    pub fn search_image(
        &self,
        collection_id: &str,
        image_bytes: Vec<u8>,
        filename: &str,
        k: Option<i32>,
        method: Option<&str>,
        filters: Option<serde_json::Value>,
    ) -> Result<SearchResult> {
        let url = format!("{}/collections/{}/search/image", self.info.base_url, percent_encoding::utf8_percent_encode(collection_id, percent_encoding::NON_ALPHANUMERIC));
        let payload = self.multipart_file_request(&url, image_bytes, filename, "image/jpeg", None, None, k, method, filters)?;
        serde_json::from_value(payload).map_err(|e| VectorDBError::SerializationError(e.to_string()))
    }

    pub fn ingest_audio(
        &self,
        collection_id: &str,
        audio_bytes: Vec<u8>,
        filename: &str,
        metadata: Option<serde_json::Value>,
        vector_id: Option<&str>,
    ) -> Result<serde_json::Value> {
        let url = format!("{}/collections/{}/ingest/audio", self.info.base_url, percent_encoding::utf8_percent_encode(collection_id, percent_encoding::NON_ALPHANUMERIC));
        self.multipart_file_request(&url, audio_bytes, filename, "audio/mpeg", metadata, vector_id, None, None, None)
    }

    pub fn search_audio(
        &self,
        collection_id: &str,
        audio_bytes: Vec<u8>,
        filename: &str,
        k: Option<i32>,
        method: Option<&str>,
        filters: Option<serde_json::Value>,
    ) -> Result<SearchResult> {
        let url = format!("{}/collections/{}/search/audio", self.info.base_url, percent_encoding::utf8_percent_encode(collection_id, percent_encoding::NON_ALPHANUMERIC));
        let payload = self.multipart_file_request(&url, audio_bytes, filename, "audio/mpeg", None, None, k, method, filters)?;
        serde_json::from_value(payload).map_err(|e| VectorDBError::SerializationError(e.to_string()))
    }

    pub fn query(&self, collection_id: &str, query: &str, k: Option<i32>) -> Result<serde_json::Value> {
        let url = format!("{}/collections/{}/query", self.info.base_url, percent_encoding::utf8_percent_encode(collection_id, percent_encoding::NON_ALPHANUMERIC));
        let body = json!({
            "query": query,
            "k": k.unwrap_or(5),
        });
        request(&self.info.client, reqwest::Method::POST, &url, Some(body), None)
    }
}

// =========================================================================
// ANN Index Management API
// =========================================================================

pub struct AnnAPI {
    info: Arc<ClientInfo>,
}

impl AnnAPI {
    pub fn new(info: Arc<ClientInfo>) -> Self {
        AnnAPI { info }
    }

    pub fn create_index(
        &self,
        index_type: &str,
        metric: Option<&str>,
        m: Option<i32>,
        m0: Option<i32>,
        ef_construction: Option<i32>,
        n_clusters: Option<i32>,
        n_probes: Option<i32>,
    ) -> Result<serde_json::Value> {
        let url = format!("{}/api/v1/ann/index", self.info.base_url);
        let mut body = json!({
            "index_type": index_type,
        });
        if let Some(met) = metric {
            body["metric"] = json!(met);
        }
        if let Some(val) = m {
            body["m"] = json!(val);
        }
        if let Some(val) = m0 {
            body["m0"] = json!(val);
        }
        if let Some(val) = ef_construction {
            body["ef_construction"] = json!(val);
        }
        if let Some(val) = n_clusters {
            body["n_clusters"] = json!(val);
        }
        if let Some(val) = n_probes {
            body["n_probes"] = json!(val);
        }

        request(&self.info.client, reqwest::Method::POST, &url, Some(body), None)
    }

    pub fn get_index_info(&self, index_type: Option<&str>) -> Result<serde_json::Value> {
        let url = format!("{}/api/v1/ann/index", self.info.base_url);
        let query = index_type.map(|t| vec![("index_type", t)]);
        request(
            &self.info.client,
            reqwest::Method::GET,
            &url,
            None,
            query.as_deref(),
        )
    }

    pub fn save_index(&self, index_type: &str) -> Result<serde_json::Value> {
        let url = format!("{}/api/v1/ann/index/save", self.info.base_url);
        let query = [("index_type", index_type)];
        request(
            &self.info.client,
            reqwest::Method::POST,
            &url,
            None,
            Some(&query),
        )
    }

    pub fn load_index(&self, index_type: &str) -> Result<serde_json::Value> {
        let url = format!("{}/api/v1/ann/index/load", self.info.base_url);
        let query = [("index_type", index_type)];
        request(
            &self.info.client,
            reqwest::Method::POST,
            &url,
            None,
            Some(&query),
        )
    }

    pub fn compare_search(&self, query_vector: Vec<f64>, k: Option<i32>) -> Result<serde_json::Value> {
        let url = format!("{}/api/v1/ann/search/compare", self.info.base_url);
        let vec_str = query_vector.iter()
            .map(|x| x.to_string())
            .collect::<Vec<String>>()
            .join(",");
        let k_str = k.unwrap_or(10).to_string();
        let query = [
            ("query_vector", vec_str.as_str()),
            ("k", k_str.as_str()),
        ];
        request(
            &self.info.client,
            reqwest::Method::GET,
            &url,
            None,
            Some(&query),
        )
    }

    pub fn get_statistics(&self) -> Result<serde_json::Value> {
        let url = format!("{}/api/v1/ann/stats", self.info.base_url);
        request(&self.info.client, reqwest::Method::GET, &url, None, None)
    }
}
