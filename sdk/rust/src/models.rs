use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Collection {
    pub collection_id: String,
    pub name: String,
    pub modality: String,
    pub dimension: Option<i32>,
    pub embedding_model: Option<String>,
    pub distance_metric: String,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionEnvelope {
    pub collection: Collection,
    pub success: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionsListEnvelope {
    pub collections: Vec<Collection>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    pub vector_id: String,
    pub distance: f64,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub success: bool,
    pub results: Vec<SearchHit>,
    pub total_results: usize,
    pub search_time: f64,
    pub method: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub memory_id: String,
    pub text: String,
    pub categories: Vec<String>,
}
