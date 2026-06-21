use mockito::Server;
use serde_json::json;
use vectordb_client::VectorDBClient;

#[test]
fn test_health() {
    let mut server = Server::new();
    let client = VectorDBClient::new(&server.url(), Some("secret_key"), Some("tenant_123"));

    let mock = server.mock("GET", "/health")
        .match_header("authorization", "Bearer secret_key")
        .match_header("x-tenant-id", "tenant_123")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"status": "ok"}"#)
        .create();

    let res = client.health().unwrap();
    assert_eq!(res["status"], "ok");
    mock.assert();
}

#[test]
fn test_collections_api() {
    let mut server = Server::new();
    let client = VectorDBClient::new(&server.url(), None, None);

    // 1. Create collection
    {
        let create_mock = server.mock("POST", "/collections")
            .match_body(mockito::Matcher::Json(json!({
                "name": "testcol",
                "modality": "text",
                "distance_metric": "cosine",
                "dimension": 128,
                "embedding_model": "all-MiniLM-L6-v2",
                "description": "test desc"
            })))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{
                "success": true,
                "collection": {
                    "collection_id": "testcol",
                    "name": "testcol",
                    "modality": "text",
                    "dimension": 128,
                    "embedding_model": "all-MiniLM-L6-v2",
                    "distance_metric": "cosine",
                    "description": "test desc"
                }
            }"#)
            .create();

        let col = client.collections.create(
            "testcol",
            None,
            Some("text"),
            Some(128),
            Some("all-MiniLM-L6-v2"),
            Some("cosine"),
            Some("test desc")
        ).unwrap();
        assert_eq!(col.name, "testcol");
        assert_eq!(col.dimension, Some(128));
        create_mock.assert();
    }

    // 2. List collections
    {
        let list_mock = server.mock("GET", "/collections")
            .match_query(mockito::Matcher::Any)
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{
                "collections": [
                    {
                        "collection_id": "testcol",
                        "name": "testcol",
                        "modality": "text",
                        "dimension": 128,
                        "embedding_model": "all-MiniLM-L6-v2",
                        "distance_metric": "cosine",
                        "description": "test desc"
                    }
                ]
            }"#)
            .create();

        let list = client.collections.list(Some(10), Some(0)).unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].collection_id, "testcol");
        list_mock.assert();
    }

    // 3. Get collection
    {
        let get_mock = server.mock("GET", "/collections/testcol")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{
                "success": true,
                "collection": {
                    "collection_id": "testcol",
                    "name": "testcol",
                    "modality": "text",
                    "dimension": 128,
                    "embedding_model": "all-MiniLM-L6-v2",
                    "distance_metric": "cosine",
                    "description": "test desc"
                }
            }"#)
            .create();

        let col_get = client.collections.get("testcol").unwrap();
        assert_eq!(col_get.collection_id, "testcol");
        get_mock.assert();
    }

    // 4. Build index
    {
        let build_mock = server.mock("POST", "/collections/testcol/index")
            .match_body(mockito::Matcher::Json(json!({
                "method": "hnsw",
                "m": 16,
                "ef_construction": 200,
                "n_clusters": 100,
                "n_probes": 10
            })))
            .with_status(200)
            .with_body(r#"{"success": true, "message": "Index built"}"#)
            .create();

        let res_build = client.collections.build_index("testcol", Some("hnsw"), Some(16), None, Some(200), Some(100), Some(10)).unwrap();
        assert_eq!(res_build["success"], true);
        build_mock.assert();
    }

    // 5. Index stats
    {
        let stats_mock = server.mock("GET", "/collections/testcol/index/stats")
            .with_status(200)
            .with_body(r#"{"vector_count": 42}"#)
            .create();

        let res_stats = client.collections.index_stats("testcol").unwrap();
        assert_eq!(res_stats["vector_count"], 42);
        stats_mock.assert();
    }

    // 6. Delete collection
    {
        let delete_mock = server.mock("DELETE", "/collections/testcol")
            .with_status(200)
            .with_body(r#"{"success": true}"#)
            .create();

        let res_del = client.collections.delete("testcol").unwrap();
        assert_eq!(res_del["success"], true);
        delete_mock.assert();
    }
}

#[test]
fn test_vectors_api() {
    let mut server = Server::new();
    let client = VectorDBClient::new(&server.url(), None, None);

    // 1. Create vector
    {
        let create_mock = server.mock("POST", "/vectors")
            .match_body(mockito::Matcher::Json(json!({
                "vector": [1.0, 2.0, 3.0],
                "metadata": {"key": "val"},
                "vector_id": "v1"
            })))
            .with_status(200)
            .with_body(r#"{"success": true, "vector_id": "v1"}"#)
            .create();

        let res_create = client.vectors.create(
            vec![1.0, 2.0, 3.0],
            Some(json!({"key": "val"})),
            Some("v1")
        ).unwrap();
        assert_eq!(res_create["vector_id"], "v1");
        create_mock.assert();
    }

    // 2. Get vector
    {
        let get_mock = server.mock("GET", "/vectors/v1")
            .with_status(200)
            .with_body(r#"{"vector": [1.0, 2.0, 3.0], "metadata": {"key": "val"}, "vector_id": "v1"}"#)
            .create();

        let res_get = client.vectors.get("v1").unwrap();
        assert_eq!(res_get["vector_id"], "v1");
        get_mock.assert();
    }

    // 3. Search vectors
    {
        let search_mock = server.mock("POST", "/search")
            .match_body(mockito::Matcher::Json(json!({
                "query_vector": [1.0, 2.0, 3.0],
                "k": 5,
                "method": "hnsw",
                "filters": {"key": "val"}
            })))
            .with_status(200)
            .with_body(r#"{"results": [{"vector_id": "v1", "distance": 0.0}]}"#)
            .create();

        let res_search = client.vectors.search(
            vec![1.0, 2.0, 3.0],
            Some(5),
            Some("hnsw"),
            Some(json!({"key": "val"}))
        ).unwrap();
        assert_eq!(res_search["results"][0]["vector_id"], "v1");
        search_mock.assert();
    }

    // 4. Delete vector
    {
        let delete_mock = server.mock("DELETE", "/vectors/v1")
            .with_status(200)
            .with_body(r#"{"success": true}"#)
            .create();

        let res_del = client.vectors.delete("v1").unwrap();
        assert_eq!(res_del["success"], true);
        delete_mock.assert();
    }
}

#[test]
fn test_multimodal_api() {
    let mut server = Server::new();
    let client = VectorDBClient::new(&server.url(), None, None);

    // 1. Ingest text
    {
        let ingest_text_mock = server.mock("POST", "/collections/col1/ingest/text")
            .match_body(mockito::Matcher::Json(json!({
                "text": "hello world",
                "metadata": {"key": "val"},
                "vector_id": "v_text"
            })))
            .with_status(200)
            .with_body(r#"{"success": true, "vector_id": "v_text"}"#)
            .create();

        let res_ingest_text = client.multimodal.ingest_text(
            "col1",
            "hello world",
            Some(json!({"key": "val"})),
            Some("v_text")
        ).unwrap();
        assert_eq!(res_ingest_text["vector_id"], "v_text");
        ingest_text_mock.assert();
    }

    // 2. Search text
    {
        let search_text_mock = server.mock("POST", "/collections/col1/search/text")
            .match_body(mockito::Matcher::Json(json!({
                "query": "find me",
                "k": 3,
                "method": "brute",
                "filters": {"type": "news"}
            })))
            .with_status(200)
            .with_body(r#"{
                "success": true,
                "results": [
                    {
                        "vector_id": "v_text",
                        "distance": 0.1,
                        "metadata": {"type": "news"}
                    }
                ],
                "total_results": 1,
                "search_time": 0.005,
                "method": "brute"
            }"#)
            .create();

        let res_search_text = client.multimodal.search_text(
            "col1",
            "find me",
            Some(3),
            Some("brute"),
            Some(json!({"type": "news"}))
        ).unwrap();
        assert_eq!(res_search_text.results[0].vector_id, "v_text");
        assert_eq!(res_search_text.total_results, 1);
        search_text_mock.assert();
    }

    // 3. Query
    {
        let query_mock = server.mock("POST", "/collections/col1/query")
            .match_body(mockito::Matcher::Json(json!({
                "query": "what is rust",
                "k": 5
            })))
            .with_status(200)
            .with_body(r#"{"results": []}"#)
            .create();

        let res_query = client.multimodal.query("col1", "what is rust", Some(5)).unwrap();
        assert!(res_query["results"].is_array());
        query_mock.assert();
    }

    // 4. Ingest image (multipart)
    {
        let ingest_image_mock = server.mock("POST", "/collections/col1/ingest/image")
            .with_status(200)
            .with_body(r#"{"success": true, "vector_id": "v_image"}"#)
            .create();

        let res_ingest_image = client.multimodal.ingest_image(
            "col1",
            vec![1, 2, 3, 4],
            "image.jpg",
            Some(json!({"source": "camera"})),
            Some("v_image")
        ).unwrap();
        assert_eq!(res_ingest_image["vector_id"], "v_image");
        ingest_image_mock.assert();
    }

    // 5. Search image (multipart)
    {
        let search_image_mock = server.mock("POST", "/collections/col1/search/image")
            .with_status(200)
            .with_body(r#"{
                "success": true,
                "results": [
                    {
                        "vector_id": "v_image",
                        "distance": 0.25
                    }
                ],
                "total_results": 1,
                "search_time": 0.012,
                "method": "brute"
            }"#)
            .create();

        let res_search_image = client.multimodal.search_image(
            "col1",
            vec![5, 6, 7, 8],
            "query.jpg",
            Some(5),
            Some("brute"),
            None
        ).unwrap();
        assert_eq!(res_search_image.results[0].vector_id, "v_image");
        search_image_mock.assert();
    }
}

#[test]
fn test_ann_api() {
    let mut server = Server::new();
    let client = VectorDBClient::new(&server.url(), None, None);

    // 1. Create index
    {
        let create_mock = server.mock("POST", "/api/v1/ann/index")
            .match_body(mockito::Matcher::Json(json!({
                "index_type": "hnsw",
                "m": 16,
                "ef_construction": 200
            })))
            .with_status(200)
            .with_body(r#"{"success": true, "message": "Created hnsw index"}"#)
            .create();

        let res_create = client.ann.create_index(
            "hnsw",
            None,
            Some(16),
            None,
            Some(200),
            None,
            None
        ).unwrap();
        assert_eq!(res_create["success"], true);
        create_mock.assert();
    }

    // 2. Get index info
    {
        let info_type_mock = server.mock("GET", "/api/v1/ann/index")
            .match_query(mockito::Matcher::UrlEncoded("index_type".into(), "hnsw".into()))
            .with_status(200)
            .with_body(r#"{"success": true, "index_type": "hnsw"}"#)
            .create();

        let res_info = client.ann.get_index_info(Some("hnsw")).unwrap();
        assert_eq!(res_info["index_type"], "hnsw");
        info_type_mock.assert();
    }

    // 3. Get index info all
    {
        let info_all_mock = server.mock("GET", "/api/v1/ann/index")
            .with_status(200)
            .with_body(r#"{"hnsw": {"status": "loaded"}}"#)
            .create();

        let res_info_all = client.ann.get_index_info(None).unwrap();
        assert!(res_info_all.get("hnsw").is_some());
        info_all_mock.assert();
    }

    // 4. Save index
    {
        let save_mock = server.mock("POST", "/api/v1/ann/index/save")
            .match_query(mockito::Matcher::UrlEncoded("index_type".into(), "hnsw".into()))
            .with_status(200)
            .with_body(r#"{"success": true}"#)
            .create();

        let res_save = client.ann.save_index("hnsw").unwrap();
        assert_eq!(res_save["success"], true);
        save_mock.assert();
    }

    // 5. Load index
    {
        let load_mock = server.mock("POST", "/api/v1/ann/index/load")
            .match_query(mockito::Matcher::UrlEncoded("index_type".into(), "hnsw".into()))
            .with_status(200)
            .with_body(r#"{"success": true}"#)
            .create();

        let res_load = client.ann.load_index("hnsw").unwrap();
        assert_eq!(res_load["success"], true);
        load_mock.assert();
    }

    // 6. Compare search
    {
        let compare_mock = server.mock("GET", "/api/v1/ann/search/compare")
            .match_query(mockito::Matcher::Any)
            .with_status(200)
            .with_body(r#"{"hnsw": {"results": []}, "ivf": {"results": []}}"#)
            .create();

        let res_compare = client.ann.compare_search(vec![0.1, 0.2, 0.3], Some(5)).unwrap();
        assert!(res_compare.get("hnsw").is_some());
        compare_mock.assert();
    }

    // 7. Get statistics
    {
        let stats_mock = server.mock("GET", "/api/v1/ann/stats")
            .with_status(200)
            .with_body(r#"{"vector_count": 1000}"#)
            .create();

        let res_stats = client.ann.get_statistics().unwrap();
        assert_eq!(res_stats["vector_count"], 1000);
        stats_mock.assert();
    }
}
