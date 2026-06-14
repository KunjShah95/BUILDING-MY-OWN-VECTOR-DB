# Vector DB — Go Client

Go client for the Vector Database multimodal API. Works with Go 1.21+.

## Install

```bash
go get github.com/KunjShah95/BUILDING-MY-OWN-VECTOR-DB/sdk/go
```

## Quickstart

```go
package main

import (
    "fmt"
    "log"

    "github.com/KunjShah95/BUILDING-MY-OWN-VECTOR-DB/sdk/go/vectordb"
)

func main() {
    client := vectordb.NewClient(vectordb.ClientOptions{
        BaseURL: "http://localhost:8000",
    })

    // ── Health ──────────────────────────────────────

    health, err := client.Health()
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println("Health:", health)

    // ── Collections ──────────────────────────────────

    coll, err := client.Collections.Create(vectordb.CreateCollectionParams{
        Name:         "Docs",
        CollectionID: "docs",
        Modality:     "text",
        Dimension:    384,
    })
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Created collection: %s\n", coll.CollectionID)

    collections, err := client.Collections.List(100, 0)
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Found %d collections\n", len(collections))

    // ── Text ingest & search ─────────────────────────

    _, err = client.Multimodal.IngestText("docs", vectordb.IngestTextParams{
        Text:     "Returns accepted within 30 days.",
        VectorID: "policy-returns",
    })
    if err != nil {
        log.Fatal(err)
    }

    result, err := client.Multimodal.SearchText("docs", vectordb.SearchTextParams{
        Query: "return policy",
        K:     5,
    })
    if err != nil {
        log.Fatal(err)
    }
    if len(result.Results) > 0 {
        fmt.Printf("Top hit: %s (%.4f)\n", result.Results[0].VectorID, result.Results[0].Distance)
    }

    // ── Vector CRUD ──────────────────────────────────

    _, err = client.Vectors.Create(vectordb.CreateVectorParams{
        Vector: []float64{0.1, 0.2, 0.3},
    })
    if err != nil {
        log.Fatal(err)
    }

    // ── RAG ──────────────────────────────────────────

    answer, err := client.Multimodal.Query("docs", "What is the return policy?", 5)
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println("Answer:", answer)
}
```

## API Surface

| Resource | Methods |
|----------|---------|
| `client.Collections` | `Create`, `List`, `Get`, `Delete`, `BuildIndex`, `IndexStats` |
| `client.Vectors` | `Create`, `Get`, `Delete`, `Search` |
| `client.Multimodal` | `IngestText`, `SearchText`, `IngestImage`, `SearchImage`, `IngestAudio`, `SearchAudio`, `Query` (RAG) |

Errors return `*vectordb.VectorDBHTTPError` with `StatusCode` and `Detail`.

## Development

```bash
cd sdk/go
go build ./vectordb/...
go vet ./vectordb/...
go test ./vectordb/...
```
