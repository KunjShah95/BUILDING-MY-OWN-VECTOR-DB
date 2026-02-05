import time
import numpy as np
from typing import List, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field
import json
import os
import statistics
from datetime import datetime


@dataclass
class BenchmarkResult:
    """Benchmark result container"""

    metric_name: str
    value: float
    unit: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """Complete benchmark report"""

    timestamp: str
    total_duration: float
    metrics: List[BenchmarkResult]
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "total_duration": self.total_duration,
            "metrics": [
                {
                    "metric_name": m.metric_name,
                    "value": m.value,
                    "unit": m.unit,
                    "metadata": m.metadata,
                }
                for m in self.metrics
            ],
            "details": self.details,
        }


class BenchmarkSuite:
    """
    Comprehensive benchmarking suite for vector database
    Measures recall, latency, throughput, and memory usage
    """

    def __init__(self, name: str = "VectorDB Benchmark"):
        """
        Initialize benchmark suite

        Args:
            name: Name of the benchmark suite
        """
        self.name = name
        self.results: List[BenchmarkReport] = []

    def measure_recall(
        self,
        ground_truth_func: Callable,
        search_func: Callable,
        query_vectors: List[List[float]],
        ground_truth_k: int = 10,
        search_k: int = 10,
        num_samples: int = 100,
    ) -> BenchmarkResult:
        """
        Measure recall (accuracy) of search results

        Args:
            ground_truth_func: Function that returns ground truth results
            search_func: Function that returns search results
            query_vectors: List of query vectors
            ground_truth_k: K for ground truth
            search_k: K for search
            num_samples: Number of samples to test

        Returns:
            BenchmarkResult with recall value
        """
        total_recall = 0.0
        tested = 0

        for query in query_vectors[:num_samples]:
            # Get ground truth (using brute force)
            gt_results = ground_truth_func(query, k=ground_truth_k)
            gt_set = set([r["vector_id"] for r in gt_results])

            # Get search results
            search_results = search_func(query, k=search_k)
            search_set = set([r["vector_id"] for r in search_results])

            # Calculate recall
            if len(gt_set) > 0:
                recall = len(gt_set.intersection(search_set)) / len(gt_set)
                total_recall += recall
                tested += 1

        avg_recall = (total_recall / tested) if tested > 0 else 0.0

        return BenchmarkResult(
            metric_name="recall",
            value=avg_recall * 100,  # Convert to percentage
            unit="%",
            metadata={
                "ground_truth_k": ground_truth_k,
                "search_k": search_k,
                "samples_tested": tested,
            },
        )

    def measure_latency(
        self,
        search_func: Callable,
        query_vectors: List[List[float]],
        k: int = 10,
        num_iterations: int = 10,
    ) -> List[BenchmarkResult]:
        """
        Measure search latency (response time)

        Args:
            search_func: Search function
            query_vectors: List of query vectors
            k: Number of results
            num_iterations: Iterations per query

        Returns:
            List of BenchmarkResults for different percentiles
        """
        latencies = []

        # Warmup
        for query in query_vectors[:5]:
            search_func(query, k=k)

        # Measure latencies
        for query in query_vectors:
            for _ in range(num_iterations):
                start = time.perf_counter()
                search_func(query, k=k)
                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # Convert to milliseconds

        # Calculate percentiles
        latencies.sort()
        n = len(latencies)

        percentiles = {
            "min": latencies[0] if latencies else 0,
            "max": latencies[-1] if latencies else 0,
            "avg": statistics.mean(latencies) if latencies else 0,
            "median": latencies[n // 2] if n > 0 else 0,
            "p95": latencies[int(n * 0.95)] if n > 0 else 0,
            "p99": latencies[int(n * 0.99)] if n > 0 else 0,
        }

        results = []
        for name, value in percentiles.items():
            results.append(
                BenchmarkResult(
                    metric_name=f"latency_{name}",
                    value=value,
                    unit="ms",
                    metadata={"k": k, "iterations": num_iterations},
                )
            )

        return results

    def measure_throughput(
        self,
        search_func: Callable,
        query_vectors: List[List[float]],
        k: int = 10,
        duration_seconds: float = 10,
    ) -> BenchmarkResult:
        """
        Measure search throughput (queries per second)

        Args:
            search_func: Search function
            query_vectors: List of query vectors
            k: Number of results
            duration_seconds: Test duration

        Returns:
            BenchmarkResult with throughput value
        """
        start_time = time.perf_counter()
        queries = 0

        query_index = 0
        while (time.perf_counter() - start_time) < duration_seconds:
            query = query_vectors[query_index % len(query_vectors)]
            search_func(query, k=k)
            queries += 1
            query_index += 1

        end_time = time.perf_counter()
        actual_duration = end_time - start_time
        throughput = queries / actual_duration

        return BenchmarkResult(
            metric_name="throughput",
            value=throughput,
            unit="queries/second",
            metadata={
                "k": k,
                "duration_seconds": actual_duration,
                "total_queries": queries,
            },
        )

    def measure_index_size(self, index) -> BenchmarkResult:
        """
        Measure index size in memory

        Args:
            index: Index object with get_graph_stats or similar method

        Returns:
            BenchmarkResult with size value
        """
        try:
            stats = index.get_graph_stats()
            total_nodes = stats.get("total_nodes", 0)
            total_edges = stats.get("total_edges", 0)

            # Estimate memory usage
            # Each edge is ~8 bytes (2 ints), each node has overhead
            estimated_bytes = total_edges * 8 + total_nodes * 100

            return BenchmarkResult(
                metric_name="index_size",
                value=estimated_bytes / (1024 * 1024),  # Convert to MB
                unit="MB",
                metadata={"total_nodes": total_nodes, "total_edges": total_edges},
            )
        except Exception as e:
            return BenchmarkResult(
                metric_name="index_size",
                value=0,
                unit="MB",
                metadata={"error": str(e)},
            )

    def run_comprehensive_benchmark(
        self,
        ground_truth_func: Callable,
        search_func: Callable,
        query_vectors: List[List[float]],
        database_vectors: int,
        k: int = 10,
    ) -> BenchmarkReport:
        """
        Run comprehensive benchmark suite

        Args:
            ground_truth_func: Function for ground truth
            search_func: Search function to benchmark
            query_vectors: Query vectors
            database_vectors: Total vectors in database
            k: Number of results

        Returns:
            Complete BenchmarkReport
        """
        start_time = time.time()
        metrics = []
        details = {}

        print(f"\n{'='*60}")
        print("Running Comprehensive Benchmark")
        print(f"{'='*60}")
        print(f"Database size: {database_vectors} vectors")
        print(f"Query samples: {len(query_vectors)}")
        print(f"Results per query (k): {k}")
        print(f"{'='*60}\n")

        # 1. Measure Recall
        print("1. Measuring Recall...")
        recall_result = self.measure_recall(
            ground_truth_func,
            search_func,
            query_vectors,
            k=k,
            num_samples=min(50, len(query_vectors)),
        )
        metrics.append(recall_result)
        print(f"   Recall: {recall_result.value:.2f}%")

        # 2. Measure Latency
        print("\n2. Measuring Latency...")
        latency_results = self.measure_latency(
            search_func, query_vectors, k=k, num_iterations=10
        )
        metrics.extend(latency_results)
        print(f"   Average: {latency_results[2].value:.2f}ms")  # avg
        print(f"   P95: {latency_results[4].value:.2f}ms")

        # 3. Measure Throughput
        print("\n3. Measuring Throughput...")
        throughput_result = self.measure_throughput(
            search_func, query_vectors, k=k, duration_seconds=5
        )
        metrics.append(throughput_result)
        print(f"   Throughput: {throughput_result.value:.2f} queries/sec")

        # Collect details
        details = {
            "database_vectors": database_vectors,
            "query_samples": len(query_vectors),
            "k": k,
            "latency_percentiles": {
                "min": latency_results[0].value,
                "max": latency_results[1].value,
                "avg": latency_results[2].value,
                "median": latency_results[3].value,
                "p95": latency_results[4].value,
                "p99": latency_results[5].value,
            },
        }

        total_duration = time.time() - start_time

        # Create report
        report = BenchmarkReport(
            timestamp=datetime.now().isoformat(),
            total_duration=total_duration,
            metrics=metrics,
            details=details,
        )

        self.results.append(report)

        print(f"\n{'='*60}")
        print(f"Benchmark completed in {total_duration:.2f} seconds")
        print(f"{'='*60}\n")

        return report

    def save_report(self, report: BenchmarkReport, filepath: str):
        """Save benchmark report to JSON file"""
        with open(filepath, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"Report saved to {filepath}")

    def generate_summary_table(self, report: BenchmarkReport) -> str:
        """Generate ASCII summary table"""
        table = f"\nBenchmark Summary for {self.name}\n"
        table += f"Timestamp: {report.timestamp}\n"
        table += f"{'='*50}\n"
        table += f"{'Metric':<25} {'Value':<15} {'Unit':<10}\n"
        table += f"{'='*50}\n"

        for metric in report.metrics:
            table += (
                f"{metric.metric_name:<25} {metric.value:<15.4f} {metric.unit:<10}\n"
            )

        table += f"{'='*50}\n"
        return table


class PerformanceComparator:
    """
    Compare performance between different index configurations
    """

    def __init__(self):
        self.comparisons: List[Dict[str, Any]] = []

    def compare_configurations(
        self,
        configs: List[Dict[str, Any]],
        benchmark_suite: BenchmarkSuite,
        ground_truth_func: Callable,
        search_func: Callable,
        query_vectors: List[List[float]],
        database_vectors: int,
        k: int = 10,
    ) -> Dict[str, Any]:
        """
        Compare different index configurations

        Args:
            configs: List of configuration dictionaries
            benchmark_suite: BenchmarkSuite instance
            ground_truth_func: Ground truth function
            search_func: Search function
            query_vectors: Query vectors
            database_vectors: Total vectors
            k: Number of results

        Returns:
            Comparison results
        """
        results = []

        for i, config in enumerate(configs):
            config_name = config.get("name", f"Config {i+1}")
            print(f"\n{'='*60}")
            print(f"Testing Configuration: {config_name}")
            print(f"Config: {config}")
            print(f"{'='*60}")

            # Run benchmark
            report = benchmark_suite.run_comprehensive_benchmark(
                ground_truth_func, search_func, query_vectors, database_vectors, k
            )

            # Extract key metrics
            result = {
                "config_name": config_name,
                "config": config,
                "recall": None,
                "latency_avg": None,
                "latency_p95": None,
                "throughput": None,
            }

            for metric in report.metrics:
                if metric.metric_name == "recall":
                    result["recall"] = metric.value
                elif metric.metric_name == "latency_avg":
                    result["latency_avg"] = metric.value
                elif metric.metric_name == "latency_p95":
                    result["latency_p95"] = metric.value
                elif metric.metric_name == "throughput":
                    result["throughput"] = metric.value

            results.append(result)

            print(f"\nResults for {config_name}:")
            print(f"  Recall: {result['recall']:.2f}%")
            print(f"  Avg Latency: {result['latency_avg']:.2f}ms")
            print(f"  P95 Latency: {result['latency_p95']:.2f}ms")
            print(f"  Throughput: {result['throughput']:.2f} qps")

        # Generate comparison summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "configs_tested": len(configs),
            "results": results,
            "best_config": self._find_best_config(results),
            "recommendation": self._generate_recommendation(results),
        }

        self.comparisons.append(summary)

        return summary

    def _find_best_config(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find best configuration based on recall and speed"""
        if not results:
            return {}

        # Calculate combined score (higher recall, lower latency is better)
        best = None
        best_score = -float("inf")

        for result in results:
            if result["recall"] is None or result["latency_avg"] is None:
                continue

            # Combined score: recall - normalized latency
            score = result["recall"] - (result["latency_avg"] / 100)  # Normalize latency
            if score > best_score:
                best_score = score
                best = result

        return best if best else {}

    def _generate_recommendation(self, results: List[Dict[str, Any]]) -> str:
        """Generate configuration recommendation"""
        if not results:
            return "No results to compare"

        best = self._find_best_config(results)
        if not best:
            return "Unable to determine best configuration"

        return (
            f"Best configuration: {best['config_name']} with "
            f"{best['recall']:.2f}% recall and {best['latency_avg']:.2f}ms avg latency"
        )

    def print_comparison_table(self, comparison_result: Dict[str, Any]):
        """Print ASCII comparison table"""
        results = comparison_result["results"]

        print(f"\n{'='*80}")
        print("Configuration Comparison")
        print(f"{'='*80}")
        print(
            f"{'Configuration':<20} {'Recall':<10} {'Avg Latency':<15} "
            f"{'P95 Latency':<15} {'Throughput':<15}"
        )
        print(f"{'='*80}")

        for result in results:
            name = result["config_name"][:19]
            recall = f"{result['recall']:.2f}%" if result["recall"] else "N/A"
            latency = (
                f"{result['latency_avg']:.2f}ms" if result["latency_avg"] else "N/A"
            )
            p95 = (
                f"{result['latency_p95']:.2f}ms" if result["latency_p95"] else "N/A"
            )
            throughput = (
                f"{result['throughput']:.2f} qps" if result["throughput"] else "N/A"
            )

            print(f"{name:<20} {recall:<10} {latency:<15} {p95:<15} {throughput:<15}")

        print(f"{'='*80}")
        print(f"\nRecommendation: {comparison_result['recommendation']}")
"""
Benchmark utilities for vector database performance testing
"""

import time
import json
import numpy as np
from typing import List, Dict, Any, Callable
from dataclasses import dataclass
from statistics import mean, median, stdev
import logging

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Result of a single benchmark run"""
    query_time: float
    recall: float
    precision: float
    f1_score: float
    throughput: float

@dataclass
class BenchmarkReport:
    """Comprehensive benchmark report"""
    name: str
    timestamp: str
    config: Dict[str, Any]
    results: List[BenchmarkResult]
    summary_stats: Dict[str, float]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "timestamp": self.timestamp,
            "config": self.config,
            "summary_stats": self.summary_stats,
            "recommendations": self.recommendations,
            "results": [
                {
                    "query_time": r.query_time,
                    "recall": r.recall,
                    "precision": r.precision,
                    "f1_score": r.f1_score,
                    "throughput": r.throughput,
                }
                for r in self.results
            ],
        }

class BenchmarkSuite:
    """
    Comprehensive benchmarking suite for vector databases
    """

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def run_comprehensive_benchmark(
        self,
        ground_truth_func: Callable,
        search_func: Callable,
        query_vectors: List[List[float]],
        database_vectors: int,
        k: int = 10
    ) -> BenchmarkReport:
        """
        Run comprehensive benchmark comparing search functions

        Args:
            ground_truth_func: Function for ground truth search
            search_func: Function to benchmark
            query_vectors: List of query vectors
            database_vectors: Number of vectors in database
            k: Number of nearest neighbors to find

        Returns:
            BenchmarkReport with detailed results
        """
        self.logger.info(f"Starting comprehensive benchmark: {self.name}")

        results = []
        total_start_time = time.time()

        for i, query in enumerate(query_vectors):
            if i % 10 == 0:
                self.logger.info(f"Processing query {i+1}/{len(query_vectors)}")

            # Get ground truth
            ground_truth_start = time.time()
            ground_truth_results = ground_truth_func(query, k=k)
            ground_truth_time = time.time() - ground_truth_start

            ground_truth_ids = set()
            if isinstance(ground_truth_results, dict) and "results" in ground_truth_results:
                ground_truth_ids = {r.get("vector_id") for r in ground_truth_results["results"]}
            elif isinstance(ground_truth_results, list):
                ground_truth_ids = {r.get("vector_id") for r in ground_truth_results}

            # Benchmark search function
            search_start = time.time()
            search_results = search_func(query, k=k)
            search_time = time.time() - search_start

            search_ids = set()
            if isinstance(search_results, dict) and "results" in search_results:
                search_ids = {r.get("vector_id") for r in search_results["results"]}
            elif isinstance(search_results, list):
                search_ids = {r.get("vector_id") for r in search_results}

            # Calculate metrics
            if ground_truth_ids:
                recall = len(search_ids & ground_truth_ids) / len(ground_truth_ids)
                precision = len(search_ids & ground_truth_ids) / len(search_ids) if search_ids else 0
                f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            else:
                recall = precision = f1_score = 0

            throughput = 1.0 / search_time if search_time > 0 else 0

            results.append(BenchmarkResult(
                query_time=search_time,
                recall=recall,
                precision=precision,
                f1_score=f1_score,
                throughput=throughput
            ))

        total_time = time.time() - total_start_time

        # Calculate summary statistics
        query_times = [r.query_time for r in results]
        recalls = [r.recall for r in results]
        precisions = [r.precision for r in results]
        f1_scores = [r.f1_score for r in results]
        throughputs = [r.throughput for r in results]

        summary_stats = {
            "total_queries": len(results),
            "total_time": total_time,
            "avg_query_time": mean(query_times),
            "median_query_time": median(query_times),
            "min_query_time": min(query_times),
            "max_query_time": max(query_times),
            "std_query_time": stdev(query_times) if len(query_times) > 1 else 0,
            "avg_recall": mean(recalls),
            "avg_precision": mean(precisions),
            "avg_f1_score": mean(f1_scores),
            "avg_throughput": mean(throughputs),
            "queries_per_second": len(results) / total_time
        }

        # Generate recommendations
        recommendations = self._generate_recommendations(summary_stats)

        report = BenchmarkReport(
            name=self.name,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            config={
                "database_vectors": database_vectors,
                "query_vectors": len(query_vectors),
                "k": k
            },
            results=results,
            summary_stats=summary_stats,
            recommendations=recommendations
        )

        self.logger.info(f"Benchmark completed in {total_time:.2f}s")
        return report

    def generate_summary_table(self, report: BenchmarkReport) -> str:
        """Generate a formatted summary table"""
        stats = report.summary_stats

        table = f"""
Benchmark Summary: {report.name}
{'='*60}
Configuration:
  Database Vectors: {report.config['database_vectors']:,}
  Query Vectors: {report.config['query_vectors']:,}
  k: {report.config['k']}

Performance Metrics:
  Total Time: {stats['total_time']:.2f}s
  Queries/Second: {stats['queries_per_second']:.2f}
  Avg Query Time: {stats['avg_query_time']*1000:.2f}ms
  Median Query Time: {stats['median_query_time']*1000:.2f}ms
  Min Query Time: {stats['min_query_time']*1000:.2f}ms
  Max Query Time: {stats['max_query_time']*1000:.2f}ms

Accuracy Metrics:
  Avg Recall@{report.config['k']}: {stats['avg_recall']:.4f}
  Avg Precision@{report.config['k']}: {stats['avg_precision']:.4f}
  Avg F1 Score: {stats['avg_f1_score']:.4f}

Recommendations:
"""
        for rec in report.recommendations:
            table += f"  • {rec}\n"

        return table

    def save_report(self, report: BenchmarkReport, filepath: str):
        """Save benchmark report to JSON file"""
        data = {
            "name": report.name,
            "timestamp": report.timestamp,
            "config": report.config,
            "summary_stats": report.summary_stats,
            "recommendations": report.recommendations,
            "results": [
                {
                    "query_time": r.query_time,
                    "recall": r.recall,
                    "precision": r.precision,
                    "f1_score": r.f1_score,
                    "throughput": r.throughput
                }
                for r in report.results
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        self.logger.info(f"Report saved to {filepath}")

    def _generate_recommendations(self, stats: Dict[str, float]) -> List[str]:
        """Generate performance recommendations based on stats"""
        recommendations = []

        if stats['avg_recall'] < 0.8:
            recommendations.append("Consider increasing index parameters (m, ef_construction) for better recall")
        elif stats['avg_recall'] > 0.95:
            recommendations.append("High recall achieved - consider optimizing for speed if needed")

        if stats['avg_query_time'] > 0.1:  # 100ms
            recommendations.append("Query times are high - consider optimizing index or reducing ef_search")
        elif stats['avg_query_time'] < 0.01:  # 10ms
            recommendations.append("Excellent query performance - can increase ef_search for better accuracy")

        if stats['std_query_time'] > stats['avg_query_time'] * 0.5:
            recommendations.append("High variance in query times - consider index optimization")

        if stats['queries_per_second'] < 100:
            recommendations.append("Low throughput - consider batch processing or index optimization")

        return recommendations if recommendations else ["Performance looks good!"]

class PerformanceComparator:
    """
    Compare performance across different configurations
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def compare_configurations(
        self,
        configs: List[Dict[str, Any]],
        benchmark_suite: BenchmarkSuite,
        ground_truth_func: Callable,
        search_func_factory: Callable,
        query_vectors: List[List[float]],
        database_vectors: int,
        k: int = 10
    ) -> Dict[str, Any]:
        """
        Compare multiple configurations

        Args:
            configs: List of configuration dictionaries
            benchmark_suite: BenchmarkSuite instance
            ground_truth_func: Ground truth search function
            search_func_factory: Function that creates search functions for configs
            query_vectors: Query vectors
            database_vectors: Number of database vectors
            k: Number of neighbors

        Returns:
            Comparison results
        """
        self.logger.info(f"Comparing {len(configs)} configurations")

        results = {}
        for config in configs:
            config_name = config["name"]
            self.logger.info(f"Running benchmark for {config_name}")

            # Create config-specific search function using the factory
            config_search_func = search_func_factory(config["m"], config["ef_construction"])

            report = benchmark_suite.run_comprehensive_benchmark(
                ground_truth_func=ground_truth_func,
                search_func=config_search_func,
                query_vectors=query_vectors,
                database_vectors=database_vectors,
                k=k
            )

            results[config_name] = {
                "config": config,
                "report": report
            }

        return results

    def print_comparison_table(self, comparison: Dict[str, Any]):
        """Print formatted comparison table"""
        print("\n" + "="*80)
        print("Configuration Comparison")
        print("="*80)

        headers = ["Configuration", "Avg Time (ms)", "Recall@10", "Precision@10", "F1 Score", "QPS"]
        print(f"{headers[0]:<15} {headers[1]:<12} {headers[2]:<10} {headers[3]:<12} {headers[4]:<9} {headers[5]:<6}")
        print("-" * 80)

        for config_name, data in comparison.items():
            stats = data["report"].summary_stats
            print(f"{config_name:<15} "
                  f"{stats['avg_query_time']*1000:<12.2f} "
                  f"{stats['avg_recall']:<10.4f} "
                  f"{stats['avg_precision']:<12.4f} "
                  f"{stats['avg_f1_score']:<9.4f} "
                  f"{stats['queries_per_second']:<6.2f}")

        print("="*80)
