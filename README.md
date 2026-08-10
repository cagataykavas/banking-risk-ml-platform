# Banking Risk ML Platform

An end-to-end portfolio project for **transaction risk scoring, anomaly detection and model-serving workflows** using a production-style data stack.

The repository is intentionally built with synthetic/public data so the full pipeline can be shared safely.

## Target architecture

```text
Synthetic/Public Transactions
          |
          v
Batch ingestion / feature engineering
          |
          +--> Spark job
          |
          v
PostgreSQL feature / score store
          |
          v
Risk models
  fraud classification
  anomaly detection
  probability calibration
          |
          v
FastAPI scoring service
          |
          +--> Docker
          +--> AWS reference deployment
          +--> CI/CD
```

## Planned components

- synthetic transaction generator
- PySpark feature-engineering job
- PostgreSQL schema and persistence layer
- supervised fraud baseline
- unsupervised anomaly baseline
- model-evaluation report
- FastAPI scoring endpoint
- Docker Compose local stack
- AWS reference architecture
- GitHub Actions tests/build

## Engineering goals

The project is designed to demonstrate more than fitting a classifier. It will include data contracts, reproducible training, batch and online paths, database persistence, model serialization, service health checks and deployment configuration.

## Safety / provenance

No real customer, banking, employer or confidential data is used. All examples are synthetic or publicly reproducible.

## Portfolio context

This repository targets Data Scientist / Applied ML roles where Python and modeling are expected, but familiarity with **SQL, Spark, databases, APIs, cloud and DevOps** is a strong differentiator.
