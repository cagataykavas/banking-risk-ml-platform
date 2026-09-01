terraform {
  required_version = ">= 1.6.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

variable "project_id" {
  type        = string
  description = "GCP project that hosts the demo platform."
}

variable "region" {
  type        = string
  default     = "europe-west1"
  description = "Primary deployment region."
}

variable "image" {
  type        = string
  description = "Container image for the FastAPI scoring service."
}

provider "google" {
  project = var.project_id
  region  = var.region
}

locals {
  services = toset([
    "artifactregistry.googleapis.com",
    "bigquery.googleapis.com",
    "run.googleapis.com",
    "pubsub.googleapis.com",
    "secretmanager.googleapis.com",
    "storage.googleapis.com",
  ])
}

resource "google_project_service" "required" {
  for_each           = local.services
  service            = each.value
  disable_on_destroy = false
}

resource "google_artifact_registry_repository" "models_api" {
  location      = var.region
  repository_id = "banking-risk-api"
  description   = "Container images for the synthetic banking-risk demo"
  format        = "DOCKER"

  depends_on = [google_project_service.required]
}

resource "google_storage_bucket" "model_artifacts" {
  name                        = "${var.project_id}-banking-risk-artifacts"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = false

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      num_newer_versions = 10
    }
    action {
      type = "Delete"
    }
  }

  depends_on = [google_project_service.required]
}

resource "google_bigquery_dataset" "risk_monitoring" {
  dataset_id                 = "banking_risk_monitoring"
  friendly_name              = "Banking Risk Monitoring"
  description                = "Synthetic scoring, drift and delayed-label monitoring marts"
  location                   = "EU"
  delete_contents_on_destroy = false

  depends_on = [google_project_service.required]
}

resource "google_pubsub_topic" "scoring_events" {
  name = "banking-risk-scoring-events"

  message_retention_duration = "86600s"
  depends_on                 = [google_project_service.required]
}

resource "google_service_account" "runtime" {
  account_id   = "banking-risk-runtime"
  display_name = "Banking Risk Runtime"
}

resource "google_project_iam_member" "runtime_bigquery" {
  project = var.project_id
  role    = "roles/bigquery.dataEditor"
  member  = "serviceAccount:${google_service_account.runtime.email}"
}

resource "google_project_iam_member" "runtime_pubsub" {
  project = var.project_id
  role    = "roles/pubsub.publisher"
  member  = "serviceAccount:${google_service_account.runtime.email}"
}

resource "google_storage_bucket_iam_member" "runtime_artifacts" {
  bucket = google_storage_bucket.model_artifacts.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.runtime.email}"
}

resource "google_cloud_run_v2_service" "api" {
  name     = "banking-risk-api"
  location = var.region

  template {
    service_account = google_service_account.runtime.email

    scaling {
      min_instance_count = 0
      max_instance_count = 10
    }

    containers {
      image = var.image

      resources {
        limits = {
          cpu    = "1"
          memory = "1Gi"
        }
      }

      env {
        name  = "MODEL_ARTIFACT_BUCKET"
        value = google_storage_bucket.model_artifacts.name
      }
      env {
        name  = "MONITORING_DATASET"
        value = google_bigquery_dataset.risk_monitoring.dataset_id
      }
      env {
        name  = "SCORING_TOPIC"
        value = google_pubsub_topic.scoring_events.name
      }
    }
  }

  depends_on = [google_project_service.required]
}

output "cloud_run_service" {
  value = google_cloud_run_v2_service.api.uri
}

output "artifact_bucket" {
  value = google_storage_bucket.model_artifacts.name
}

output "monitoring_dataset" {
  value = google_bigquery_dataset.risk_monitoring.dataset_id
}
