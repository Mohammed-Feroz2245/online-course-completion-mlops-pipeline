# ================================================================
# ecr.tf
# Both ECR repos already exist — we import them, not recreate.
# This file describes both: one for FastAPI, one for Lambda.
# You never edit this file.
# ================================================================

# ECR repo for your FastAPI Docker image
resource "aws_ecr_repository" "api" {
  name                 = var.ecr_api_repo_name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Project   = var.project_name
    ManagedBy = "Terraform"
  }
}

# ECR repo for your Lambda Docker image
resource "aws_ecr_repository" "lambda_repo" {
  name                 = var.ecr_lambda_repo_name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Project   = var.project_name
    ManagedBy = "Terraform"
  }
}

# Outputs — printed after terraform apply
output "ecr_api_url" {
  description = "FastAPI ECR URL — already in your ci.yml"
  value       = aws_ecr_repository.api.repository_url
}

output "ecr_lambda_url" {
  description = "Lambda ECR URL — use when pushing Lambda image"
  value       = aws_ecr_repository.lambda_repo.repository_url
}
