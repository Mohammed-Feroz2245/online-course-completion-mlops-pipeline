# ================================================================
# variables.tf
# Defines what variables exist. Actual values come from terraform.tfvars
# You never edit this file.
# ================================================================

variable "aws_account_id" {
  description = "Your 12-digit AWS account ID"
  type        = string
}

variable "aws_access_key" {
  description = "AWS access key for F0zDEV IAM user"
  type        = string
  sensitive   = true
}

variable "aws_secret_key" {
  description = "AWS secret key for F0zDEV IAM user"
  type        = string
  sensitive   = true
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "eu-north-1"
}

variable "project_name" {
  description = "Project prefix used in resource names"
  type        = string
  default     = "course-completion"
}

variable "s3_bucket_name" {
  description = "S3 bucket that holds your CSV data and model.pkl"
  type        = string
  default     = "course-completion-ml-artifacts"
}

variable "ecr_api_repo_name" {
  description = "ECR repository name for the FastAPI Docker image"
  type        = string
}

variable "ecr_lambda_repo_name" {
  description = "ECR repository name for the Lambda Docker image"
  type        = string
}
