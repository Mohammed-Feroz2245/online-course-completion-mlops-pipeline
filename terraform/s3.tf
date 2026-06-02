# ================================================================
# s3.tf
# Your S3 bucket already exists — we import it, not recreate it.
# This file just describes what it looks like so Terraform can manage it.
# You never edit this file.
# ================================================================

resource "aws_s3_bucket" "ml_artifacts" {
  bucket = var.s3_bucket_name

  tags = {
    Project   = var.project_name
    ManagedBy = "Terraform"
  }
}

# Block all public access — bucket must be private
resource "aws_s3_bucket_public_access_block" "ml_artifacts" {
  bucket = aws_s3_bucket.ml_artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
