# ================================================================
# main.tf
# Tells Terraform: use AWS, in this region, with these credentials
# You never edit this file.
# ================================================================

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  # State file stays local for now (terraform.tfstate in your terraform/ folder)
  # This is fine for solo projects.
}

provider "aws" {
  region     = var.aws_region
  access_key = var.aws_access_key
  secret_key = var.aws_secret_key
  # These values come from terraform.tfvars — never hardcoded here
}
