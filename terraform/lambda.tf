# ================================================================
# lambda.tf
# Creates the Lambda function for on-demand model retraining.
# This is what your mentor means by "resource provision Lambda".
# You never edit this file.
# ================================================================

resource "aws_lambda_function" "retrain" {
  function_name = "${var.project_name}-retrain"
  role          = aws_iam_role.lambda_role.arn

  # Uses your Lambda Docker image from ECR
  package_type = "Image"
  image_uri    = "${var.aws_account_id}.dkr.ecr.${var.aws_region}.amazonaws.com/${var.ecr_lambda_repo_name}:latest"

  timeout     = 900   # 15 minutes — training takes time
  memory_size = 1024  # 1 GB RAM — XGBoost needs memory

  environment {
    variables = {
      S3_BUCKET = var.s3_bucket_name
    }
  }

  tags = { Project = var.project_name, ManagedBy = "Terraform" }
}

output "lambda_function_name" {
  description = "Lambda function name — use this to trigger manually"
  value       = aws_lambda_function.retrain.function_name
}

output "lambda_function_arn" {
  description = "Lambda function ARN"
  value       = aws_lambda_function.retrain.arn
}
