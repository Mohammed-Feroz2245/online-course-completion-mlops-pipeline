# ================================================================
# ecs.tf
# Creates ECS cluster and task definition for your FastAPI container.
# ECS = the service that runs your Docker container in the cloud.
# You never edit this file.
# ================================================================

resource "aws_ecs_cluster" "main" {
  name = "${var.project_name}-cluster"
  tags = { Project = var.project_name, ManagedBy = "Terraform" }
}

resource "aws_ecs_task_definition" "api" {
  family                   = "${var.project_name}-api"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = "256"
  memory                   = "512"

  execution_role_arn = aws_iam_role.ecs_execution_role.arn
  task_role_arn      = aws_iam_role.ecs_task_role.arn

  container_definitions = jsonencode([{
    name  = "${var.project_name}-api"
    image = "${var.aws_account_id}.dkr.ecr.${var.aws_region}.amazonaws.com/${var.ecr_api_repo_name}:latest"

    portMappings = [{
      containerPort = 8000
      protocol      = "tcp"
    }]

    environment = [
      { name = "S3_BUCKET", value = var.s3_bucket_name }
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = "/ecs/${var.project_name}-api"
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "ecs"
      }
    }
  }])

  tags = { Project = var.project_name, ManagedBy = "Terraform" }
}

resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/${var.project_name}-api"
  retention_in_days = 7
  tags = { Project = var.project_name, ManagedBy = "Terraform" }
}

output "ecs_cluster_name" {
  value = aws_ecs_cluster.main.name
}
