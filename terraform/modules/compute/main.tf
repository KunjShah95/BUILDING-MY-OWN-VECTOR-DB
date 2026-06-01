# Azure: Linux Web App for Containers (with auto-scaling)
resource "azurerm_service_plan" "asp" {
  count = var.cloud == "azure" ? 1 : 0
  name                = "${var.prefix}-asp"
  resource_group_name = var.resource_group_name
  location           = var.location
  os_type            = "Linux"
  sku_name           = var.azure_sku

  tags = { Name = "${var.prefix}-asp" }
}

resource "azurerm_linux_web_app" "api" {
  count = var.cloud == "azure" ? 1 : 0
  name                = "${var.prefix}-api"
  resource_group_name = var.resource_group_name
  location           = var.location
  service_plan_id    = azurerm_service_plan.asp[0].id
  https_only         = true

  site_config {
    always_on     = true
    http2_enabled = true
    minimum_tls_version = "1.2"
    health_check_path = "/health"
    application_stack {
      docker_image     = var.docker_image
      docker_image_tag = var.image_tag
    }
  }

  app_settings = {
    DOCKER_ENABLE_CI                  = "true"
    WEBSITES_PORT                     = "8000"
    DATABASE_URL                      = "postgresql://${var.db_user}:${var.db_password}@${var.db_host}:${var.db_port}/${var.db_name}"
    APP_NAME                          = "Vector DB API"
    APP_VERSION                       = "1.0.0"
    DEBUG                             = "false"
    DEFAULT_M                         = "16"
    DEFAULT_EF_CONSTRUCTION           = "200"
    DEFAULT_N_CLUSTERS                = "100"
    UVICORN_WORKERS                   = "4"
    API_KEY                           = var.api_key
  }

  tags = { Name = "${var.prefix}-api" }
}

resource "azurerm_monitor_autoscale_setting" "autoscale" {
  count = var.cloud == "azure" ? 1 : 0
  name                = "${var.prefix}-autoscale"
  resource_group_name = var.resource_group_name
  location           = var.location
  target_resource_id  = azurerm_service_plan.asp[0].id

  profile {
    name = "default"

    capacity {
      default = var.min_capacity
      minimum = var.min_capacity
      maximum = var.max_capacity
    }

    rule {
      metric_trigger {
        metric_name        = "CpuPercentage"
        metric_resource_id = azurerm_service_plan.asp[0].id
        time_grain         = "PT1M"
        statistic          = "Average"
        time_window        = "PT5M"
        time_aggregation   = "Average"
        operator           = "GreaterThan"
        threshold          = 70
      }

      scale_action {
        direction = "Increase"
        type      = "ChangeCount"
        value     = "1"
        cooldown  = "PT5M"
      }
    }

    rule {
      metric_trigger {
        metric_name        = "CpuPercentage"
        metric_resource_id = azurerm_service_plan.asp[0].id
        time_grain         = "PT1M"
        statistic          = "Average"
        time_window        = "PT5M"
        time_aggregation   = "Average"
        operator           = "LessThan"
        threshold          = 30
      }

      scale_action {
        direction = "Decrease"
        type      = "ChangeCount"
        value     = "1"
        cooldown  = "PT10M"
      }
    }
  }
}

# AWS: ECS Fargate
resource "aws_ecs_cluster" "cluster" {
  count = var.cloud == "aws" ? 1 : 0
  name = "${var.prefix}-ecs"
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
  tags = { Name = "${var.prefix}-ecs" }
}

resource "aws_ecs_task_definition" "task" {
  count = var.cloud == "aws" ? 1 : 0
  family                   = "${var.prefix}-api"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.aws_cpu
  memory                   = var.aws_memory
  execution_role_arn       = aws_iam_role.ecs_execution[0].arn
  task_role_arn            = aws_iam_role.ecs_task[0].arn

  container_definitions = jsonencode([
    {
      name  = "${var.prefix}-api"
      image = "${var.docker_image}:${var.image_tag}"
      essential = true
      portMappings = [
        {
          containerPort = 8000
          hostPort      = 8000
          protocol      = "tcp"
        }
      ]
      environment = [
        { name = "DATABASE_URL",    value = "postgresql://${var.db_user}:${var.db_password}@${var.db_host}:${var.db_port}/${var.db_name}" },
        { name = "APP_NAME",        value = "Vector DB API" },
        { name = "APP_VERSION",     value = "1.0.0" },
        { name = "DEBUG",           value = "false" },
        { name = "DEFAULT_M",       value = "16" },
        { name = "DEFAULT_EF_CONSTRUCTION", value = "200" },
        { name = "DEFAULT_N_CLUSTERS",      value = "100" },
        { name = "UVICORN_WORKERS", value = "4" },
        { name = "API_KEY",         value = var.api_key }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.api[0].name
          "awslogs-region"        = var.location
          "awslogs-stream-prefix" = "ecs"
        }
      }
      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
        interval    = 30
        timeout     = 10
        retries     = 3
        startPeriod = 10
      }
    }
  ])

  tags = { Name = "${var.prefix}-task" }
}

resource "aws_ecs_service" "service" {
  count = var.cloud == "aws" ? 1 : 0
  name            = "${var.prefix}-svc"
  cluster         = aws_ecs_cluster.cluster[0].id
  task_definition = aws_ecs_task_definition.task[0].arn
  desired_count   = var.min_capacity
  launch_type     = "FARGATE"

  network_configuration {
    subnets         = [var.subnet_id]
    security_groups = [var.security_group_id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.api[0].arn
    container_name   = "${var.prefix}-api"
    container_port   = 8000
  }

  health_check_grace_period_seconds = 60
  depends_on = [aws_lb_listener.http]
}

resource "aws_lb" "alb" {
  count = var.cloud == "aws" ? 1 : 0
  name               = "${var.prefix}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [var.security_group_id]
  subnets            = [var.subnet_id]
  tags = { Name = "${var.prefix}-alb" }
}

resource "aws_lb_target_group" "api" {
  count = var.cloud == "aws" ? 1 : 0
  name        = "${var.prefix}-tg"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = var.vpc_id
  target_type = "ip"

  health_check {
    enabled             = true
    healthy_threshold   = 2
    unhealthy_threshold = 3
    interval            = 30
    timeout             = 10
    path                = "/health"
    port                = "traffic-port"
  }
  tags = { Name = "${var.prefix}-tg" }
}

resource "aws_lb_listener" "http" {
  count = var.cloud == "aws" ? 1 : 0
  load_balancer_arn = aws_lb.alb[0].arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api[0].arn
  }
}

# Auto-scaling for ECS
resource "aws_appautoscaling_target" "ecs" {
  count = var.cloud == "aws" ? 1 : 0
  max_capacity       = var.max_capacity
  min_capacity       = var.min_capacity
  resource_id        = "service/${aws_ecs_cluster.cluster[0].name}/${aws_ecs_service.service[0].name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "cpu" {
  count = var.cloud == "aws" ? 1 : 0
  name               = "${var.prefix}-cpu-autoscale"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs[0].resource_id
  scalable_dimension = aws_appautoscaling_target.ecs[0].scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs[0].service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    target_value = 70
  }
}

resource "aws_appautoscaling_policy" "memory" {
  count = var.cloud == "aws" ? 1 : 0
  name               = "${var.prefix}-mem-autoscale"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs[0].resource_id
  scalable_dimension = aws_appautoscaling_target.ecs[0].scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs[0].service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageMemoryUtilization"
    }
    target_value = 70
  }
}

# IAM roles for ECS
resource "aws_iam_role" "ecs_execution" {
  count = var.cloud == "aws" ? 1 : 0
  name = "${var.prefix}-ecs-execution"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_execution" {
  count = var.cloud == "aws" ? 1 : 0
  role       = aws_iam_role.ecs_execution[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "ecs_task" {
  count = var.cloud == "aws" ? 1 : 0
  name = "${var.prefix}-ecs-task"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_cloudwatch_log_group" "api" {
  count = var.cloud == "aws" ? 1 : 0
  name              = "/ecs/${var.prefix}-api"
  retention_in_days = 30
  tags = { Name = "${var.prefix}-logs" }
}
