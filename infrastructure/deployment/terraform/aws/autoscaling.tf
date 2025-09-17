# Auto Scaling Configuration for AIVillage Fog Computing Platform

# Application Load Balancer
resource "aws_lb" "aivillage" {
  name               = "aivillage-alb-${var.environment}"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id

  enable_deletion_protection = var.enable_deletion_protection

  access_logs {
    bucket  = aws_s3_bucket.alb_logs.bucket
    prefix  = "alb-logs"
    enabled = true
  }

  tags = {
    Name = "aivillage-alb-${var.environment}"
  }
}

# ALB Security Group
resource "aws_security_group" "alb" {
  name_prefix = "aivillage-alb-"
  vpc_id      = aws_vpc.aivillage.id

  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "BetaNet Bridge"
    from_port   = 8080
    to_port     = 8081
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  ingress {
    description = "Python Bridge"
    from_port   = 9876
    to_port     = 9876
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  ingress {
    description = "Monitoring"
    from_port   = 3000
    to_port     = 3000
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "aivillage-alb-sg-${var.environment}"
  }
}

# Target Groups
resource "aws_lb_target_group" "bridge_orchestrator" {
  name     = "aivillage-bridge-${var.environment}"
  port     = 8080
  protocol = "HTTP"
  vpc_id   = aws_vpc.aivillage.id

  health_check {
    enabled             = true
    healthy_threshold   = 2
    unhealthy_threshold = 2
    timeout             = 5
    interval            = 30
    path                = "/health"
    matcher             = "200"
    port                = "traffic-port"
    protocol            = "HTTP"
  }

  tags = {
    Name = "aivillage-bridge-tg-${var.environment}"
  }
}

resource "aws_lb_target_group" "python_bridge" {
  name     = "aivillage-python-${var.environment}"
  port     = 9876
  protocol = "HTTP"
  vpc_id   = aws_vpc.aivillage.id

  health_check {
    enabled             = true
    healthy_threshold   = 2
    unhealthy_threshold = 2
    timeout             = 5
    interval            = 30
    path                = "/health"
    matcher             = "200"
    port                = "traffic-port"
    protocol            = "HTTP"
  }

  tags = {
    Name = "aivillage-python-tg-${var.environment}"
  }
}

resource "aws_lb_target_group" "monitoring" {
  name     = "aivillage-monitoring-${var.environment}"
  port     = 3000
  protocol = "HTTP"
  vpc_id   = aws_vpc.aivillage.id

  health_check {
    enabled             = true
    healthy_threshold   = 2
    unhealthy_threshold = 2
    timeout             = 5
    interval            = 30
    path                = "/api/health"
    matcher             = "200"
    port                = "traffic-port"
    protocol            = "HTTP"
  }

  tags = {
    Name = "aivillage-monitoring-tg-${var.environment}"
  }
}

# ALB Listeners
resource "aws_lb_listener" "https" {
  load_balancer_arn = aws_lb.aivillage.arn
  port              = "443"
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS-1-2-2017-01"
  certificate_arn   = aws_acm_certificate.aivillage.arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.bridge_orchestrator.arn
  }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.aivillage.arn
  port              = "80"
  protocol          = "HTTP"

  default_action {
    type = "redirect"

    redirect {
      port        = "443"
      protocol    = "HTTPS"
      status_code = "HTTP_301"
    }
  }
}

# Listener Rules
resource "aws_lb_listener_rule" "bridge_orchestrator" {
  listener_arn = aws_lb_listener.https.arn
  priority     = 100

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.bridge_orchestrator.arn
  }

  condition {
    path_pattern {
      values = ["/api/*", "/bridge/*"]
    }
  }
}

resource "aws_lb_listener_rule" "python_bridge" {
  listener_arn = aws_lb_listener.https.arn
  priority     = 200

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.python_bridge.arn
  }

  condition {
    path_pattern {
      values = ["/betanet/*", "/python/*"]
    }
  }
}

resource "aws_lb_listener_rule" "monitoring" {
  listener_arn = aws_lb_listener.https.arn
  priority     = 300

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.monitoring.arn
  }

  condition {
    path_pattern {
      values = ["/monitoring/*", "/grafana/*"]
    }
  }
}

# Auto Scaling Groups (for managed node groups, these are automatically created)
# The node groups defined in main.tf will automatically create ASGs

# CloudWatch Alarms for Auto Scaling
resource "aws_cloudwatch_metric_alarm" "high_cpu" {
  alarm_name          = "aivillage-high-cpu-${var.environment}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EKS"
  period              = "120"
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "This metric monitors EKS node CPU utilization"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  dimensions = {
    ClusterName = aws_eks_cluster.aivillage.name
  }

  tags = {
    Name = "aivillage-high-cpu-alarm-${var.environment}"
  }
}

resource "aws_cloudwatch_metric_alarm" "high_memory" {
  alarm_name          = "aivillage-high-memory-${var.environment}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "MemoryUtilization"
  namespace           = "CWAgent"
  period              = "120"
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "This metric monitors EKS node memory utilization"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  dimensions = {
    ClusterName = aws_eks_cluster.aivillage.name
  }

  tags = {
    Name = "aivillage-high-memory-alarm-${var.environment}"
  }
}

resource "aws_cloudwatch_metric_alarm" "alb_target_response_time" {
  alarm_name          = "aivillage-alb-response-time-${var.environment}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "TargetResponseTime"
  namespace           = "AWS/ApplicationELB"
  period              = "60"
  statistic           = "Average"
  threshold           = "0.075"  # 75ms
  alarm_description   = "This metric monitors ALB target response time"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  dimensions = {
    LoadBalancer = aws_lb.aivillage.arn_suffix
  }

  tags = {
    Name = "aivillage-alb-response-time-alarm-${var.environment}"
  }
}

resource "aws_cloudwatch_metric_alarm" "alb_unhealthy_targets" {
  alarm_name          = "aivillage-alb-unhealthy-targets-${var.environment}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "UnHealthyHostCount"
  namespace           = "AWS/ApplicationELB"
  period              = "60"
  statistic           = "Average"
  threshold           = "0"
  alarm_description   = "This metric monitors ALB unhealthy targets"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  dimensions = {
    TargetGroup  = aws_lb_target_group.bridge_orchestrator.arn_suffix
    LoadBalancer = aws_lb.aivillage.arn_suffix
  }

  tags = {
    Name = "aivillage-alb-unhealthy-targets-alarm-${var.environment}"
  }
}

# SNS Topic for Alerts
resource "aws_sns_topic" "alerts" {
  name = "aivillage-alerts-${var.environment}"

  tags = {
    Name = "aivillage-alerts-${var.environment}"
  }
}

resource "aws_sns_topic_subscription" "email_alerts" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# S3 Bucket for ALB Access Logs
resource "aws_s3_bucket" "alb_logs" {
  bucket        = "aivillage-alb-logs-${var.environment}-${random_string.bucket_suffix.result}"
  force_destroy = true

  tags = {
    Name = "aivillage-alb-logs-${var.environment}"
  }
}

resource "aws_s3_bucket_versioning" "alb_logs" {
  bucket = aws_s3_bucket.alb_logs.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "alb_logs" {
  bucket = aws_s3_bucket.alb_logs.id

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

resource "aws_s3_bucket_public_access_block" "alb_logs" {
  bucket = aws_s3_bucket.alb_logs.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_policy" "alb_logs" {
  bucket = aws_s3_bucket.alb_logs.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AWSLogDeliveryWrite"
        Effect = "Allow"
        Principal = {
          AWS = data.aws_elb_service_account.main.arn
        }
        Action   = "s3:PutObject"
        Resource = "${aws_s3_bucket.alb_logs.arn}/alb-logs/AWSLogs/${data.aws_caller_identity.current.account_id}/*"
      },
      {
        Sid    = "AWSLogDeliveryAclCheck"
        Effect = "Allow"
        Principal = {
          AWS = data.aws_elb_service_account.main.arn
        }
        Action   = "s3:GetBucketAcl"
        Resource = aws_s3_bucket.alb_logs.arn
      }
    ]
  })
}

data "aws_elb_service_account" "main" {}

resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}