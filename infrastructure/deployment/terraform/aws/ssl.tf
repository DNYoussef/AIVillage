# SSL/TLS Certificate Configuration for AIVillage

# Route53 Hosted Zone (if domain is provided)
resource "aws_route53_zone" "aivillage" {
  count = var.domain_name != "" ? 1 : 0
  name  = var.domain_name

  tags = {
    Name = "aivillage-zone-${var.environment}"
  }
}

# ACM Certificate for ALB and CloudFront
resource "aws_acm_certificate" "aivillage" {
  domain_name       = var.domain_name != "" ? var.domain_name : "*.elb.amazonaws.com"
  validation_method = var.domain_name != "" ? "DNS" : "EMAIL"

  subject_alternative_names = var.domain_name != "" ? [
    "*.${var.domain_name}",
    "www.${var.domain_name}"
  ] : []

  lifecycle {
    create_before_destroy = true
  }

  tags = {
    Name = "aivillage-certificate-${var.environment}"
  }
}

# Route53 Certificate Validation Records
resource "aws_route53_record" "aivillage_validation" {
  for_each = var.domain_name != "" ? {
    for dvo in aws_acm_certificate.aivillage.domain_validation_options : dvo.domain_name => {
      name   = dvo.resource_record_name
      record = dvo.resource_record_value
      type   = dvo.resource_record_type
    }
  } : {}

  allow_overwrite = true
  name            = each.value.name
  records         = [each.value.record]
  ttl             = 60
  type            = each.value.type
  zone_id         = aws_route53_zone.aivillage[0].zone_id
}

# Certificate Validation
resource "aws_acm_certificate_validation" "aivillage" {
  certificate_arn         = aws_acm_certificate.aivillage.arn
  validation_record_fqdns = var.domain_name != "" ? [for record in aws_route53_record.aivillage_validation : record.fqdn] : null

  timeouts {
    create = "5m"
  }
}

# Route53 A Record for ALB
resource "aws_route53_record" "aivillage_alb" {
  count   = var.domain_name != "" ? 1 : 0
  zone_id = aws_route53_zone.aivillage[0].zone_id
  name    = "api.${var.domain_name}"
  type    = "A"

  alias {
    name                   = aws_lb.aivillage.dns_name
    zone_id                = aws_lb.aivillage.zone_id
    evaluate_target_health = true
  }
}

# Route53 A Record for CloudFront
resource "aws_route53_record" "aivillage_cloudfront" {
  count   = var.domain_name != "" ? 1 : 0
  zone_id = aws_route53_zone.aivillage[0].zone_id
  name    = var.domain_name
  type    = "A"

  alias {
    name                   = aws_cloudfront_distribution.aivillage.domain_name
    zone_id                = aws_cloudfront_distribution.aivillage.hosted_zone_id
    evaluate_target_health = false
  }
}

# Route53 AAAA Record for CloudFront (IPv6)
resource "aws_route53_record" "aivillage_cloudfront_ipv6" {
  count   = var.domain_name != "" ? 1 : 0
  zone_id = aws_route53_zone.aivillage[0].zone_id
  name    = var.domain_name
  type    = "AAAA"

  alias {
    name                   = aws_cloudfront_distribution.aivillage.domain_name
    zone_id                = aws_cloudfront_distribution.aivillage.hosted_zone_id
    evaluate_target_health = false
  }
}

# Route53 CNAME Record for www subdomain
resource "aws_route53_record" "aivillage_www" {
  count   = var.domain_name != "" ? 1 : 0
  zone_id = aws_route53_zone.aivillage[0].zone_id
  name    = "www.${var.domain_name}"
  type    = "CNAME"
  ttl     = 300
  records = [aws_cloudfront_distribution.aivillage.domain_name]
}

# Route53 Health Check for ALB
resource "aws_route53_health_check" "aivillage_alb" {
  count                           = var.domain_name != "" ? 1 : 0
  fqdn                           = aws_lb.aivillage.dns_name
  port                           = 443
  type                           = "HTTPS_STR_MATCH"
  resource_path                  = "/health"
  failure_threshold              = "3"
  request_interval               = "30"
  search_string                  = "healthy"
  cloudwatch_alarm_region        = var.aws_region
  cloudwatch_alarm_name          = aws_cloudwatch_metric_alarm.route53_health_check[0].alarm_name
  insufficient_data_health_status = "Failure"

  tags = {
    Name = "aivillage-alb-health-check-${var.environment}"
  }
}

# CloudWatch Alarm for Route53 Health Check
resource "aws_cloudwatch_metric_alarm" "route53_health_check" {
  count               = var.domain_name != "" ? 1 : 0
  alarm_name          = "aivillage-route53-health-check-${var.environment}"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "HealthCheckStatus"
  namespace           = "AWS/Route53"
  period              = "60"
  statistic           = "Minimum"
  threshold           = "1"
  alarm_description   = "This metric monitors Route53 health check status"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  dimensions = {
    HealthCheckId = aws_route53_health_check.aivillage_alb[0].id
  }

  tags = {
    Name = "aivillage-route53-health-check-alarm-${var.environment}"
  }
}

# Security Headers CloudFront Function
resource "local_file" "security_headers_function" {
  content = <<-EOT
function handler(event) {
    var response = event.response;
    var headers = response.headers;

    // Set security headers
    headers['strict-transport-security'] = { value: 'max-age=63072000; includeSubDomains; preload' };
    headers['content-security-policy'] = {
        value: "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' data:; connect-src 'self' wss: https:; frame-ancestors 'none';"
    };
    headers['x-content-type-options'] = { value: 'nosniff' };
    headers['x-frame-options'] = { value: 'DENY' };
    headers['x-xss-protection'] = { value: '1; mode=block' };
    headers['referrer-policy'] = { value: 'strict-origin-when-cross-origin' };
    headers['permissions-policy'] = {
        value: 'camera=(), microphone=(), geolocation=(), interest-cohort=()'
    };

    // Remove server headers that might leak information
    delete headers['server'];
    delete headers['x-powered-by'];

    return response;
}
EOT
  filename = "${path.module}/functions/security-headers.js"
}

# SSL Configuration for ALB Listener
resource "aws_lb_listener_certificate" "aivillage_additional" {
  count           = var.domain_name != "" && length(var.additional_domains) > 0 ? 1 : 0
  listener_arn    = aws_lb_listener.https.arn
  certificate_arn = aws_acm_certificate.additional[0].arn
}

# Additional ACM Certificate for multiple domains
resource "aws_acm_certificate" "additional" {
  count                     = var.domain_name != "" && length(var.additional_domains) > 0 ? 1 : 0
  domain_name               = var.additional_domains[0]
  subject_alternative_names = slice(var.additional_domains, 1, length(var.additional_domains))
  validation_method         = "DNS"

  lifecycle {
    create_before_destroy = true
  }

  tags = {
    Name = "aivillage-additional-certificate-${var.environment}"
  }
}