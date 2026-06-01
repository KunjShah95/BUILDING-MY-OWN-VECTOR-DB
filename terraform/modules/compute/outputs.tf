output "app_url" {
  description = "Application URL"
  value = var.cloud == "azure" ? "https://${azurerm_linux_web_app.api[0].default_hostname}" : "http://${aws_lb.alb[0].dns_name}"
}

output "compute_id" {
  description = "Compute resource ID"
  value = var.cloud == "azure" ? azurerm_linux_web_app.api[0].id : aws_ecs_service.service[0].id
}

output "alb_dns_name" {
  description = "ALB DNS name (AWS only)"
  value = var.cloud == "aws" ? aws_lb.alb[0].dns_name : null
}

output "service_plan_id" {
  description = "App Service Plan ID (Azure only)"
  value = var.cloud == "azure" ? azurerm_service_plan.asp[0].id : null
}
