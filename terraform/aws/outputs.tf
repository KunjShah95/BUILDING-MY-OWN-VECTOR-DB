output "app_url" {
  description = "Application URL (ALB DNS name)"
  value       = module.compute.app_url
}

output "alb_dns_name" {
  description = "ALB DNS name"
  value       = module.compute.alb_dns_name
}

output "db_host" {
  description = "Database hostname"
  value       = module.database.db_host
  sensitive   = true
}

output "vpc_id" {
  description = "VPC ID"
  value       = module.networking.vpc_id
}

output "subnet_id" {
  description = "Subnet ID"
  value       = module.networking.subnet_id
}

output "ecs_cluster_name" {
  description = "ECS Cluster name"
  value       = module.compute.compute_id
}
