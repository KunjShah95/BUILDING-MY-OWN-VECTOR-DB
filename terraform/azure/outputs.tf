output "resource_group_name" {
  description = "Azure resource group name"
  value       = azurerm_resource_group.main.name
}

output "app_url" {
  description = "Application URL"
  value       = module.compute.app_url
}

output "db_host" {
  description = "Database hostname"
  value       = module.database.db_host
  sensitive   = true
}

output "vnet_id" {
  description = "Virtual Network ID"
  value       = module.networking.vnet_id
}

output "subnet_id" {
  description = "Subnet ID"
  value       = module.networking.subnet_id
}
