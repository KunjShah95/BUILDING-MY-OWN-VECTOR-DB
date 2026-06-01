output "db_host" {
  description = "Database hostname"
  value = var.cloud == "azure" ? azurerm_postgresql_flexible_server.pg[0].fqdn : aws_db_instance.pg[0].address
}

output "db_port" {
  description = "Database port"
  value = var.cloud == "azure" ? 5432 : aws_db_instance.pg[0].port
}

output "db_name" {
  description = "Database name"
  value = var.cloud == "azure" ? azurerm_postgresql_flexible_server_database.db[0].name : aws_db_instance.pg[0].db_name
}

output "db_id" {
  description = "Database resource ID"
  value = var.cloud == "azure" ? azurerm_postgresql_flexible_server.pg[0].id : aws_db_instance.pg[0].id
}
