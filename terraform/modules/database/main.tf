# Azure: PostgreSQL Flexible Server
resource "azurerm_postgresql_flexible_server" "pg" {
  count = var.cloud == "azure" ? 1 : 0
  name                   = "${var.prefix}-pg"
  resource_group_name    = var.resource_group_name
  location              = var.location
  version               = var.pg_version
  administrator_login   = var.db_user
  administrator_password = var.db_password
  storage_mb            = var.storage_size_gb * 1024
  sku_name              = var.azure_pg_sku
  backup_retention_days = var.backup_retention_days
  geo_redundant_backup_enabled = var.geo_redundant_backup

  depends_on = [azurerm_private_endpoint.pg]

  tags = { Name = "${var.prefix}-pg" }
}

resource "azurerm_private_endpoint" "pg" {
  count = var.cloud == "azure" ? 1 : 0
  name                = "${var.prefix}-pg-pe"
  resource_group_name = var.resource_group_name
  location           = var.location
  subnet_id          = var.subnet_id

  private_service_connection {
    name                           = "${var.prefix}-pg-psc"
    private_connection_resource_id = azurerm_postgresql_flexible_server.pg[0].id
    subresource_names              = ["postgresqlServer"]
    is_manual_connection           = false
  }

  private_dns_zone_group {
    name                 = "${var.prefix}-pg-dns"
    private_dns_zone_ids = [azurerm_private_dns_zone.pg[0].id]
  }
}

resource "azurerm_private_dns_zone" "pg" {
  count = var.cloud == "azure" ? 1 : 0
  name                = "privatelink.postgres.database.azure.com"
  resource_group_name = var.resource_group_name
}

resource "azurerm_private_dns_zone_virtual_network_link" "pg" {
  count = var.cloud == "azure" ? 1 : 0
  name                  = "${var.prefix}-pg-dns-link"
  resource_group_name   = var.resource_group_name
  private_dns_zone_name = azurerm_private_dns_zone.pg[0].name
  virtual_network_id    = var.vnet_id
  registration_enabled  = false
}

resource "azurerm_postgresql_flexible_server_database" "db" {
  count = var.cloud == "azure" ? 1 : 0
  name      = var.db_name
  server_id = azurerm_postgresql_flexible_server.pg[0].id
  collation = "en_US.UTF8"
  charset   = "UTF8"
}

resource "azurerm_postgresql_flexible_server_firewall_rule" "allow_azure" {
  count = var.cloud == "azure" ? 1 : 0
  name             = "${var.prefix}-allow-azure"
  server_id        = azurerm_postgresql_flexible_server.pg[0].id
  start_ip_address = "0.0.0.0"
  end_ip_address   = "0.0.0.0"
}

# AWS: RDS PostgreSQL
resource "aws_db_subnet_group" "pg" {
  count = var.cloud == "aws" ? 1 : 0
  name       = "${var.prefix}-pg-subnet"
  subnet_ids = [var.subnet_id]
  tags = { Name = "${var.prefix}-pg-subnet" }
}

resource "aws_db_instance" "pg" {
  count = var.cloud == "aws" ? 1 : 0
  identifier = "${var.prefix}-pg"
  engine         = "postgres"
  engine_version = var.pg_version
  instance_class = var.aws_rds_instance_class

  db_name  = var.db_name
  username = var.db_user
  password = var.db_password

  allocated_storage     = var.storage_size_gb
  max_allocated_storage = var.storage_size_gb * 2
  storage_type          = "gp3"
  storage_encrypted     = true

  db_subnet_group_name   = aws_db_subnet_group.pg[0].name
  vpc_security_group_ids = [var.security_group_id]

  backup_retention_period = var.backup_retention_days
  backup_window           = "03:00-04:00"
  maintenance_window      = "sun:04:00-sun:05:00"

  skip_final_snapshot     = false
  final_snapshot_identifier = "${var.prefix}-pg-final-${formatdate("YYYYMMDD-HHmmss", timestamp())}"

  auto_minor_version_upgrade = true
  deletion_protection        = true

  enabled_cloudwatch_logs_exports = ["postgresql"]

  tags = { Name = "${var.prefix}-pg" }
}
