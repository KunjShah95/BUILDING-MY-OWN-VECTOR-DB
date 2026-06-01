terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }

  backend "azurerm" {}
}

provider "azurerm" {
  features {
    key_vault {
      purge_soft_delete_on_destroy = false
    }
  }
}

data "azurerm_client_config" "current" {}

resource "azurerm_resource_group" "main" {
  name     = "${var.prefix}-rg"
  location = var.location
  tags     = var.tags
}

module "networking" {
  source = "../modules/networking"
  cloud  = "azure"
  prefix = var.prefix
  location           = var.location
  resource_group_name = azurerm_resource_group.main.name
  vnet_cidr          = var.vnet_cidr
  subnet_cidr        = var.subnet_cidr
  allowed_cidrs      = var.allowed_cidrs
}

module "database" {
  source = "../modules/database"
  cloud  = "azure"
  prefix = var.prefix
  location                = var.location
  subnet_id               = module.networking.subnet_id
  vnet_id                 = module.networking.vnet_id
  resource_group_name     = azurerm_resource_group.main.name
  db_user                 = var.db_user
  db_password             = var.db_password
  db_name                 = var.db_name
  pg_version              = var.pg_version
  storage_size_gb         = var.db_storage_size_gb
  backup_retention_days   = var.backup_retention_days
  geo_redundant_backup    = var.geo_redundant_backup
  azure_pg_sku            = var.azure_pg_sku
}

module "compute" {
  source = "../modules/compute"
  cloud  = "azure"
  prefix = var.prefix
  location            = var.location
  subnet_id           = module.networking.subnet_id
  resource_group_name = azurerm_resource_group.main.name
  db_host             = module.database.db_host
  db_port             = module.database.db_port
  db_user             = var.db_user
  db_password         = var.db_password
  db_name             = var.db_name
  api_key             = var.api_key
  docker_image        = var.docker_image
  image_tag           = var.image_tag
  azure_sku           = var.azure_asp_sku
  min_capacity        = var.min_capacity
  max_capacity        = var.max_capacity
}
