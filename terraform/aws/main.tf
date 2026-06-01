terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {}
}

provider "aws" {
  region = var.location
}

module "networking" {
  source = "../modules/networking"
  cloud  = "aws"
  prefix = var.prefix
  location       = var.location
  vnet_cidr      = var.vpc_cidr
  subnet_cidr    = var.subnet_cidr
  allowed_cidrs  = var.allowed_cidrs
}

module "database" {
  source = "../modules/database"
  cloud  = "aws"
  prefix = var.prefix
  location            = var.location
  subnet_id           = module.networking.subnet_id
  security_group_id   = module.networking.security_group_id
  db_user             = var.db_user
  db_password         = var.db_password
  db_name             = var.db_name
  pg_version          = var.pg_version
  storage_size_gb     = var.db_storage_size_gb
  backup_retention_days = var.backup_retention_days
  aws_rds_instance_class = var.aws_rds_instance_class
}

module "compute" {
  source = "../modules/compute"
  cloud  = "aws"
  prefix = var.prefix
  location            = var.location
  subnet_id           = module.networking.subnet_id
  vpc_id              = module.networking.vpc_id
  security_group_id   = module.networking.security_group_id
  db_host             = module.database.db_host
  db_port             = module.database.db_port
  db_user             = var.db_user
  db_password         = var.db_password
  db_name             = var.db_name
  api_key             = var.api_key
  docker_image        = var.docker_image
  image_tag           = var.image_tag
  aws_cpu             = var.aws_cpu
  aws_memory          = var.aws_memory
  min_capacity        = var.min_capacity
  max_capacity        = var.max_capacity
}
