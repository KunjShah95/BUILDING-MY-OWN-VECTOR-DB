variable "cloud" {
  description = "Target cloud provider (azure or aws)"
  type        = string
  validation {
    condition     = contains(["azure", "aws"], var.cloud)
    error_message = "cloud must be either 'azure' or 'aws'."
  }
}

variable "prefix" {
  description = "Resource name prefix"
  type        = string
}

variable "location" {
  description = "Azure region / AWS region"
  type        = string
}

variable "subnet_id" {
  description = "Subnet ID for private endpoint / subnet group"
  type        = string
}

variable "vnet_id" {
  description = "VNet ID for DNS zone link (required for Azure)"
  type        = string
  default     = ""
}

variable "security_group_id" {
  description = "Security group ID (required for AWS)"
  type        = string
  default     = ""
}

variable "resource_group_name" {
  description = "Azure resource group name"
  type        = string
  default     = ""
}

variable "db_user" {
  description = "Database administrator username"
  type        = string
  default     = "vector_user"
}

variable "db_password" {
  description = "Database administrator password"
  type        = string
  sensitive   = true
}

variable "db_name" {
  description = "Database name"
  type        = string
  default     = "vector_db"
}

variable "pg_version" {
  description = "PostgreSQL version"
  type        = string
  default     = "15"
}

variable "storage_size_gb" {
  description = "Storage size in GB"
  type        = number
  default     = 100
}

variable "backup_retention_days" {
  description = "Backup retention period in days"
  type        = number
  default     = 30
}

variable "geo_redundant_backup" {
  description = "Enable geo-redundant backup (Azure only)"
  type        = bool
  default     = false
}

variable "azure_pg_sku" {
  description = "Azure PostgreSQL Flexible Server SKU"
  type        = string
  default     = "GP_Standard_D2ds_v4"
}

variable "aws_rds_instance_class" {
  description = "AWS RDS instance class"
  type        = string
  default     = "db.t3.medium"
}
