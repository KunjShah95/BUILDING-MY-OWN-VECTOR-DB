variable "prefix" {
  description = "Resource name prefix"
  type        = string
  default     = "vectordb"
}

variable "location" {
  description = "Azure region"
  type        = string
  default     = "eastus"
}

variable "tags" {
  description = "Resource tags"
  type        = map(string)
  default = {
    Environment = "production"
    Project     = "vector-db"
    ManagedBy   = "terraform"
  }
}

variable "vnet_cidr" {
  description = "VNet CIDR blocks"
  type        = list(string)
  default     = ["10.0.0.0/16"]
}

variable "subnet_cidr" {
  description = "Subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.1.0/24"]
}

variable "allowed_cidrs" {
  description = "CIDR blocks allowed to access the API"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "db_user" {
  description = "PostgreSQL admin username"
  type        = string
  default     = "vector_user"
}

variable "db_password" {
  description = "PostgreSQL admin password"
  type        = string
  sensitive   = true
}

variable "db_name" {
  description = "PostgreSQL database name"
  type        = string
  default     = "vector_db"
}

variable "pg_version" {
  description = "PostgreSQL version"
  type        = string
  default     = "15"
}

variable "db_storage_size_gb" {
  description = "Database storage size in GB"
  type        = number
  default     = 100
}

variable "backup_retention_days" {
  description = "Backup retention period in days"
  type        = number
  default     = 30
}

variable "geo_redundant_backup" {
  description = "Enable geo-redundant backup"
  type        = bool
  default     = false
}

variable "azure_pg_sku" {
  description = "Azure PostgreSQL Flexible Server SKU"
  type        = string
  default     = "GP_Standard_D2ds_v4"
}

variable "azure_asp_sku" {
  description = "Azure App Service Plan SKU"
  type        = string
  default     = "P1v2"
}

variable "docker_image" {
  description = "Docker image for the API"
  type        = string
  default     = "vector-db-api"
}

variable "image_tag" {
  description = "Docker image tag"
  type        = string
  default     = "latest"
}

variable "api_key" {
  description = "API key for authentication"
  type        = string
  sensitive   = true
}

variable "min_capacity" {
  description = "Minimum number of instances"
  type        = number
  default     = 1
}

variable "max_capacity" {
  description = "Maximum number of instances"
  type        = number
  default     = 5
}
