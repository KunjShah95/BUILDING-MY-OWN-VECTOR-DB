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
  description = "Subnet ID for network integration"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID (required for AWS)"
  type        = string
  default     = ""
}

variable "security_group_id" {
  description = "Security group ID"
  type        = string
  default     = ""
}

variable "resource_group_name" {
  description = "Azure resource group name"
  type        = string
  default     = ""
}

variable "db_host" {
  description = "Database hostname"
  type        = string
}

variable "db_port" {
  description = "Database port"
  type        = string
  default     = "5432"
}

variable "db_user" {
  description = "Database user"
  type        = string
  default     = "vector_user"
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}

variable "db_name" {
  description = "Database name"
  type        = string
  default     = "vector_db"
}

variable "api_key" {
  description = "API key for authentication"
  type        = string
  sensitive   = true
  default     = ""
}

variable "docker_image" {
  description = "Docker image name"
  type        = string
  default     = "vector-db-api"
}

variable "image_tag" {
  description = "Docker image tag"
  type        = string
  default     = "latest"
}

variable "azure_sku" {
  description = "Azure App Service Plan SKU"
  type        = string
  default     = "P1v2"
}

variable "aws_cpu" {
  description = "AWS ECS task CPU units"
  type        = string
  default     = "512"
}

variable "aws_memory" {
  description = "AWS ECS task memory (MB)"
  type        = string
  default     = "1024"
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
