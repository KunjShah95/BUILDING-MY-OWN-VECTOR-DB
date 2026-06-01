variable "prefix" {
  description = "Resource name prefix"
  type        = string
  default     = "vectordb"
}

variable "location" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
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

variable "vpc_cidr" {
  description = "VPC CIDR blocks"
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
  description = "PostgreSQL engine version"
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

variable "aws_rds_instance_class" {
  description = "AWS RDS instance class"
  type        = string
  default     = "db.t3.medium"
}

variable "docker_image" {
  description = "Docker image for the API (ECR or public registry)"
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

variable "aws_cpu" {
  description = "ECS Fargate task CPU (256|512|1024|2048|4096)"
  type        = string
  default     = "512"
}

variable "aws_memory" {
  description = "ECS Fargate task memory in MB"
  type        = string
  default     = "1024"
}

variable "min_capacity" {
  description = "Minimum number of ECS tasks"
  type        = number
  default     = 1
}

variable "max_capacity" {
  description = "Maximum number of ECS tasks"
  type        = number
  default     = 5
}
