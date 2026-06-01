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
  description = "Azure region / AWS availability zone"
  type        = string
}

variable "resource_group_name" {
  description = "Azure resource group name (unused for AWS)"
  type        = string
  default     = ""
}

variable "vnet_cidr" {
  description = "CIDR blocks for VNet/VPC"
  type        = list(string)
  default     = ["10.0.0.0/16"]
}

variable "subnet_cidr" {
  description = "CIDR blocks for subnet"
  type        = list(string)
  default     = ["10.0.1.0/24"]
}

variable "allowed_cidrs" {
  description = "CIDR blocks allowed to access the API"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}
