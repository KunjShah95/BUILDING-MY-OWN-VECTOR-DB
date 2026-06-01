output "subnet_id" {
  description = "Subnet ID (Azure or AWS)"
  value = var.cloud == "azure" ? azurerm_subnet.subnet[0].id : aws_subnet.subnet[0].id
}

output "vnet_id" {
  description = "VNet or VPC ID"
  value = var.cloud == "azure" ? azurerm_virtual_network.vnet[0].id : aws_vpc.vpc[0].id
}

output "security_group_id" {
  description = "Security group ID"
  value = var.cloud == "azure" ? azurerm_network_security_group.sg[0].id : aws_security_group.sg[0].id
}

output "vpc_id" {
  description = "VPC ID (AWS only)"
  value = var.cloud == "aws" ? aws_vpc.vpc[0].id : null
}
