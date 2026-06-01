# Azure VNet + subnet
resource "azurerm_virtual_network" "vnet" {
  count = var.cloud == "azure" ? 1 : 0
  name                = "${var.prefix}-vnet"
  resource_group_name = var.resource_group_name
  location           = var.location
  address_space      = var.vnet_cidr
}

resource "azurerm_subnet" "subnet" {
  count = var.cloud == "azure" ? 1 : 0
  name                 = "${var.prefix}-subnet"
  resource_group_name  = var.resource_group_name
  virtual_network_name = azurerm_virtual_network.vnet[0].name
  address_prefixes     = var.subnet_cidr
}

resource "azurerm_network_security_group" "sg" {
  count = var.cloud == "azure" ? 1 : 0
  name                = "${var.prefix}-nsg"
  resource_group_name = var.resource_group_name
  location           = var.location

  security_rule {
    name                       = "allow-api"
    priority                   = 100
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "8000"
    source_address_prefixes    = var.allowed_cidrs
    destination_address_prefix = "*"
  }

  security_rule {
    name                       = "allow-db"
    priority                   = 110
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "5432"
    source_address_prefix      = var.vnet_cidr[0]
    destination_address_prefix = "*"
  }
}

resource "azurerm_subnet_network_security_group_association" "sg_assoc" {
  count = var.cloud == "azure" ? 1 : 0
  subnet_id                 = azurerm_subnet.subnet[0].id
  network_security_group_id = azurerm_network_security_group.sg[0].id
}

# AWS VPC + subnet
resource "aws_vpc" "vpc" {
  count = var.cloud == "aws" ? 1 : 0
  cidr_block = var.vnet_cidr[0]
  enable_dns_hostnames = true
  enable_dns_support   = true
  tags = { Name = "${var.prefix}-vpc" }
}

resource "aws_subnet" "subnet" {
  count = var.cloud == "aws" ? 1 : 0
  vpc_id     = aws_vpc.vpc[0].id
  cidr_block = var.subnet_cidr[0]
  map_public_ip_on_launch = false
  tags = { Name = "${var.prefix}-subnet" }
}

resource "aws_internet_gateway" "igw" {
  count = var.cloud == "aws" ? 1 : 0
  vpc_id = aws_vpc.vpc[0].id
  tags = { Name = "${var.prefix}-igw" }
}

resource "aws_route_table" "rt" {
  count = var.cloud == "aws" ? 1 : 0
  vpc_id = aws_vpc.vpc[0].id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw[0].id
  }
  tags = { Name = "${var.prefix}-rt" }
}

resource "aws_route_table_association" "rta" {
  count = var.cloud == "aws" ? 1 : 0
  subnet_id      = aws_subnet.subnet[0].id
  route_table_id = aws_route_table.rt[0].id
}

resource "aws_security_group" "sg" {
  count = var.cloud == "aws" ? 1 : 0
  name        = "${var.prefix}-sg"
  description = "Security group for ${var.prefix}"
  vpc_id      = aws_vpc.vpc[0].id

  ingress {
    description = "API traffic"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidrs
  }

  ingress {
    description = "PostgreSQL"
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [var.vnet_cidr[0]]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.prefix}-sg" }
}
