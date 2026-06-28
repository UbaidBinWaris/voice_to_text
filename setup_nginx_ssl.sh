#!/bin/bash
echo "=================================================="
echo " Setting up Nginx & SSL for ai.ubaidbinwaris.com"
echo "=================================================="

# Install Nginx and Certbot
apt-get update && apt-get install -y nginx certbot python3-certbot-nginx

# Copy Nginx config
cp nginx_ai.conf /etc/nginx/sites-available/ai.ubaidbinwaris.com

# Enable site
ln -sf /etc/nginx/sites-available/ai.ubaidbinwaris.com /etc/nginx/sites-enabled/

# Test configuration and reload Nginx
nginx -t && systemctl reload nginx || service nginx reload

echo "=================================================="
echo " 🔒 Generating SSL Certificate with Certbot..."
echo "=================================================="
certbot --nginx -d ai.ubaidbinwaris.com --non-interactive --agree-tos -m admin@ubaidbinwaris.com --redirect

echo "=================================================="
echo " ✅ Nginx & SSL Setup Complete!"
echo " 👉 Visit https://ai.ubaidbinwaris.com"
echo "=================================================="
