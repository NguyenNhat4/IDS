# Các loại tấn công trong Network Security

## Tổng quan

Trong project IDS này, chúng ta tập trung vào 4 nhóm tấn công chính từ NSL-KDD dataset:

1. **DoS (Denial of Service)** - Tấn công từ chối dịch vụ
2. **Probe** - Tấn công thăm dò
3. **R2L (Remote to Local)** - Truy cập trái phép từ xa
4. **U2R (User to Root)** - Leo thang đặc quyền

## 1. DoS (Denial of Service) Attacks

### Khái niệm
Tấn công **DoS** nhằm làm cho hệ thống, mạng, hoặc dịch vụ không thể phục vụ người dùng hợp lệ bằng cách làm quá tải tài nguyên.

### Các loại DoS trong NSL-KDD

#### a) **SYN Flood**
```
Attacker → SYN → Server
Server → SYN-ACK → Attacker (không nhận)
Server chờ đợi ACK (timeout)
→ Lặp lại hàng ngàn lần → Server hết tài nguyên
```

**Cách hoạt động:**
1. Attacker gửi hàng loạt SYN packets
2. Server mở connection và chờ ACK
3. Attacker không bao giờ gửi ACK
4. Server hết connection slots

**Đặc điểm nhận dạng:**
- Số lượng SYN packets cực lớn
- Không có ACK tương ứng
- Half-open connections tăng đột biến
- Source IP thường bị spoof

#### b) **Ping of Death**
```
Attacker gửi ICMP packet size > 65,535 bytes
→ Server crash khi reassemble
```

**Đặc điểm:**
- ICMP packet size bất thường
- Fragmented packets

#### c) **Smurf Attack**
```
Attacker → ICMP Echo Request (IP giả = victim) → Broadcast
→ Tất cả máy trong mạng reply về victim
→ Victim bị ngập lụt
```

**Đặc điểm:**
- ICMP Echo Replies từ nhiều nguồn
- Đến cùng một destination

#### d) **Teardrop**
Gửi fragmented packets với offset lỗi, làm crash hệ thống khi reassemble.

### Ví dụ real-world

**GitHub DDoS Attack (2018)**
- 1.35 Tbps traffic
- Sử dụng memcached servers
- Amplification factor: 50,000x

### Phòng chống DoS

1. **Rate Limiting**: Giới hạn requests/second
2. **SYN Cookies**: Không lưu state cho SYN packets
3. **Filtering**: Drop packets từ suspicious sources
4. **CDN**: Phân tán traffic (Cloudflare, Akamai)

---

## 2. Probe Attacks (Reconnaissance)

### Khái niệm
**Probe** là các tấn công thăm dò/do thám để thu thập thông tin về hệ thống trước khi tấn công thực sự.

### Các loại Probe

#### a) **Port Scanning**
Quét các port đang mở để tìm điểm yếu.

**Nmap TCP SYN Scan:**
```bash
nmap -sS target.com
```

```
Attacker → SYN (port 80) → Server
Server → SYN-ACK → Attacker (port mở)
hoặc
Server → RST → Attacker (port đóng)
```

**Các loại port scan:**

1. **TCP Connect Scan**: Full 3-way handshake
2. **SYN Scan** (Stealth scan): Không complete connection
3. **FIN Scan**: Gửi FIN packet
4. **XMAS Scan**: Gửi FIN+PSH+URG flags
5. **NULL Scan**: Không có flags

**Đặc điểm nhận dạng:**
- Quét tuần tự nhiều ports
- Connection attempts trong thời gian ngắn
- Không có data transfer sau khi connect

#### b) **Network Mapping**
Xác định topology mạng, routers, firewalls.

```bash
# Traceroute
tracert target.com

# Ping sweep
nmap -sn 192.168.1.0/24
```

**Đặc điểm:**
- ICMP packets đến nhiều hosts
- TTL manipulation
- Traceroute patterns

#### c) **OS Fingerprinting**
Xác định hệ điều hành của target.

**Passive fingerprinting:**
- Phân tích TTL, Window Size, TCP Options
- Không gửi packets chủ động

**Active fingerprinting:**
```bash
nmap -O target.com
```

**Đặc điểm:**
- Unusual TCP options
- Rare packet types
- Specific probe sequences

#### d) **Vulnerability Scanning**
Tìm kiếm lỗ hổng bảo mật.

```bash
# Nessus, OpenVAS, Nikto
nikto -h http://target.com
```

**Đặc điểm:**
- HTTP requests đến nhiều paths
- Tìm kiếm known vulnerabilities
- Login page probing

### Ví dụ thực tế

**Shodan - "The search engine for hackers"**
- Index các thiết bị IoT, servers exposed
- Tìm devices với default passwords
- Không phải scanning trực tiếp, nhưng cùng mục đích

### Phòng chống Probe

1. **Firewall Rules**: Block suspicious scanning
2. **IDS Alerts**: Phát hiện port scan patterns
3. **Honeypots**: Lừa attackers
4. **Rate Limiting**: Giới hạn connection attempts

---

## 3. R2L (Remote to Local) Attacks

### Khái niệm
**R2L** là tấn công mà attacker không có account trên hệ thống, nhưng cố gắng truy cập từ xa.

### Các loại R2L

#### a) **Brute Force Login**
Thử nhiều password combinations.

```python
# Pseudo-code
passwords = ["123456", "password", "admin", ...]
for pwd in passwords:
    try_login(username="admin", password=pwd)
    if success:
        break
```

**Đặc điểm:**
- Nhiều login attempts liên tiếp
- Cùng username, khác password
- Failed login rate cao

**Ví dụ:**
```
POST /login
username=admin&password=123456  → 401
POST /login
username=admin&password=password → 401
POST /login
username=admin&password=admin123 → 200 ✓
```

#### b) **Dictionary Attack**
Sử dụng wordlist thông dụng.

```bash
# Hydra
hydra -l admin -P /usr/share/wordlists/rockyou.txt ssh://target.com
```

**Wordlists phổ biến:**
- rockyou.txt (14M passwords)
- SecLists
- CrackStation

#### c) **Guess Password**
Đoán password dựa vào context.

Ví dụ:
- Username: `john` → Password: `john123`, `john@2024`
- Company: `ABC Corp` → Password: `abc123`, `ABC@2024`

#### d) **FTP/SSH Brute Force**
```bash
# Medusa
medusa -h target.com -u admin -P passwords.txt -M ssh

# Hydra
hydra -l root -P pass.txt ftp://target.com
```

#### e) **SQL Injection** (Dạng R2L)
```sql
-- Bypass login
username: admin' OR '1'='1
password: anything

-- Kết quả query:
SELECT * FROM users
WHERE username='admin' OR '1'='1' AND password='...'
-- '1'='1' luôn đúng → login thành công!
```

**Ví dụ chi tiết:**
```php
// Vulnerable code
$query = "SELECT * FROM users WHERE username='$username' AND password='$password'";

// Attack
username: admin' --
password: (bỏ trống)

// Query trở thành:
SELECT * FROM users WHERE username='admin' -- AND password=''
// -- là comment, bỏ qua password check!
```

#### f) **Command Injection**
```bash
# Vulnerable: ping command
ping -c 4 192.168.1.1

# Attack payload:
192.168.1.1; cat /etc/passwd
192.168.1.1 && whoami
192.168.1.1 | ls -la
```

**Web example:**
```
GET /ping?host=192.168.1.1;cat /etc/passwd
```

### Phòng chống R2L

1. **Strong Password Policy**
   - Min 12 characters
   - Mix uppercase, lowercase, numbers, symbols
   - No common passwords

2. **Account Lockout**
   - Lock after 5 failed attempts
   - Temporary lockout (15-30 minutes)

3. **CAPTCHA**
   - Prevent automated brute force

4. **Two-Factor Authentication (2FA)**
   - OTP, SMS, Authenticator apps

5. **Input Validation**
   - Sanitize user inputs
   - Parameterized queries (SQL Injection prevention)
   - Whitelist validation

6. **Rate Limiting**
   - Limit login attempts per IP
   - Slow down brute force

---

## 4. U2R (User to Root) Attacks

### Khái niệm
**U2R** là tấn công mà attacker đã có account user bình thường, nhưng cố gắng leo thang thành root/administrator.

### Các loại U2R

#### a) **Buffer Overflow**
Ghi đè bộ nhớ để execute malicious code.

```c
// Vulnerable code
void vulnerable_function(char *input) {
    char buffer[64];
    strcpy(buffer, input);  // No bounds checking!
}

// Attack:
// Input > 64 bytes → overflow → overwrite return address
// → Execute shellcode
```

**Ví dụ:**
```
Normal: [buffer][return_address]
Attack: [buffer + overflow + shellcode][shellcode_address]
```

#### b) **Rootkit**
Software ẩn mình và cung cấp quyền root.

**Types:**
1. **User-mode rootkit**: Thay đổi binaries (ls, ps, netstat)
2. **Kernel-mode rootkit**: Modify kernel modules
3. **Bootkit**: Infect bootloader

**Detection:**
```bash
# Rootkit Hunter
rkhunter --check

# Chkrootkit
chkrootkit
```

#### c) **Loadmodule**
Load kernel module độc hại.

```bash
# Attacker
insmod malicious_module.ko

# Module này có thể:
# - Hook system calls
# - Hide processes
# - Grant root access
```

#### d) **Perl/Python Exploits**
Sử dụng scripts để exploit vulnerabilities.

```python
# Exploit SUID binary
import os
os.system("cp /bin/sh /tmp/rootshell")
os.system("chmod 4755 /tmp/rootshell")
# /tmp/rootshell → root shell!
```

#### e) **Privilege Escalation via SUID**
```bash
# Find SUID binaries
find / -perm -4000 2>/dev/null

# Exploit misconfigured SUID
# Example: /usr/bin/find (if SUID)
/usr/bin/find /etc/passwd -exec whoami \;  # Runs as root!
```

#### f) **Exploiting Sudo Misconfiguration**
```bash
# /etc/sudoers
user ALL=(ALL) NOPASSWD: /usr/bin/vim

# Exploit:
sudo vim
:!bash  # Escape to shell as root!
```

### Linux Privilege Escalation Checklist

1. **SUID/SGID binaries**
   ```bash
   find / -perm -4000 -type f 2>/dev/null
   ```

2. **Writable /etc/passwd**
   ```bash
   ls -la /etc/passwd
   ```

3. **Sudo rights**
   ```bash
   sudo -l
   ```

4. **Cron jobs**
   ```bash
   cat /etc/crontab
   ```

5. **Kernel exploits**
   ```bash
   uname -a  # Check kernel version
   searchsploit kernel 4.15  # Find exploits
   ```

### Phòng chống U2R

1. **Principle of Least Privilege**
   - Users chỉ có quyền tối thiểu cần thiết

2. **Keep Systems Updated**
   - Patch security vulnerabilities
   - Update kernel, packages

3. **Disable Unnecessary Services**
   - Giảm attack surface

4. **Use SELinux/AppArmor**
   - Mandatory Access Control

5. **Monitor System Changes**
   - File integrity monitoring (AIDE, Tripwire)
   - Audit logs

6. **Secure Coding Practices**
   - Avoid buffer overflows
   - Input validation
   - Use safe functions (strncpy vs strcpy)

---

## Web-specific Attacks (Bonus)

### 1. **Cross-Site Scripting (XSS)**
```html
<!-- Vulnerable page -->
<div>Welcome, <?php echo $_GET['name']; ?></div>

<!-- Attack URL -->
http://site.com/welcome?name=<script>alert(document.cookie)</script>

<!-- Result -->
<div>Welcome, <script>alert(document.cookie)</script></div>
<!-- Script executes! Steals cookies! -->
```

**Types:**
- **Reflected XSS**: Malicious script trong URL
- **Stored XSS**: Script lưu trong database
- **DOM-based XSS**: Client-side manipulation

### 2. **Cross-Site Request Forgery (CSRF)**
```html
<!-- Attacker's page -->
<img src="http://bank.com/transfer?to=attacker&amount=1000">

<!-- Victim visit attacker's page while logged in to bank.com -->
<!-- Request auto executes with victim's cookies! -->
```

### 3. **Directory Traversal**
```
http://site.com/download?file=../../../etc/passwd
http://site.com/image?path=....//....//etc/passwd
```

---

## Attack Signatures trong NSL-KDD

### Feature Patterns

| Attack Type | High values in features | Low values in features |
|-------------|------------------------|------------------------|
| **DoS** | count, srv_count | dst_host_srv_count |
| **Probe** | dst_host_diff_srv_rate | duration |
| **R2L** | num_failed_logins | logged_in (0) |
| **U2R** | num_file_creations, root_shell | - |

### Traffic Patterns

```
Normal Traffic:
- Stable packet rate
- Reasonable packet sizes
- Valid protocols
- Completed connections

DoS Attack:
- Spike in traffic
- Many half-open connections
- High SYN count

Probe Attack:
- Sequential port access
- Low data transfer
- Many short connections

R2L Attack:
- Repeated login failures
- Unusual access times
- Suspicious user agents

U2R Attack:
- Unusual process spawning
- System file modifications
- Privilege changes
```

---

## Lab Exercise Ideas

### 1. **DoS Simulator**
```python
import socket
for i in range(10000):
    s = socket.socket()
    s.connect(('target', 80))
    s.send(b'GET / HTTP/1.1\r\n')
    # Don't close, keep connection open
```

### 2. **Port Scanner**
```python
import socket
for port in range(1, 1000):
    s = socket.socket()
    s.settimeout(0.1)
    result = s.connect_ex(('target', port))
    if result == 0:
        print(f"Port {port} is open")
    s.close()
```

### 3. **SQL Injection Tester**
```python
payloads = [
    "' OR '1'='1",
    "' OR '1'='1' --",
    "admin' --",
]
for payload in payloads:
    test_login(username=payload, password="test")
```

---

## Summary Table

| Attack Type | Goal | Common Techniques | Detection Features |
|-------------|------|-------------------|-------------------|
| **DoS** | Làm crash/overload service | SYN flood, Ping flood, Smurf | High packet rate, half-open connections |
| **Probe** | Thu thập thông tin | Port scan, OS fingerprint | Sequential port access, low duration |
| **R2L** | Truy cập trái phép từ xa | Brute force, SQL injection | Failed logins, unusual patterns |
| **U2R** | Leo thang đặc quyền | Buffer overflow, rootkit | File modifications, privilege changes |

---

**Next**: [03 - How Websites Get Attacked](03-how-websites-get-attacked.md)
