from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import random

router = APIRouter()


class SimulatedConnection(BaseModel):
    """Simulated network connection"""
    type: str
    features: Dict[str, Any]
    description: str
    explanation: str


@router.get("/simulate/{attack_type}", response_model=SimulatedConnection)
async def simulate_attack(attack_type: str):
    """
    Generate simulated attack features for demonstration

    Args:
        attack_type: Type of attack to simulate (normal, dos, probe, r2l, u2r)

    Returns:
        Simulated connection with features
    """

    attack_type = attack_type.lower()

    if attack_type == "dos":
        return SimulatedConnection(
            type='DoS',
            description='SYN Flood Attack - Overwhelming server with connection requests',
            explanation='This attack sends numerous SYN packets without completing the handshake, '
                       'exhausting server resources. Notice the high count (511 connections), '
                       'very high serror_rate (99% SYN errors), and zero duration.',
            features={
                'duration': 0,
                'protocol_type': 'tcp',
                'service': 'http',
                'flag': 'S0',  # SYN, no response
                'src_bytes': 0,
                'dst_bytes': 0,
                'count': 511,  # Very high! Max in dataset
                'srv_count': 511,
                'serror_rate': 0.99,  # 99% SYN errors!
                'srv_serror_rate': 0.99,
                'same_srv_rate': 1.0,
                'diff_srv_rate': 0.0,
                'dst_host_count': 255,
                'dst_host_srv_count': 255,
                'dst_host_serror_rate': 0.99,
                'dst_host_srv_serror_rate': 0.99,
            }
        )

    elif attack_type == "probe":
        return SimulatedConnection(
            type='Probe',
            description='Port Scan - Scanning for open ports and services',
            explanation='This attack attempts to discover open ports by connecting to many different services. '
                       'Notice the high diff_srv_rate (90% to different services), high rerror_rate '
                       '(many rejections), and zero duration (no data transfer).',
            features={
                'duration': 0,
                'protocol_type': 'tcp',
                'service': 'private',
                'flag': 'REJ',  # Rejected connections
                'src_bytes': 0,
                'dst_bytes': 0,
                'count': 100,
                'srv_count': 10,
                'serror_rate': 0.0,
                'diff_srv_rate': 0.9,  # 90% to different services (scanning!)
                'same_srv_rate': 0.1,
                'rerror_rate': 0.8,  # 80% rejection rate
                'srv_rerror_rate': 0.8,
                'dst_host_count': 255,
                'dst_host_diff_srv_rate': 0.9,  # Scanning different services
                'dst_host_rerror_rate': 0.8,
            }
        )

    elif attack_type == "r2l":
        return SimulatedConnection(
            type='R2L',
            description='Brute Force Login - Attempting to guess passwords',
            explanation='This attack tries multiple password combinations to gain unauthorized access. '
                       'Notice the high num_failed_logins (5 failed attempts), logged_in=0 '
                       '(not authenticated), and repeated connections to same service (FTP).',
            features={
                'duration': 5,
                'protocol_type': 'tcp',
                'service': 'ftp',
                'flag': 'SF',
                'src_bytes': 100,
                'dst_bytes': 200,
                'num_failed_logins': 5,  # Multiple failed login attempts!
                'logged_in': 0,  # Not logged in
                'count': 10,
                'srv_count': 10,
                'same_srv_rate': 1.0,  # All attempts to same service
                'diff_srv_rate': 0.0,
                'dst_host_count': 10,
                'dst_host_srv_count': 10,
            }
        )

    elif attack_type == "u2r":
        return SimulatedConnection(
            type='U2R',
            description='Buffer Overflow - Attempting privilege escalation',
            explanation='This attack exploits a buffer overflow vulnerability to gain root access. '
                       'Notice logged_in=1 (already authenticated), num_root=1 (root access attempts), '
                       'root_shell=1 (obtained root shell), and num_file_creations (installing backdoor).',
            features={
                'duration': 10,
                'protocol_type': 'tcp',
                'service': 'telnet',
                'flag': 'SF',
                'src_bytes': 500,
                'dst_bytes': 1000,
                'logged_in': 1,  # Already logged in as normal user
                'num_root': 1,  # Attempting root access
                'num_file_creations': 3,  # Creating files (backdoor/rootkit)
                'num_shells': 1,
                'root_shell': 1,  # Got root shell!
                'su_attempted': 1,
                'count': 5,
                'srv_count': 5,
            }
        )

    elif attack_type == "normal":
        return SimulatedConnection(
            type='Normal',
            description='Normal Web Browsing',
            explanation='This represents typical legitimate web traffic. Notice the reasonable duration (5s), '
                       'successful connection (flag=SF), balanced src/dst bytes (request/response), '
                       'logged_in=1 (authenticated), and no error rates.',
            features={
                'duration': 5,
                'protocol_type': 'tcp',
                'service': 'http',
                'flag': 'SF',  # Normal connection
                'src_bytes': 200,  # Request
                'dst_bytes': 5000,  # Response (HTML page)
                'logged_in': 1,
                'count': 5,  # Reasonable connection count
                'srv_count': 5,
                'serror_rate': 0.0,  # No errors
                'rerror_rate': 0.0,
                'same_srv_rate': 1.0,  # Browsing same website
                'diff_srv_rate': 0.0,
                'dst_host_count': 5,
                'dst_host_srv_count': 5,
                'dst_host_same_srv_rate': 1.0,
            }
        )

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown attack type: {attack_type}. "
                   f"Supported types: normal, dos, probe, r2l, u2r"
        )


@router.get("/simulate/random")
async def simulate_random():
    """
    Generate a random simulated connection

    Returns:
        Random simulated connection (normal or attack)
    """
    attack_types = ['normal', 'dos', 'probe', 'r2l', 'u2r']
    selected = random.choice(attack_types)
    return await simulate_attack(selected)


@router.get("/attack_info/{attack_type}")
async def get_attack_info(attack_type: str):
    """
    Get detailed information about an attack type

    Args:
        attack_type: Type of attack (dos, probe, r2l, u2r)

    Returns:
        Detailed attack information
    """

    attack_type = attack_type.lower()

    attack_info = {
        'dos': {
            'name': 'Denial of Service (DoS)',
            'category': 'DoS',
            'description': 'Attacks that aim to make a system or network resource unavailable',
            'techniques': [
                'SYN Flood - Overwhelming with SYN packets',
                'Ping Flood - Flooding with ICMP Echo requests',
                'Smurf Attack - ICMP broadcast amplification',
                'Teardrop - Sending malformed fragmented packets'
            ],
            'indicators': [
                'Extremely high connection count (>500)',
                'High SYN error rate (>90%)',
                'Zero or very low duration',
                'Zero data transfer (src_bytes=0, dst_bytes=0)',
                'Flag status: S0, S1, S2 (incomplete handshakes)'
            ],
            'impact': 'High - System becomes unresponsive, legitimate users cannot access services',
            'mitigation': [
                'Rate limiting',
                'SYN cookies',
                'Firewall rules',
                'Load balancing',
                'DDoS protection services (Cloudflare, AWS Shield)'
            ]
        },
        'probe': {
            'name': 'Probing/Reconnaissance',
            'category': 'Probe',
            'description': 'Attacks that gather information about the target system',
            'techniques': [
                'Port Scanning - Testing which ports are open',
                'Network Mapping - Discovering network topology',
                'OS Fingerprinting - Identifying operating system',
                'Vulnerability Scanning - Finding security weaknesses'
            ],
            'indicators': [
                'High diff_srv_rate (>80%) - scanning different services',
                'High rerror_rate (>70%) - many rejected connections',
                'Low or zero duration - no data transfer',
                'Sequential port access pattern',
                'Many short connections'
            ],
            'impact': 'Medium - Precursor to actual attack, information gathering',
            'mitigation': [
                'Intrusion Detection Systems (IDS)',
                'Port scan detection tools',
                'Firewall with stealth mode',
                'Honeypots to trap attackers',
                'Network segmentation'
            ]
        },
        'r2l': {
            'name': 'Remote to Local (R2L)',
            'category': 'R2L',
            'description': 'Attacks where attacker gains unauthorized local access from remote',
            'techniques': [
                'Brute Force - Trying many password combinations',
                'Dictionary Attack - Using common password lists',
                'SQL Injection - Exploiting database vulnerabilities',
                'Command Injection - Injecting malicious commands',
                'Password Guessing - Using context-based guesses'
            ],
            'indicators': [
                'High num_failed_logins (>3)',
                'logged_in = 0 (not authenticated)',
                'Repeated connection attempts',
                'Same service targeted multiple times',
                'Unusual access patterns'
            ],
            'impact': 'High - Unauthorized access to system and data',
            'mitigation': [
                'Strong password policies',
                'Account lockout after failed attempts',
                'Two-factor authentication (2FA)',
                'CAPTCHA for login forms',
                'Input validation and sanitization',
                'Web Application Firewall (WAF)'
            ]
        },
        'u2r': {
            'name': 'User to Root (U2R)',
            'category': 'U2R',
            'description': 'Attacks where normal user escalates to root/administrator privileges',
            'techniques': [
                'Buffer Overflow - Overwriting memory to execute code',
                'Rootkit - Installing privileged malware',
                'Kernel Exploits - Exploiting OS vulnerabilities',
                'Privilege Escalation - Abusing misconfigurations',
                'SUID Exploits - Exploiting setuid binaries'
            ],
            'indicators': [
                'logged_in = 1 (already authenticated)',
                'num_root > 0 (root access attempts)',
                'root_shell = 1 (obtained root shell)',
                'num_file_creations > 0 (creating files)',
                'su_attempted = 1 (trying to switch user)'
            ],
            'impact': 'Critical - Complete system compromise, full control',
            'mitigation': [
                'Principle of least privilege',
                'Regular security patches and updates',
                'SELinux/AppArmor mandatory access control',
                'Disable unnecessary services',
                'File integrity monitoring',
                'Secure coding practices'
            ]
        }
    }

    if attack_type not in attack_info:
        raise HTTPException(
            status_code=404,
            detail=f"Attack info not found for: {attack_type}"
        )

    return attack_info[attack_type]


@router.get("/all_attack_types")
async def get_all_attack_types():
    """
    Get list of all supported attack types

    Returns:
        List of attack types with brief descriptions
    """
    return {
        'attack_types': [
            {
                'id': 'normal',
                'name': 'Normal Traffic',
                'category': 'Normal',
                'description': 'Legitimate network activity',
                'severity': 'none'
            },
            {
                'id': 'dos',
                'name': 'Denial of Service',
                'category': 'DoS',
                'description': 'Overwhelming system resources',
                'severity': 'high'
            },
            {
                'id': 'probe',
                'name': 'Probing/Scanning',
                'category': 'Probe',
                'description': 'Information gathering',
                'severity': 'medium'
            },
            {
                'id': 'r2l',
                'name': 'Remote to Local',
                'category': 'R2L',
                'description': 'Unauthorized remote access',
                'severity': 'high'
            },
            {
                'id': 'u2r',
                'name': 'User to Root',
                'category': 'U2R',
                'description': 'Privilege escalation',
                'severity': 'critical'
            }
        ]
    }
