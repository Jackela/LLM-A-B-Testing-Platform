# Security Operations Guide

## Overview

This guide covers security operations, incident response, and compliance procedures for the LLM A/B Testing Platform.

## Security Architecture

### Defense-in-Depth Strategy

1. **Network Security**
   - IP whitelisting/blacklisting
   - Rate limiting and DDoS protection
   - TLS 1.3+ encryption for all communications

2. **Application Security**
   - Input validation and sanitization
   - SQL injection, XSS, and CSRF protection
   - Secure authentication with JWT and MFA
   - Role-based access control (RBAC)

3. **Data Security**
   - Encryption at rest for sensitive data
   - PII tokenization and anonymization
   - Secure secret management with rotation

4. **Infrastructure Security**
   - Container security hardening
   - Secrets management with rotation
   - Security scanning in CI/CD pipeline

## Authentication & Authorization

### Enhanced JWT System

**Features:**
- Access tokens (30min expiry)
- Refresh tokens (7 days expiry)
- Multi-factor authentication support
- Account lockout after 5 failed attempts
- Session management with unique IDs

**User Roles:**
- `super_admin`: All permissions
- `admin`: User management, system config, advanced analytics
- `manager`: Test management, advanced analytics
- `analyst`: Analytics and reporting
- `user`: Basic test operations
- `viewer`: Read-only access
- `api_user`: API access with bulk operations

### MFA Configuration

```python
# Enable MFA for user
from src.infrastructure.security.auth import get_auth_system

auth_system = get_auth_system()
success, qr_uri, backup_codes = auth_system.enable_mfa("username")
```

**Backup Codes:**
- 10 single-use codes generated per user
- Store securely and provide to user
- Log usage for audit trail

## Secrets Management

### Environment Variables

```bash
# Authentication
JWT_SECRET_KEY=your-jwt-secret-key-here
ENCRYPTION_KEY=your-encryption-key-here

# Database
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_USER=postgres
DATABASE_PASSWORD=your-db-password
DATABASE_NAME=llm_ab_testing

# External APIs
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
GOOGLE_API_KEY=your-google-key

# Monitoring
SLACK_WEBHOOK_URL=https://hooks.slack.com/your/webhook
ALERT_FROM_EMAIL=alerts@yourcompany.com
ALERT_TO_EMAILS=admin@yourcompany.com,security@yourcompany.com

# Security
ALLOWED_ORIGINS=http://localhost:3000,https://yourdomain.com
```

### Secrets Rotation

**Automated Rotation:**
```bash
# Run secrets rotation script
python scripts/rotate_secrets.py --secret-name jwt_secret
```

**Manual Rotation Process:**
1. Generate new secret value
2. Update in secrets manager
3. Deploy to all environments
4. Verify functionality
5. Remove old secret after grace period

## Security Monitoring

### Log Categories

1. **Security Events** (`security.log`)
   - Authentication failures
   - Authorization violations
   - Rate limit violations
   - Input validation failures
   - Suspicious activity

2. **Application Logs** (`app.log`)
   - All application events
   - Performance metrics
   - Error tracking

3. **Access Logs** (`access.log`)
   - HTTP request/response logs
   - API access patterns
   - User activity tracking

### Alert Configuration

**Critical Alerts** (Immediate Response):
- Authentication system failure
- Database connection failure
- Multiple failed login attempts
- Potential security breach
- System resource exhaustion (>95%)

**High Priority Alerts** (30 minutes):
- High error rate (>5%)
- Performance degradation
- Security scan failures
- MFA bypass attempts

**Medium Priority Alerts** (1 hour):
- Rate limit exceeded
- Configuration drift
- Certificate expiration warnings

### Security Metrics

**Key Performance Indicators:**
- Failed authentication rate
- Active user sessions
- API rate limit hits
- Security scan pass rate
- Vulnerability resolution time

## Incident Response

### Security Incident Classification

**Level 1 - Critical:**
- Active data breach
- System compromise
- Authentication system failure
- Critical vulnerability exploitation

**Level 2 - High:**
- Suspected intrusion
- Service disruption
- High-severity vulnerability
- Compliance violation

**Level 3 - Medium:**
- Policy violations
- Minor security misconfigurations
- Phishing attempts
- Low-severity vulnerabilities

**Level 4 - Low:**
- Security awareness issues
- Minor policy violations
- Informational security events

### Incident Response Workflow

1. **Detection & Analysis**
   - Monitor security alerts
   - Analyze logs and metrics
   - Determine incident severity
   - Document initial findings

2. **Containment**
   - Isolate affected systems
   - Block malicious traffic
   - Disable compromised accounts
   - Preserve evidence

3. **Eradication & Recovery**
   - Remove threat from environment
   - Patch vulnerabilities
   - Restore from clean backups
   - Implement additional controls

4. **Post-Incident Activities**
   - Conduct lessons learned
   - Update procedures
   - Improve monitoring
   - Report to stakeholders

### Emergency Contacts

```
Security Team: security@yourcompany.com
On-Call Engineer: +1-XXX-XXX-XXXX
Management: management@yourcompany.com
Legal: legal@yourcompany.com
```

## Security Testing

### Automated Security Scans

**Daily Scans:**
```bash
# Dependency vulnerability scan
safety check --json --output safety-report.json

# Static code analysis
bandit -r src/ -f json -o bandit-report.json

# Container security scan
docker run --rm -v $(pwd):/app securecodewarrior/docker-security-scanning
```

**Weekly Scans:**
- Full security test suite
- Penetration testing (automated)
- Compliance validation
- Access review audit

**Monthly Scans:**
- External security assessment
- Red team exercises
- Security awareness training
- Policy review and updates

### Security Test Suite

```bash
# Run comprehensive security tests
python src/infrastructure/security/testing.py /path/to/project http://localhost:8000 security-report.json

# View results
cat security-report.json | jq '.summary'
```

## Compliance & Audit

### Security Standards

**OWASP Top 10 Compliance:**
- A01: Broken Access Control ✓
- A02: Cryptographic Failures ✓
- A03: Injection ✓
- A04: Insecure Design ✓
- A05: Security Misconfiguration ✓
- A06: Vulnerable Components ✓
- A07: Authentication Failures ✓
- A08: Software Integrity Failures ✓
- A09: Logging Failures ✓
- A10: Server-Side Request Forgery ✓

**Additional Standards:**
- ISO 27001 security controls
- NIST Cybersecurity Framework
- SOC 2 Type II requirements
- GDPR privacy compliance

### Audit Trail

**Required Logging:**
- User authentication and authorization
- Data access and modifications
- Administrative actions
- Security configuration changes
- Incident response activities

**Log Retention:**
- Security logs: 7 years
- Access logs: 1 year
- Application logs: 6 months
- Debug logs: 30 days

### Privacy & Data Protection

**PII Handling:**
- Data classification and labeling
- Encryption of sensitive data
- Data minimization principles
- Right to be forgotten procedures

**Data Retention:**
- User data: As per user agreement
- Analytics data: 2 years
- Log data: As per retention policy
- Backup data: 90 days

## Operational Procedures

### Daily Security Operations

1. **Morning Security Review** (30 minutes)
   - Review overnight security alerts
   - Check system health status
   - Verify backup completion
   - Monitor threat intelligence feeds

2. **Ongoing Monitoring**
   - Watch security dashboards
   - Investigate anomalies
   - Respond to alerts
   - Update threat indicators

3. **End-of-Day Review** (15 minutes)
   - Summary of security events
   - Outstanding issues review
   - Tomorrow's priorities
   - Handover to next shift

### Weekly Security Tasks

- **Monday**: Security scan analysis and remediation
- **Tuesday**: Access review and user management
- **Wednesday**: Vulnerability assessment and patching
- **Thursday**: Security awareness and training
- **Friday**: Security metrics review and reporting

### Monthly Security Activities

- Security risk assessment
- Policy and procedure updates
- Vendor security reviews
- Disaster recovery testing
- Security awareness campaigns

## Emergency Procedures

### System Compromise Response

**Immediate Actions (0-1 hour):**
1. Isolate affected systems
2. Preserve evidence
3. Notify security team
4. Enable enhanced monitoring
5. Document all actions

**Short-term Actions (1-4 hours):**
1. Analyze attack vectors
2. Identify scope of compromise
3. Notify stakeholders
4. Begin containment procedures
5. Contact external resources if needed

**Recovery Actions (4-24 hours):**
1. Eradicate threat from environment
2. Apply security patches
3. Restore from clean backups
4. Implement additional controls
5. Monitor for recurring activity

### Data Breach Response

**Detection to Notification Timeline:**
- Internal notification: Within 1 hour
- Management notification: Within 2 hours
- Customer notification: Within 24 hours
- Regulatory notification: As required by law

**Breach Response Team:**
- Incident Commander
- Security Analyst
- Legal Counsel
- Public Relations
- Customer Service

## Maintenance & Updates

### Security Updates

**Critical Updates** (Within 24 hours):
- Zero-day vulnerability patches
- Security framework updates
- Authentication system fixes

**High Priority Updates** (Within 1 week):
- Regular security patches
- Dependency updates
- Configuration improvements

**Standard Updates** (Monthly):
- Non-critical security updates
- Feature security enhancements
- Policy updates

### Change Management

**Security Change Approval:**
1. Security impact assessment
2. Risk analysis and mitigation
3. Testing in staging environment
4. Security team approval
5. Implementation and monitoring

**Emergency Changes:**
- Can be implemented immediately for critical security fixes
- Must be documented and reviewed within 24 hours
- Require post-implementation validation

## Contact Information

**Security Team:**
- Primary: security@yourcompany.com
- Emergency: +1-XXX-XXX-XXXX
- Slack: #security-incidents

**Escalation Path:**
1. Security Engineer
2. Security Manager
3. CISO
4. CTO
5. CEO

## Additional Resources

- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Security Incident Response Plan Template](./incident-response-template.md)
- [Security Awareness Training Materials](./security-training/)