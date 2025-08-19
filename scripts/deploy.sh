#!/bin/bash

# LLM A/B Testing Platform Deployment Script
# Production deployment with health checks and rollback capability

set -euo pipefail

# Configuration
PROJECT_NAME="llm-ab-testing"
COMPOSE_FILE="docker-compose.yml"
COMPOSE_PROD_FILE="docker-compose.prod.yml"
ENV_FILE=".env"
BACKUP_DIR="./backups"
LOG_FILE="deploy.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

# Help function
show_help() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

LLM A/B Testing Platform Deployment Script

COMMANDS:
    dev         Start development environment
    prod        Start production environment
    stop        Stop all services
    restart     Restart all services
    logs        Show service logs
    backup      Create database backup
    restore     Restore from backup
    health      Check service health
    clean       Clean up unused containers and images
    update      Update services with zero-downtime

OPTIONS:
    -h, --help     Show this help message
    -v, --verbose  Enable verbose output
    -f, --force    Force operation without prompts
    --no-backup    Skip backup creation during updates

EXAMPLES:
    $0 dev              # Start development environment
    $0 prod             # Start production environment
    $0 update           # Update with zero-downtime deployment
    $0 backup           # Create database backup
    $0 logs api         # Show API service logs
EOF
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if Docker is installed and running
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        error "Docker is not running"
        exit 1
    fi
    
    # Check if Docker Compose is available
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        error "Docker Compose is not installed"
        exit 1
    fi
    
    # Set compose command
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD="docker compose"
    fi
    
    success "Prerequisites check passed"
}

# Create necessary directories
setup_directories() {
    log "Setting up directories..."
    
    mkdir -p logs
    mkdir -p data
    mkdir -p "$BACKUP_DIR"
    mkdir -p configs/ssl
    
    success "Directories created"
}

# Check environment file
check_environment() {
    log "Checking environment configuration..."
    
    if [[ ! -f "$ENV_FILE" ]]; then
        warning "Environment file not found, creating from template..."
        if [[ -f ".env.example" ]]; then
            cp .env.example "$ENV_FILE"
            warning "Please edit $ENV_FILE with your configuration"
        else
            error "No .env.example template found"
            exit 1
        fi
    fi
    
    # Check for required variables
    required_vars=("DATABASE_URL" "REDIS_URL" "SECRET_KEY")
    for var in "${required_vars[@]}"; do
        if ! grep -q "^$var=" "$ENV_FILE"; then
            error "Required environment variable $var not found in $ENV_FILE"
            exit 1
        fi
    done
    
    success "Environment configuration validated"
}

# Wait for service to be healthy
wait_for_service() {
    local service=$1
    local max_attempts=${2:-30}
    local attempt=1
    
    log "Waiting for $service to be healthy..."
    
    while [[ $attempt -le $max_attempts ]]; do
        if $COMPOSE_CMD ps "$service" | grep -q "healthy\|Up"; then
            success "$service is healthy"
            return 0
        fi
        
        log "Attempt $attempt/$max_attempts: $service not ready yet..."
        sleep 10
        ((attempt++))
    done
    
    error "$service failed to become healthy within timeout"
    return 1
}

# Health check for all services
health_check() {
    log "Performing health checks..."
    
    # Check API health
    if curl -f http://localhost:8000/health &> /dev/null; then
        success "API is healthy"
    else
        error "API health check failed"
        return 1
    fi
    
    # Check dashboard
    if curl -f http://localhost:8501/_stcore/health &> /dev/null; then
        success "Dashboard is healthy"
    else
        warning "Dashboard health check failed (may be starting up)"
    fi
    
    # Check database
    if $COMPOSE_CMD exec -T postgres pg_isready -U llm_testing &> /dev/null; then
        success "Database is healthy"
    else
        error "Database health check failed"
        return 1
    fi
    
    # Check Redis
    if $COMPOSE_CMD exec -T redis redis-cli ping | grep -q "PONG"; then
        success "Redis is healthy"
    else
        error "Redis health check failed"
        return 1
    fi
    
    success "All health checks passed"
}

# Create database backup
create_backup() {
    log "Creating database backup..."
    
    local backup_file="$BACKUP_DIR/backup_$(date +%Y%m%d_%H%M%S).sql"
    
    if $COMPOSE_CMD exec -T postgres pg_dump -U llm_testing llm_testing > "$backup_file"; then
        success "Backup created: $backup_file"
        
        # Compress backup
        gzip "$backup_file"
        success "Backup compressed: $backup_file.gz"
        
        # Clean old backups (keep last 10)
        find "$BACKUP_DIR" -name "backup_*.sql.gz" -type f | sort -r | tail -n +11 | xargs -r rm
        log "Old backups cleaned up"
    else
        error "Backup creation failed"
        return 1
    fi
}

# Restore from backup
restore_backup() {
    local backup_file=$1
    
    if [[ ! -f "$backup_file" ]]; then
        error "Backup file not found: $backup_file"
        exit 1
    fi
    
    log "Restoring from backup: $backup_file"
    
    # Stop services except database
    $COMPOSE_CMD stop api celery-worker celery-beat dashboard
    
    # Restore database
    if [[ "$backup_file" == *.gz ]]; then
        gunzip -c "$backup_file" | $COMPOSE_CMD exec -T postgres psql -U llm_testing -d llm_testing
    else
        $COMPOSE_CMD exec -T postgres psql -U llm_testing -d llm_testing < "$backup_file"
    fi
    
    success "Database restored"
    
    # Restart services
    $COMPOSE_CMD start api celery-worker celery-beat dashboard
    
    # Wait for services
    wait_for_service api
    
    success "Restore completed"
}

# Zero-downtime update
zero_downtime_update() {
    log "Starting zero-downtime update..."
    
    # Create backup unless --no-backup flag is set
    if [[ "${NO_BACKUP:-}" != "true" ]]; then
        create_backup
    fi
    
    # Pull latest images
    log "Pulling latest images..."
    $COMPOSE_CMD pull
    
    # Build new images
    log "Building new images..."
    $COMPOSE_CMD build --no-cache
    
    # Rolling update strategy
    local services=("api" "celery-worker" "celery-beat" "dashboard")
    
    for service in "${services[@]}"; do
        log "Updating $service..."
        
        # Get current container
        local old_container=$($COMPOSE_CMD ps -q "$service")
        
        # Start new container
        $COMPOSE_CMD up -d --no-deps "$service"
        
        # Wait for new container to be healthy
        if wait_for_service "$service" 60; then
            # Remove old container if it exists
            if [[ -n "$old_container" ]]; then
                docker stop "$old_container" &> /dev/null || true
                docker rm "$old_container" &> /dev/null || true
            fi
            success "$service updated successfully"
        else
            error "Failed to update $service"
            
            # Rollback: stop new container and start old one
            if [[ -n "$old_container" ]]; then
                log "Rolling back $service..."
                $COMPOSE_CMD stop "$service"
                docker start "$old_container" &> /dev/null || true
                error "Rollback completed for $service"
            fi
            
            exit 1
        fi
    done
    
    # Final health check
    health_check
    
    success "Zero-downtime update completed successfully"
}

# Start development environment
start_dev() {
    log "Starting development environment..."
    
    setup_directories
    check_environment
    
    # Use development compose file if it exists
    local compose_files="-f $COMPOSE_FILE"
    if [[ -f "docker-compose.dev.yml" ]]; then
        compose_files="$compose_files -f docker-compose.dev.yml"
    fi
    
    $COMPOSE_CMD $compose_files up -d --build
    
    # Wait for services
    wait_for_service postgres
    wait_for_service redis
    wait_for_service api
    
    # Run database migrations
    log "Running database migrations..."
    $COMPOSE_CMD exec api alembic upgrade head
    
    health_check
    
    success "Development environment started"
    log "API: http://localhost:8000"
    log "Dashboard: http://localhost:8501"
    log "Prometheus: http://localhost:9090"
    log "Grafana: http://localhost:3000 (admin/admin)"
}

# Start production environment
start_prod() {
    log "Starting production environment..."
    
    setup_directories
    check_environment
    
    # Use production compose file
    local compose_files="-f $COMPOSE_FILE"
    if [[ -f "$COMPOSE_PROD_FILE" ]]; then
        compose_files="$compose_files -f $COMPOSE_PROD_FILE"
    fi
    
    # Create backup if database exists
    if $COMPOSE_CMD ps postgres | grep -q "Up"; then
        create_backup
    fi
    
    $COMPOSE_CMD $compose_files up -d --build
    
    # Wait for services
    wait_for_service postgres
    wait_for_service redis
    wait_for_service api
    
    # Run database migrations
    log "Running database migrations..."
    $COMPOSE_CMD exec api alembic upgrade head
    
    health_check
    
    success "Production environment started"
}

# Show logs
show_logs() {
    local service=${1:-}
    
    if [[ -n "$service" ]]; then
        $COMPOSE_CMD logs -f --tail=100 "$service"
    else
        $COMPOSE_CMD logs -f --tail=100
    fi
}

# Clean up
cleanup() {
    log "Cleaning up unused containers and images..."
    
    # Remove stopped containers
    docker container prune -f
    
    # Remove unused images
    docker image prune -f
    
    # Remove unused volumes (be careful with this)
    if [[ "${FORCE:-}" == "true" ]]; then
        docker volume prune -f
    fi
    
    success "Cleanup completed"
}

# Main script logic
main() {
    local command=""
    local service=""
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -v|--verbose)
                set -x
                shift
                ;;
            -f|--force)
                FORCE="true"
                shift
                ;;
            --no-backup)
                NO_BACKUP="true"
                shift
                ;;
            dev|prod|stop|restart|logs|backup|restore|health|clean|update)
                command=$1
                shift
                ;;
            *)
                if [[ -z "$command" ]]; then
                    error "Unknown option: $1"
                    show_help
                    exit 1
                else
                    service=$1
                    shift
                fi
                ;;
        esac
    done
    
    if [[ -z "$command" ]]; then
        show_help
        exit 1
    fi
    
    # Check prerequisites for most commands
    if [[ "$command" != "help" ]]; then
        check_prerequisites
    fi
    
    # Execute command
    case $command in
        dev)
            start_dev
            ;;
        prod)
            start_prod
            ;;
        stop)
            log "Stopping all services..."
            $COMPOSE_CMD down
            success "All services stopped"
            ;;
        restart)
            log "Restarting all services..."
            $COMPOSE_CMD restart
            wait_for_service api
            health_check
            success "All services restarted"
            ;;
        logs)
            show_logs "$service"
            ;;
        backup)
            create_backup
            ;;
        restore)
            if [[ -z "$service" ]]; then
                error "Please specify backup file to restore"
                exit 1
            fi
            restore_backup "$service"
            ;;
        health)
            health_check
            ;;
        clean)
            cleanup
            ;;
        update)
            zero_downtime_update
            ;;
        *)
            error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"