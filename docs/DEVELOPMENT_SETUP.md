# Development Environment Setup

This guide will help you set up the LLM A/B Testing Platform for local development.

## Prerequisites

- Python 3.11+
- Poetry
- Docker and Docker Compose
- Git

## Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd LLM-A-B-Testing-Platform
```

### 2. Start Infrastructure Services

```bash
# Start PostgreSQL and Redis
docker-compose -f docker-compose.dev.yml up -d

# Optional: Start PgAdmin for database administration
docker-compose -f docker-compose.dev.yml --profile admin up -d
```

### 3. Install Dependencies

```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install --with dev
```

### 4. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# The default configuration should work with Docker Compose
# Edit .env if you need to customize settings
```

### 5. Initialize Database

```bash
# Run database migrations
poetry run alembic upgrade head
```

### 6. Verify Setup

```bash
# Run tests to verify everything is working
poetry run pytest tests/unit/ -v

# Run the API server
poetry run uvicorn src.presentation.api.main:app --reload
```

## Database Management

### Accessing PostgreSQL

- **Host**: localhost
- **Port**: 5432
- **Username**: postgres
- **Password**: postgres
- **Development DB**: llm_testing_dev
- **Test DB**: test_db

### Using PgAdmin (Optional)

If you started PgAdmin:
- **URL**: http://localhost:8080
- **Email**: admin@example.com
- **Password**: admin

### Database Migrations

```bash
# Create a new migration
poetry run alembic revision --autogenerate -m "description"

# Apply migrations
poetry run alembic upgrade head

# Check current migration status
poetry run alembic current

# View migration history
poetry run alembic history
```

## Running Tests

### Unit Tests
```bash
poetry run pytest tests/unit/ -v
```

### Integration Tests
```bash
poetry run pytest tests/integration/ -v
```

### All Tests
```bash
poetry run pytest -v
```

### With Coverage
```bash
poetry run pytest --cov=src --cov-report=html
```

## Development Workflow

### 1. Code Quality

```bash
# Format code
poetry run black src tests

# Sort imports
poetry run isort src tests

# Lint code
poetry run flake8 src tests

# Type checking
poetry run mypy src
```

### 2. Pre-commit Hooks

```bash
# Install pre-commit hooks
poetry run pre-commit install

# Run hooks manually
poetry run pre-commit run --all-files
```

## Troubleshooting

### Database Connection Issues

1. Check if PostgreSQL is running:
   ```bash
   docker-compose -f docker-compose.dev.yml ps
   ```

2. Check database logs:
   ```bash
   docker-compose -f docker-compose.dev.yml logs postgres
   ```

3. Test connection manually:
   ```bash
   psql -h localhost -p 5432 -U postgres -d llm_testing_dev
   ```

### Migration Issues

1. Check Alembic configuration:
   ```bash
   poetry run alembic check
   ```

2. View current migration status:
   ```bash
   poetry run alembic current
   ```

3. Reset database (development only):
   ```bash
   docker-compose -f docker-compose.dev.yml down -v
   docker-compose -f docker-compose.dev.yml up -d
   poetry run alembic upgrade head
   ```

### Port Conflicts

If you get port conflicts, update the ports in `docker-compose.dev.yml`:

```yaml
services:
  postgres:
    ports:
      - "5433:5432"  # Changed from 5432:5432
  redis:
    ports:
      - "6380:6379"  # Changed from 6379:6379
```

Then update your `.env` file accordingly.

## IDE Configuration

### VS Code

Recommended extensions:
- Python
- Pylance
- Black Formatter
- isort
- GitLens

### PyCharm

1. Configure Poetry interpreter
2. Enable code formatting on save
3. Configure database connection

## CI/CD

The project uses GitHub Actions for CI/CD. The workflow:

1. Sets up PostgreSQL and Redis services
2. Installs dependencies with Poetry
3. Runs database migrations
4. Executes linting and type checking
5. Runs test suites (unit, integration, e2e)
6. Generates coverage reports

See `.github/workflows/test.yml` for details.