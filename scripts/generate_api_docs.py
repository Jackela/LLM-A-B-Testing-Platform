"""Generate comprehensive API documentation for LLM A/B Testing Platform."""

import json
import yaml
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.presentation.api.app import create_app
from src.presentation.api.documentation.api_examples import DocumentationExamples


class APIDocumentationGenerator:
    """Generate comprehensive API documentation in multiple formats."""
    
    def __init__(self, output_dir: str = "docs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.app = create_app()
        
    def generate_openapi_schema(self) -> Dict[str, Any]:
        """Generate OpenAPI schema."""
        return self.app.openapi()
    
    def save_openapi_json(self, schema: Dict[str, Any]) -> Path:
        """Save OpenAPI schema as JSON."""
        output_file = self.output_dir / "openapi.json"
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2, default=str)
        
        print(f"✅ OpenAPI JSON saved to: {output_file}")
        return output_file
    
    def save_openapi_yaml(self, schema: Dict[str, Any]) -> Path:
        """Save OpenAPI schema as YAML."""
        output_file = self.output_dir / "openapi.yaml"
        
        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(schema, f, default_flow_style=False, sort_keys=False)
        
        print(f"✅ OpenAPI YAML saved to: {output_file}")
        return output_file
    
    def generate_markdown_docs(self, schema: Dict[str, Any]) -> Path:
        """Generate comprehensive Markdown documentation."""
        output_file = self.output_dir / "API_Documentation.md"
        
        markdown_content = self._build_markdown_content(schema)
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        
        print(f"✅ Markdown documentation saved to: {output_file}")
        return output_file
    
    def _build_markdown_content(self, schema: Dict[str, Any]) -> str:
        """Build comprehensive Markdown documentation."""
        info = schema.get("info", {})
        paths = schema.get("paths", {})
        components = schema.get("components", {})
        
        content = f"""# {info.get('title', 'API Documentation')}

Version: {info.get('version', '1.0.0')}  
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

{info.get('description', 'API documentation for LLM A/B Testing Platform').strip()}

## Base URLs

- Development: `http://localhost:8000`
- Production: `https://api.llm-testing.example.com`

## Authentication

This API uses Bearer token authentication. Include your token in the Authorization header:

```
Authorization: Bearer <your-jwt-token>
```

### Getting a Token

1. **Login**: `POST /api/v1/auth/login`
2. **Use Token**: Include in Authorization header for all requests

## API Endpoints

"""
        
        # Group endpoints by tags
        endpoint_groups = {}
        for path, methods in paths.items():
            for method, details in methods.items():
                if method.upper() in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']:
                    tags = details.get('tags', ['Untagged'])
                    tag = tags[0] if tags else 'Untagged'
                    
                    if tag not in endpoint_groups:
                        endpoint_groups[tag] = []
                    
                    endpoint_groups[tag].append({
                        'path': path,
                        'method': method.upper(),
                        'details': details
                    })
        
        # Generate documentation for each group
        for tag, endpoints in endpoint_groups.items():
            content += f"### {tag}\n\n"
            
            for endpoint in endpoints:
                content += self._format_endpoint(endpoint)
        
        # Add schemas section
        content += "\n## Data Models\n\n"
        schemas = components.get("schemas", {})
        for schema_name, schema_def in schemas.items():
            content += self._format_schema(schema_name, schema_def)
        
        # Add examples section
        content += "\n## API Examples\n\n"
        content += self._generate_examples_section()
        
        # Add error handling section
        content += self._generate_error_handling_section()
        
        return content
    
    def _format_endpoint(self, endpoint: Dict[str, Any]) -> str:
        """Format a single endpoint for Markdown."""
        path = endpoint['path']
        method = endpoint['method']
        details = endpoint['details']
        
        summary = details.get('summary', f'{method} {path}')
        description = details.get('description', '')
        
        content = f"#### `{method} {path}`\n\n"
        content += f"**{summary}**\n\n"
        
        if description:
            content += f"{description}\n\n"
        
        # Parameters
        parameters = details.get('parameters', [])
        if parameters:
            content += "**Parameters:**\n\n"
            for param in parameters:
                name = param.get('name', 'unnamed')
                location = param.get('in', 'query')
                required = param.get('required', False)
                description = param.get('description', '')
                param_type = param.get('schema', {}).get('type', 'string')
                
                required_text = " (required)" if required else " (optional)"
                content += f"- `{name}` ({location}, {param_type}){required_text}: {description}\n"
            content += "\n"
        
        # Request body
        request_body = details.get('requestBody', {})
        if request_body:
            content += "**Request Body:**\n\n"
            content_types = request_body.get('content', {})
            for content_type, body_details in content_types.items():
                content += f"Content-Type: `{content_type}`\n\n"
                schema = body_details.get('schema', {})
                if schema:
                    content += f"```json\n{self._generate_example_from_schema(schema)}\n```\n\n"
        
        # Responses
        responses = details.get('responses', {})
        if responses:
            content += "**Responses:**\n\n"
            for status_code, response_details in responses.items():
                description = response_details.get('description', '')
                content += f"- `{status_code}`: {description}\n"
            content += "\n"
        
        content += "---\n\n"
        return content
    
    def _format_schema(self, schema_name: str, schema_def: Dict[str, Any]) -> str:
        """Format a schema definition for Markdown."""
        content = f"### {schema_name}\n\n"
        
        description = schema_def.get('description', '')
        if description:
            content += f"{description}\n\n"
        
        properties = schema_def.get('properties', {})
        if properties:
            content += "**Properties:**\n\n"
            for prop_name, prop_details in properties.items():
                prop_type = prop_details.get('type', 'unknown')
                prop_description = prop_details.get('description', '')
                example = prop_details.get('example', '')
                
                content += f"- `{prop_name}` ({prop_type}): {prop_description}"
                if example:
                    content += f" (example: `{example}`)"
                content += "\n"
            content += "\n"
        
        return content
    
    def _generate_example_from_schema(self, schema: Dict[str, Any]) -> str:
        """Generate a simple JSON example from schema."""
        if '$ref' in schema:
            return '{ /* See referenced schema */ }'
        
        properties = schema.get('properties', {})
        if not properties:
            return '{}'
        
        example = {}
        for prop_name, prop_details in properties.items():
            if 'example' in prop_details:
                example[prop_name] = prop_details['example']
            elif prop_details.get('type') == 'string':
                example[prop_name] = "string"
            elif prop_details.get('type') == 'integer':
                example[prop_name] = 0
            elif prop_details.get('type') == 'number':
                example[prop_name] = 0.0
            elif prop_details.get('type') == 'boolean':
                example[prop_name] = True
            elif prop_details.get('type') == 'array':
                example[prop_name] = []
            else:
                example[prop_name] = None
        
        return json.dumps(example, indent=2)
    
    def _generate_examples_section(self) -> str:
        """Generate examples section."""
        content = ""
        examples = DocumentationExamples.get_all_examples()
        
        for category, category_examples in examples.items():
            content += f"### {category.value.title()} Examples\n\n"
            
            for example in category_examples:
                content += f"#### {example.summary}\n\n"
                if example.description:
                    content += f"{example.description}\n\n"
                
                content += "```json\n"
                content += json.dumps(example.value, indent=2)
                content += "\n```\n\n"
        
        return content
    
    def _generate_error_handling_section(self) -> str:
        """Generate error handling documentation."""
        return """
## Error Handling

The API uses standard HTTP status codes and returns consistent error responses:

### Error Response Format

```json
{
  "error": "error_type",
  "message": "Human-readable error message",
  "details": {
    "additional": "context information"
  }
}
```

### Status Codes

| Code | Description | Action |
|------|-------------|--------|
| 200 | Success | Continue |
| 201 | Created | Resource created successfully |
| 400 | Bad Request | Check request format and data |
| 401 | Unauthorized | Provide valid authentication |
| 403 | Forbidden | Check permissions |
| 404 | Not Found | Verify resource exists |
| 422 | Validation Error | Fix validation issues |
| 429 | Rate Limited | Reduce request rate |
| 500 | Server Error | Contact support |

### Rate Limiting

API requests are rate-limited:
- **Standard Users**: 100 requests per minute
- **Premium Users**: 1000 requests per minute
- **Enterprise**: Custom limits

When rate limited, the response includes a `Retry-After` header indicating when to retry.

## SDK and Client Libraries

### Python

```python
import httpx

class LLMTestingClient:
    def __init__(self, token: str, base_url: str = "http://localhost:8000"):
        self.token = token
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {token}"}
    
    async def create_provider(self, provider_data: dict):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/v1/providers/",
                json=provider_data,
                headers=self.headers
            )
            return response.json()
```

### JavaScript

```javascript
class LLMTestingClient {
    constructor(token, baseUrl = 'http://localhost:8000') {
        this.token = token;
        this.baseUrl = baseUrl;
        this.headers = { 'Authorization': `Bearer ${token}` };
    }
    
    async createProvider(providerData) {
        const response = await fetch(`${this.baseUrl}/api/v1/providers/`, {
            method: 'POST',
            headers: { ...this.headers, 'Content-Type': 'application/json' },
            body: JSON.stringify(providerData)
        });
        return response.json();
    }
}
```

## Support

- **Documentation**: `/api/v1/docs`
- **Interactive API**: `/api/v1/redoc`
- **Health Check**: `/health`
- **Support Email**: support@llm-testing.example.com
"""
    
    def generate_postman_collection(self, schema: Dict[str, Any]) -> Path:
        """Generate Postman collection for API testing."""
        output_file = self.output_dir / "postman_collection.json"
        
        collection = {
            "info": {
                "name": schema.get("info", {}).get("title", "LLM A/B Testing Platform API"),
                "description": "Postman collection for LLM A/B Testing Platform API",
                "version": schema.get("info", {}).get("version", "1.0.0"),
                "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
            },
            "auth": {
                "type": "bearer",
                "bearer": [
                    {
                        "key": "token",
                        "value": "{{ACCESS_TOKEN}}",
                        "type": "string"
                    }
                ]
            },
            "variable": [
                {
                    "key": "BASE_URL",
                    "value": "http://localhost:8000",
                    "type": "string"
                },
                {
                    "key": "ACCESS_TOKEN",
                    "value": "your-jwt-token-here",
                    "type": "string"
                }
            ],
            "item": []
        }
        
        # Convert OpenAPI paths to Postman requests
        paths = schema.get("paths", {})
        for path, methods in paths.items():
            for method, details in methods.items():
                if method.upper() in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']:
                    request = self._create_postman_request(path, method.upper(), details)
                    collection["item"].append(request)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(collection, f, indent=2)
        
        print(f"✅ Postman collection saved to: {output_file}")
        return output_file
    
    def _create_postman_request(self, path: str, method: str, details: Dict[str, Any]) -> Dict[str, Any]:
        """Create a Postman request from OpenAPI endpoint."""
        name = details.get('summary', f'{method} {path}')
        description = details.get('description', '')
        
        # Convert path parameters
        url_path = path
        parameters = details.get('parameters', [])
        query_params = []
        
        for param in parameters:
            if param.get('in') == 'path':
                url_path = url_path.replace(f"{{{param['name']}}}", f"{{{{param['name']}}}}")
            elif param.get('in') == 'query':
                query_params.append({
                    "key": param['name'],
                    "value": f"{{{{param['name']}}}}",
                    "description": param.get('description', '')
                })
        
        request = {
            "name": name,
            "request": {
                "method": method,
                "header": [
                    {
                        "key": "Content-Type",
                        "value": "application/json",
                        "type": "text"
                    }
                ],
                "url": {
                    "raw": f"{{{{BASE_URL}}}}{url_path}",
                    "host": ["{{BASE_URL}}"],
                    "path": url_path.strip('/').split('/'),
                    "query": query_params
                },
                "description": description
            }
        }
        
        # Add request body for POST/PUT methods
        if method in ['POST', 'PUT', 'PATCH']:
            request_body = details.get('requestBody', {})
            if request_body:
                content = request_body.get('content', {})
                if 'application/json' in content:
                    schema = content['application/json'].get('schema', {})
                    example_body = self._generate_example_from_schema(schema)
                    request["request"]["body"] = {
                        "mode": "raw",
                        "raw": example_body
                    }
        
        return request
    
    def generate_all_docs(self) -> Dict[str, Path]:
        """Generate all documentation formats."""
        print("📚 Generating comprehensive API documentation...")
        
        # Generate OpenAPI schema
        schema = self.generate_openapi_schema()
        
        # Generate all formats
        results = {
            "openapi_json": self.save_openapi_json(schema),
            "openapi_yaml": self.save_openapi_yaml(schema),
            "markdown": self.generate_markdown_docs(schema),
            "postman": self.generate_postman_collection(schema)
        }
        
        print(f"\n🎉 Documentation generated successfully!")
        print(f"📁 Output directory: {self.output_dir.absolute()}")
        
        return results


def main():
    """Main entry point for documentation generation."""
    parser = argparse.ArgumentParser(description="Generate API documentation")
    parser.add_argument("--output", "-o", default="docs", help="Output directory")
    parser.add_argument("--format", "-f", choices=["all", "json", "yaml", "markdown", "postman"], 
                       default="all", help="Documentation format to generate")
    
    args = parser.parse_args()
    
    try:
        generator = APIDocumentationGenerator(args.output)
        
        if args.format == "all":
            generator.generate_all_docs()
        else:
            schema = generator.generate_openapi_schema()
            
            if args.format == "json":
                generator.save_openapi_json(schema)
            elif args.format == "yaml":
                generator.save_openapi_yaml(schema)
            elif args.format == "markdown":
                generator.generate_markdown_docs(schema)
            elif args.format == "postman":
                generator.generate_postman_collection(schema)
        
        print("\n✅ Documentation generation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error generating documentation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()