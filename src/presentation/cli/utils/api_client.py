"""API client for CLI operations."""

import json
import logging
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class APIClientError(Exception):
    """API client error."""

    pass


class APIClient:
    """REST API client for LLM A/B Testing Platform."""

    def __init__(self, base_url: str, token: Optional[str] = None, timeout: int = 30):
        """Initialize API client.

        Args:
            base_url: Base URL for the API
            token: Authentication token
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

        # Set authentication header
        if token:
            self.session.headers.update({"Authorization": f"Bearer {token}"})

        # Set common headers
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "llm-test-cli/1.0.0",
            }
        )

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request to API.

        Args:
            method: HTTP method
            endpoint: API endpoint
            **kwargs: Additional request parameters

        Returns:
            Response data

        Raises:
            APIClientError: If request fails
        """
        url = f"{self.base_url}/api/v1{endpoint}"

        try:
            logger.debug(f"Making {method} request to {url}")

            response = self.session.request(method=method, url=url, timeout=self.timeout, **kwargs)

            # Log response details
            logger.debug(f"Response status: {response.status_code}")

            # Handle different response types
            if response.headers.get("content-type", "").startswith("application/json"):
                data = response.json()
            else:
                data = {"content": response.text}

            # Check for errors
            if not response.ok:
                error_msg = data.get("error", {}).get("message", f"HTTP {response.status_code}")
                raise APIClientError(f"API request failed: {error_msg}")

            return data

        except requests.exceptions.ConnectionError:
            raise APIClientError(f"Could not connect to API at {self.base_url}")
        except requests.exceptions.Timeout:
            raise APIClientError("Request timed out")
        except requests.exceptions.RequestException as e:
            raise APIClientError(f"Request failed: {e}")
        except json.JSONDecodeError:
            raise APIClientError("Invalid JSON response")

    def get(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make GET request."""
        return self._request("GET", endpoint, params=params)

    def post(self, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make POST request."""
        return self._request("POST", endpoint, json=data)

    def put(self, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make PUT request."""
        return self._request("PUT", endpoint, json=data)

    def delete(self, endpoint: str) -> Dict[str, Any]:
        """Make DELETE request."""
        return self._request("DELETE", endpoint)

    # Health and status endpoints
    def get_health(self) -> Dict[str, Any]:
        """Get API health status."""
        return self.get("/health")

    def get_dashboard_overview(self, days: int = 30) -> Dict[str, Any]:
        """Get dashboard overview data."""
        return self.get("/analytics/dashboard/overview", params={"days": days})

    # Test management endpoints
    def create_test(self, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create new test."""
        return self.post("/tests", test_config)

    def get_test(self, test_id: str) -> Dict[str, Any]:
        """Get test by ID."""
        return self.get(f"/tests/{test_id}")

    def list_tests(
        self, filters: Optional[Dict] = None, limit: int = 20, page: int = 1
    ) -> Dict[str, Any]:
        """List tests with optional filtering."""
        params = {"page": page, "page_size": limit}
        if filters:
            params.update(filters)
        return self.get("/tests", params=params)

    def update_test(self, test_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update test configuration."""
        return self.put(f"/tests/{test_id}", updates)

    def delete_test(self, test_id: str) -> Dict[str, Any]:
        """Delete test."""
        try:
            self.delete(f"/tests/{test_id}")
            return {"success": True}
        except APIClientError:
            return {"success": False, "error": "Failed to delete test"}

    def start_test(self, test_id: str, config: Optional[Dict] = None) -> Dict[str, Any]:
        """Start test execution."""
        return self.post(f"/tests/{test_id}/start", config or {})

    def stop_test(self, test_id: str) -> Dict[str, Any]:
        """Stop test execution."""
        return self.post(f"/tests/{test_id}/stop")

    def get_test_progress(self, test_id: str) -> Dict[str, Any]:
        """Get test execution progress."""
        return self.get(f"/tests/{test_id}/progress")

    def get_test_results(self, test_id: str) -> Dict[str, Any]:
        """Get test results."""
        return self.get(f"/analytics/tests/{test_id}/results")

    def get_detailed_results(self, test_id: str) -> Dict[str, Any]:
        """Get detailed test analysis."""
        return self.get(f"/analytics/tests/{test_id}/analysis")

    def export_test_results(self, test_id: str, format: str = "csv") -> Dict[str, Any]:
        """Export test results."""
        return self.post(f"/analytics/tests/{test_id}/export", {"format": format})

    def add_test_samples(self, test_id: str, samples: List[Dict]) -> Dict[str, Any]:
        """Add samples to test."""
        return self.post(f"/tests/{test_id}/samples", {"samples": samples})

    # Provider endpoints
    def list_providers(self) -> Dict[str, Any]:
        """List model providers."""
        return self.get("/providers")

    def get_provider(self, provider_id: str) -> Dict[str, Any]:
        """Get provider details."""
        return self.get(f"/providers/{provider_id}")

    def get_provider_models(self, provider_id: str) -> Dict[str, Any]:
        """Get available models for provider."""
        return self.get(f"/providers/{provider_id}/models")

    def test_provider_connection(self, provider_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Test provider connection."""
        return self.post(f"/providers/{provider_id}/test", config)

    def get_provider_health(self, provider_id: str) -> Dict[str, Any]:
        """Get provider health status."""
        return self.get(f"/providers/{provider_id}/health")

    def get_provider_usage(self, provider_id: str, days: int = 7) -> Dict[str, Any]:
        """Get provider usage statistics."""
        return self.get(f"/providers/{provider_id}/usage", params={"days": days})

    # Evaluation endpoints
    def list_evaluation_templates(self) -> Dict[str, Any]:
        """List evaluation templates."""
        return self.get("/evaluation/templates")

    def create_evaluation_template(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """Create custom evaluation template."""
        return self.post("/evaluation/templates", template)

    def list_judges(self) -> Dict[str, Any]:
        """List available evaluation judges."""
        return self.get("/evaluation/judges")

    def list_evaluation_dimensions(self) -> Dict[str, Any]:
        """List available evaluation dimensions."""
        return self.get("/evaluation/dimensions")

    # Analytics endpoints
    def get_model_comparisons(self, model_a: str, model_b: str, days: int = 30) -> Dict[str, Any]:
        """Get model comparison analysis."""
        params = {"model_a": model_a, "model_b": model_b, "days": days}
        return self.get("/analytics/comparisons", params=params)
