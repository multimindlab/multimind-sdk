# MultiMind SDK Usage Guide

This guide provides examples of how to use the MultiMind SDK for privacy compliance management.

## Installation

```bash
pip install multimind-sdk
```

## Basic Usage

### Initialize Privacy Compliance Manager

```python
from multimind.compliance.privacy import PrivacyCompliance, GovernanceConfig

# Initialize with your organization's configuration
config = GovernanceConfig(
    organization_id="your_org_id",
    jurisdiction="global",
    regulations=["GDPR", "CCPA", "PDPA"]
)
privacy_manager = PrivacyCompliance(config=config)
```

### Managing Data Purposes

```python
async def manage_privacy():
    # Add a new data purpose
    purpose = await privacy_manager.add_data_purpose(
        purpose_id="marketing_001",
        name="Marketing Communications",
        description="Process user data for marketing communications",
        legal_basis="consent",
        retention_period=365,  # 1 year
        data_categories={DataCategory.PERSONAL, DataCategory.CONTACT}
    )

    # Process privacy-sensitive data
    user_data = await privacy_manager.process_privacy_data(
        data_id="user_123",
        data_type="user_profile",
        content={
            "name": "John Doe",
            "email": "john@example.com",
            "preferences": {"marketing": True}
        },
        jurisdiction="EU",
        data_categories={DataCategory.PERSONAL, DataCategory.CONTACT},
        purposes={"marketing_001"},
        consent_status={"marketing_001": True}
    )
```

### Risk Assessment and Monitoring

```python
# Calculate risk score
risk_score = await privacy_manager.calculate_risk_score(
    entity_id="system_001",
    entity_type="system"
)

# Create compliance dashboard
dashboard = await privacy_manager.create_compliance_dashboard(
    dashboard_id="main_dashboard",
    name="Main Compliance Dashboard",
    description="Overview of compliance status and risks",
    refresh_interval=3600  # 1 hour
)

# Update dashboard metrics
metrics = await privacy_manager.update_dashboard_metrics("main_dashboard")
```

### Compliance Reporting

```python
# Create report template
template = await privacy_manager.create_report_template(
    template_id="quarterly_report",
    name="Quarterly Compliance Report",
    description="Quarterly compliance status and findings",
    regulation="GDPR",
    jurisdiction="EU",
    sections=[
        {
            "id": "compliance_status",
            "type": "compliance_status",
            "title": "Compliance Status"
        },
        {
            "id": "risk_assessment",
            "type": "risk_assessment",
            "title": "Risk Assessment"
        }
    ]
)

# Generate compliance report
report = await privacy_manager.generate_compliance_report(
    template_id="quarterly_report",
    period_start=datetime.now() - timedelta(days=90),
    period_end=datetime.now(),
    jurisdiction="EU",
    regulation="GDPR"
)
```

### Anomaly Detection

```python
# Detect anomalies
anomalies = await privacy_manager.detect_anomalies()

# Create policy violation alert
alert = await privacy_manager.create_policy_alert(
    rule_id="data_retention_001",
    severity="high",
    description="Data retention period exceeded",
    context={
        "data_id": "user_123",
        "retention_period": 365,
        "current_age": 400
    },
    notification_channels=["email", "slack"]
)
```

### Compliance Training

```python
# Create compliance training
training = await privacy_manager.create_compliance_training(
    training_id="privacy_101",
    title="Privacy Compliance Basics",
    description="Introduction to privacy regulations and compliance",
    modules=[
        {
            "id": "module_1",
            "title": "GDPR Overview",
            "duration": 30
        },
        {
            "id": "module_2",
            "title": "Data Protection Principles",
            "duration": 45
        }
    ],
    target_audience=["employees", "contractors"],
    duration=120,  # 2 hours
    completion_criteria={
        "required_modules": ["module_1", "module_2"],
        "minimum_percentage": 80
    }
)

# Track training completion
completion = await privacy_manager.track_training_completion(
    training_id="privacy_101",
    user_id="employee_001",
    completed_modules=["module_1", "module_2"],
    completion_date=datetime.now()
)
```

## Command Line Interface

The MultiMind SDK provides a command-line interface for common operations:

```bash
# Initialize privacy compliance manager
multimind compliance init --org-id your_org_id --jurisdiction global --regulations GDPR CCPA PDPA

# Add a new data purpose
multimind compliance add-purpose \
    --purpose-id marketing_001 \
    --name "Marketing Communications" \
    --description "Process user data for marketing communications" \
    --legal-basis consent \
    --retention-period 365 \
    --categories PERSONAL CONTACT

# Calculate risk score
multimind compliance calculate-risk --entity-id system_001

# Create compliance dashboard
multimind compliance create-dashboard \
    --dashboard-id main_dashboard \
    --name "Main Compliance Dashboard" \
    --description "Overview of compliance status and risks"

# Create report template
multimind compliance create-report-template \
    --template-id quarterly_report \
    --name "Quarterly Compliance Report" \
    --regulation GDPR \
    --jurisdiction EU

# Create compliance training
multimind compliance create-training \
    --training-id privacy_101 \
    --title "Privacy Compliance Basics" \
    --description "Introduction to privacy regulations and compliance" \
    --duration 120

# Detect anomalies
multimind compliance detect-anomalies
```

## API Interface

The MultiMind SDK also provides a REST API interface. Here are some example API calls:

```bash
# Add a new data purpose
curl -X POST http://localhost:8000/purposes \
    -H "Content-Type: application/json" \
    -d '{
        "purpose_id": "marketing_001",
        "name": "Marketing Communications",
        "description": "Process user data for marketing communications",
        "legal_basis": "consent",
        "retention_period": 365,
        "data_categories": ["PERSONAL", "CONTACT"]
    }'

# Calculate risk score
curl -X POST http://localhost:8000/risk/calculate \
    -H "Content-Type: application/json" \
    -d '{
        "entity_id": "system_001",
        "entity_type": "system"
    }'

# Create compliance dashboard
curl -X POST http://localhost:8000/dashboards \
    -H "Content-Type: application/json" \
    -d '{
        "dashboard_id": "main_dashboard",
        "name": "Main Compliance Dashboard",
        "description": "Overview of compliance status and risks",
        "refresh_interval": 3600
    }'

# Create report template
curl -X POST http://localhost:8000/templates \
    -H "Content-Type: application/json" \
    -d '{
        "template_id": "quarterly_report",
        "name": "Quarterly Compliance Report",
        "description": "Quarterly compliance status and findings",
        "regulation": "GDPR",
        "jurisdiction": "EU",
        "sections": [
            {
                "id": "compliance_status",
                "type": "compliance_status",
                "title": "Compliance Status"
            },
            {
                "id": "risk_assessment",
                "type": "risk_assessment",
                "title": "Risk Assessment"
            }
        ]
    }'

# Create compliance training
curl -X POST http://localhost:8000/training \
    -H "Content-Type: application/json" \
    -d '{
        "training_id": "privacy_101",
        "title": "Privacy Compliance Basics",
        "description": "Introduction to privacy regulations and compliance",
        "modules": [
            {
                "id": "module_1",
                "title": "GDPR Overview",
                "duration": 30
            },
            {
                "id": "module_2",
                "title": "Data Protection Principles",
                "duration": 45
            }
        ],
        "target_audience": ["employees", "contractors"],
        "duration": 120,
        "completion_criteria": {
            "required_modules": ["module_1", "module_2"],
            "minimum_percentage": 80
        }
    }'

# Detect anomalies
curl -X GET http://localhost:8000/anomalies
```

## Best Practices

1. **Regular Risk Assessments**: Schedule regular risk assessments to identify and address potential compliance issues.

2. **Automated Monitoring**: Use the dashboard and anomaly detection features to monitor compliance in real-time.

3. **Documentation**: Maintain detailed documentation of data purposes, processing activities, and compliance measures.

4. **Training**: Ensure all employees complete required compliance training and track their progress.

5. **Incident Response**: Have a clear process for handling policy violations and data breaches.

6. **Regular Updates**: Keep the SDK and its dependencies up to date to ensure you have the latest security patches and features.

## Support

For support, please contact:
- Email: contact@multimind.dev
- Documentation: (link to be updated)
- GitHub: https://github.com/multimind-ai/multimind-sdk

## Model Client Usage Examples

### Basic ModelClient Subclass
```python
from multimind.client.model_client import ModelClient

class MyCustomModelClient(ModelClient):
    def generate(self, prompt: str, **kwargs) -> str:
        # Custom model logic here
        return "response"
```

### LSTMModelClient Example
```python
from multimind.client.model_client import LSTMModelClient
# Assume you have a trained model and tokenizer
client = LSTMModelClient(model_path="lstm.pt", tokenizer=my_tokenizer)
response = client.generate("Hello world")
```

### Mixture-of-Experts (MoE) ModelClient
```python
from multimind.client.model_client import MoEModelClient, LSTMModelClient, RNNModelClient

def router_fn(prompt):
    return "lstm" if len(prompt) < 100 else "rnn"

experts = {"lstm": LSTMModelClient(...), "rnn": RNNModelClient(...)}
moe_client = MoEModelClient(expert_clients=experts, router_fn=router_fn)
response = moe_client.generate("Test input")
```

### DynamicMoEModelClient Example
```python
from multimind.client.model_client import DynamicMoEModelClient, LSTMModelClient, RNNModelClient

def dynamic_router(prompt, metrics):
    if metrics["input_length"] > 1000:
        return "rnn"
    return "lstm"

experts = {"lstm": LSTMModelClient(...), "rnn": RNNModelClient(...)}
dyn_moe_client = DynamicMoEModelClient(expert_clients=experts, router_fn=dynamic_router)
response = dyn_moe_client.generate("Test input")
```

### MultiModalClient Example
```python
from multimind.client.model_client import MultiModalClient, LSTMModelClient, ImageModelClient

client = MultiModalClient(text_client=LSTMModelClient(...), image_client=ImageModelClient())
text_response = client.generate("Hello world", input_type="text")
image_response = client.generate("Describe this image", input_type="image")
```

### FederatedRouter Example
```python
from multimind.client.federated_router import FederatedRouter
from multimind.client.model_client import LSTMModelClient

local_client = LSTMModelClient(...)
cloud_client = LSTMModelClient(...)
router = FederatedRouter(local_client, cloud_client)
response = router.generate("Test input")
```