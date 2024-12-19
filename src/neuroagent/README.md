To generate the pydantic models from an API openapi.json spec, run the following command:
```bash
pip install datamodel-code-generator
datamodel-codegen --enum-field-as-literal=all --target-python-version=3.10 --use-annotated --reuse-model  --input-file-type=openapi --url=TARGET_URL/openapi.json --output=OUTPUT --output-model-type=pydantic_v2.BaseModel
```
