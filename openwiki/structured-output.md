---
type: "Reference"
title: "Structured Output and Response Formatting"
description: "Constraining model outputs to structured formats via with_structured_output, schema conversion, provider-specific patterns, and output parsers with validation and streaming support."
tags: [structured-output, response-format, schema, validation, output-parsing, streaming]
verified:
  - by: openwiki/0.5.0
    at: 2026-10-11T08:26:15.994Z
sources:
  - id: openwiki-source-132f3183693cd9cf79d029a5
    resource: repo://libs/core/langchain_core/language_models/base.py
  - id: openwiki-source-c52037e7b642f7ac5a7642a8
    resource: repo://libs/core/langchain_core/language_models/chat_models.py
  - id: openwiki-source-b6969337572a90d4b6262e7b
    resource: repo://libs/core/langchain_core/output_parsers/json.py
  - id: openwiki-source-d89c127dbba78772dc5369ed
    resource: repo://libs/core/langchain_core/output_parsers/openai_tools.py
  - id: openwiki-source-50275a07684ed251c972e512
    resource: repo://libs/core/langchain_core/output_parsers/pydantic.py
  - id: openwiki-source-716439ad921db0ba7cb11ebf
    resource: repo://libs/core/langchain_core/utils/function_calling.py
  - id: openwiki-source-71e882e1ac9757ea8e959a7c
    resource: repo://libs/langchain_v1/langchain/agents/factory.py
  - id: openwiki-source-ec30ab6256dd50cc670919f6
    resource: repo://libs/langchain_v1/langchain/agents/structured_output.py
  - id: openwiki-source-738512768ef81ae009b097ac
    resource: repo://libs/partners/openai/langchain_openai/chat_models/base.py
generated: { by: "openwiki/0.5.0", at: "2026-10-11T08:26:15.994Z" }
---

## Overview

Structured output is the mechanism that constrains language model responses to match a specific schema. Rather than receiving unparsed text, applications can ensure that model outputs conform to Pydantic models, dataclasses, TypedDicts, or raw JSON schemas using `with_structured_output()` on chat models. The implementation supports multiple binding strategies—function calling (tool-based), provider-native APIs, and JSON mode—each with different tradeoffs around model compatibility, validation, and error recovery.

### Core Concept

When a model's `with_structured_output(schema)` method is called, it returns a `Runnable` that:

1. **Binds** the schema to the model using a strategy (tool calling, provider API, or JSON mode)
2. **Invokes** the model with the schema constraint
3. **Parses** the model's response against the schema using validation
4. **Returns** either the parsed structured object or a dict with raw/parsed/error fields

This is distinct from tool calling—structured output constrains the response itself, not the model's tool invocations.

## with_structured_output() Pattern

The `with_structured_output()` method is available on all chat models implementing the `BaseChatModel` interface.

### Method Signature

```python
def with_structured_output(
    self,
    schema: dict[str, Any] | type,
    *,
    include_raw: bool = False,
    **kwargs: Any,
) -> Runnable[LanguageModelInput, dict[str, Any] | BaseModel]
```

### Core Parameters

- **schema**: The output schema. Can be:
  - A Pydantic `BaseModel` class
  - A `dataclass` class
  - A `TypedDict` class
  - A raw JSON Schema dict
  - An OpenAI function/tool schema dict

- **include_raw**: If `False`, only the parsed structured output is returned. If an error occurs during parsing, it is raised. If `True`, returns a dict with keys `'raw'` (the `BaseMessage`), `'parsed'` (the parsed result or `None`), and `'parsing_error'` (the exception or `None`).

### Return Value

- If `schema` is a Pydantic class and `include_raw=False`: Returns a `Runnable` that outputs an instance of `schema`
- If `include_raw=False` with other schema types: Returns a dict with the parsed output
- If `include_raw=True`: Returns a dict with keys `'raw'`, `'parsed'`, and `'parsing_error'`

## Binding Strategies

The actual mechanism for constraining output depends on the chat model implementation. `BaseChatModel` provides a default implementation using tool calling, but provider-specific implementations override it with native APIs when available.

### Tool-Based Strategy (Default)

When `with_structured_output()` is not overridden by a provider, it uses tool calling:

```python
# Default implementation in BaseChatModel
llm = self.bind_tools(
    [schema],
    tool_choice="any",
    ls_structured_output_format={
        "kwargs": {"method": "function_calling"},
        "schema": schema,
    },
)
```

**Lifecycle:**
1. Schema is wrapped as a tool via `bind_tools()`
2. Model is forced to call this tool with `tool_choice="any"`
3. Tool call arguments are extracted and parsed using `PydanticToolsParser` (for Pydantic schemas) or `JsonOutputKeyToolsParser` (for other types)
4. Parsed result is returned

**Advantages:**
- Works with all models supporting tool calling
- Full validation available for Pydantic/TypedDict/dataclass schemas
- Schema can be any JSON-serializable type

**Limitations:**
- Requires tool calling capability
- Consumes one tool slot on the model
- Raw JSON schema dicts are not validated (returned as-is)

### Provider-Native Strategy

Provider-specific implementations (e.g., `ChatOpenAI`) override `with_structured_output()` with native APIs:

```python
# Example: ChatOpenAI with method='json_schema'
response_format = _convert_to_openai_response_format(schema, strict=strict)
llm = self.bind(
    response_format=response_format,
    ls_structured_output_format={
        "kwargs": {"method": "json_schema", "strict": strict},
        "schema": convert_to_openai_tool(schema),
    },
)
```

**Lifecycle:**
1. Schema is converted to provider format (e.g., OpenAI's JSON Schema API)
2. Passed to model via `bind()` with provider-specific parameters
3. Model returns JSON guaranteed to match schema
4. Response is parsed and validated

**Provider Methods (ChatOpenAI example):**

| Method | Provider API | Validation | Strict Mode | Notes |
|--------|--------------|-----------|-------------|-------|
| `'function_calling'` | OpenAI tool calling | Full (via parser) | Supported | Default; widest compatibility |
| `'json_schema'` | OpenAI Structured Outputs API | Provider-enforced | Supported | Stricter guarantees; GPT-4o/GPT-4 Turbo+ only |
| `'json_mode'` | OpenAI JSON Mode | Partial (user provides format) | Not supported | No schema enforcement; requires format instructions |

**Advantages:**
- Provider enforces schema at generation time
- No tool slot consumption
- Works alongside tool calling
- Strict mode available (guarnteed schema compliance)

**Limitations:**
- Only available on provider-specific implementations
- Model compatibility varies by provider
- No in-model error recovery (provider errors are final)

## Schema Types and Conversion

The `with_structured_output()` method normalizes schema types via conversion utilities in `langchain_core.utils.function_calling`:

### Supported Schema Types

| Type | Example | Validation | Parser |
|------|---------|-----------|--------|
| **Pydantic model** | `class Response(BaseModel): ...` | Full via Pydantic | `PydanticToolsParser` |
| **dataclass** | `@dataclass class Response: ...` | Full via Pydantic conversion | `PydanticToolsParser` |
| **TypedDict** | `class Response(TypedDict): ...` | Full via conversion | `PydanticToolsParser` |
| **JSON Schema dict** | `{"type": "object", "properties": {...}}` | None (returned as-is) | `JsonOutputKeyToolsParser` |

### Schema Conversion Functions

**`convert_to_openai_tool(schema, strict=None)`** — Converts schema to OpenAI function-calling format:

```python
from langchain_core.utils.function_calling import convert_to_openai_tool

schema = {
    "type": "function",
    "function": {
        "name": "Response",
        "description": "...",
        "parameters": {
            "type": "object",
            "properties": {...},
            "required": [...]
        }
    }
}
```

**`convert_to_json_schema(schema, strict=None)`** — Converts schema to JSON Schema format:

```python
from langchain_core.utils.function_calling import convert_to_json_schema

json_schema = {
    "type": "object",
    "properties": {...},
    "required": [...],
    "title": "Response"
}
```

## Output Parsers

Output parsers transform model responses into structured objects. The parser used depends on the schema type and strategy:

### PydanticToolsParser

Parses tool calls for Pydantic models:

```python
from langchain_core.output_parsers.openai_tools import PydanticToolsParser

parser = PydanticToolsParser(
    tools=[MyPydanticModel],
    first_tool_only=True  # Return only first tool call
)

# Input: AIMessage with tool_calls
# Output: MyPydanticModel instance
parsed = parser.parse_result([generation])
```

**Behavior:**
- Extracts tool call arguments from `AIMessage.tool_calls`
- Validates arguments against Pydantic model schema
- Raises `OutputParserException` on validation failure
- Returns first tool call result if `first_tool_only=True`

### JsonOutputKeyToolsParser

Parses tool calls for dict/JSON schema:

```python
from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser

parser = JsonOutputKeyToolsParser(
    key_name="MyTool",
    first_tool_only=True
)

# Input: AIMessage with tool_calls
# Output: dict with parsed arguments
```

### PydanticOutputParser

Parses raw JSON text for Pydantic models:

```python
from langchain_core.output_parsers.pydantic import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=MyPydanticModel)

# Input: JSON text
# Output: MyPydanticModel instance
parsed = parser.parse(json_text)
```

### JsonOutputParser

Parses raw JSON text:

```python
from langchain_core.output_parsers.json import JsonOutputParser

parser = JsonOutputParser(pydantic_object=MyPydanticModel)  # optional

# Input: JSON text
# Output: dict or MyPydanticModel instance (if pydantic_object specified)
parsed = parser.parse(json_text)
```

**Streaming Support:**
- `parse_result(result, partial=True)` returns partial JSON objects as they stream
- Useful for progressive rendering of structured output

## Error Handling and Validation

### Output Parser Exceptions

Parsers raise `OutputParserException` when parsing fails:

```python
from langchain_core.exceptions import OutputParserException

try:
    parsed = parser.parse(text)
except OutputParserException as e:
    print(f"Parsing error: {e}")
```

Error details include:
- The parser that failed
- The input that caused the error
- The underlying validation error (if applicable)

### Include Raw Pattern for Error Recovery

Using `include_raw=True` allows applications to handle errors gracefully:

```python
structured_model = model.with_structured_output(
    MySchema,
    include_raw=True
)

result = structured_model.invoke(prompt)
# result = {
#     'raw': AIMessage(...),
#     'parsed': MySchema(...) | None,
#     'parsing_error': Exception | None
# }

if result['parsing_error']:
    print(f"Error: {result['parsing_error']}")
    # Fall back to raw model response
    text = result['raw'].content
else:
    structured_output = result['parsed']
```

### Validation Limitations

**Tool-based strategy (default):**
- Pydantic/dataclass/TypedDict schemas get full validation
- Raw JSON schema dicts skip validation (returned as-is from tool arguments)
- To enable validation, express schemas as Pydantic models or TypedDicts

**Provider-native strategy:**
- Provider enforces schema at generation time; model output is guaranteed valid JSON
- Schema validation happens server-side; no retry mechanism on validation failure
- Strict mode (when supported) guarantees exact schema compliance

## Streaming Structured Output

Structured output can be streamed for progressive parsing, though streaming behavior varies by strategy and parser.

### Partial JSON Parsing

`JsonOutputParser` and `PydanticOutputParser` support partial parsing during streaming:

```python
parser = JsonOutputParser(pydantic_object=MySchema)

# Streaming with partial parsing
for chunk in parser.stream(text_stream):
    if chunk:
        print(f"Partial: {chunk}")
        # chunk is a partial object with received fields so far
```

### Tool Call Streaming

When using tool-calling strategy, tool arguments stream incrementally:

```python
# Stream events from a structured model
for event in structured_model.stream_events(
    prompt,
    version="v3"
):
    if event["event"] == "on_parser_stream":
        print(f"Partial parse: {event['data']}")
```

### Provider Streaming

Provider-native implementations may stream JSON incrementally. Streaming behavior depends on the provider's API:

```python
# OpenAI JSON Schema API with streaming
structured_model = model.with_structured_output(
    MySchema,
    method="json_schema"
)

# Model streams JSON as it generates
for chunk in structured_model.stream(prompt):
    if chunk:
        print(f"Streamed output: {chunk}")
```

## Provider-Specific Implementations

### ChatOpenAI

`ChatOpenAI` provides three methods:

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")

# Method 1: Function calling (default)
structured_model = model.with_structured_output(
    MySchema,
    method="function_calling"  # Default
)

# Method 2: JSON Schema API (strict enforcement)
structured_model = model.with_structured_output(
    MySchema,
    method="json_schema",
    strict=True
)

# Method 3: JSON Mode (user-directed format)
structured_model = model.with_structured_output(
    MySchema,
    method="json_mode"
    # Note: Must include format instructions in prompt
)
```

**Method Selection Logic:**
- `method="json_schema"`: Supported on GPT-4o and GPT-4 Turbo+ models. Provides strict schema enforcement
- `method="function_calling"`: Works on all models with tool calling
- `method="json_mode"`: Works on models with JSON mode support; no schema validation

### Tool-Based Binding for Middleware

Both default and provider implementations set `ls_structured_output_format` metadata for middleware:

```python
llm = model.bind(...)  # or bind_tools(...)
# The bound kwargs include:
# {
#     "ls_structured_output_format": {
#         "kwargs": {"method": "..."},
#         "schema": schema_object
#     }
# }
```

This metadata allows middleware to:
- Observe structured output usage
- Transform or validate schemas
- Change parsing strategy
- Track schema compliance

## Examples

### Basic Pydantic Schema

```python
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

class WeatherResponse(BaseModel):
    """Current weather information."""
    location: str
    temperature_f: int
    condition: str
    humidity_percent: int

model = ChatOpenAI(model="gpt-4o")
structured_model = model.with_structured_output(WeatherResponse)

result = structured_model.invoke("What's the weather in San Francisco?")
# result is a WeatherResponse instance
print(f"Temp: {result.temperature_f}°F, Condition: {result.condition}")
```

### TypedDict Schema

```python
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI

class WeatherResponse(TypedDict):
    """Current weather information."""
    location: str
    temperature_f: int
    condition: str

model = ChatOpenAI(model="gpt-4o")
structured_model = model.with_structured_output(WeatherResponse)

result = structured_model.invoke("What's the weather in NYC?")
# result is a dict
print(result["temperature_f"])
```

### Include Raw for Error Handling

```python
from pydantic import BaseModel

class Answer(BaseModel):
    """Answer with reasoning."""
    answer: str
    reasoning: str

model = ChatOpenAI(model="gpt-4o")
structured_model = model.with_structured_output(Answer, include_raw=True)

result = structured_model.invoke("What is 2+2?")
# result = {
#     'raw': AIMessage(...),
#     'parsed': Answer(...) | None,
#     'parsing_error': Exception | None
# }

if result['parsing_error']:
    # Fallback to raw response
    raw_text = result['raw'].content
    print(f"Parsing failed, using raw: {raw_text}")
else:
    print(f"Answer: {result['parsed'].answer}")
    print(f"Reasoning: {result['parsed'].reasoning}")
```

### JSON Schema Dict

```python
from langchain_openai import ChatOpenAI

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["name", "age"]
}

model = ChatOpenAI(model="gpt-4o")
structured_model = model.with_structured_output(schema)

result = structured_model.invoke("Extract person info from: John Smith, age 28")
# result is a dict: {"name": "John Smith", "age": 28}
```

### Streaming Partial Results

```python
from pydantic import BaseModel

class Response(BaseModel):
    text: str
    category: str

model = ChatOpenAI(model="gpt-4o")
structured_model = model.with_structured_output(Response)

# Stream partial results
for chunk in structured_model.stream("Generate a response..."):
    print(f"Chunk: {chunk}")
```

## Relationship to Related Patterns

### vs. Tool Calling

Structured output constrains the response content directly. Tool calling constrains which tools the model invokes. You can use both together in provider-native APIs:

```python
# Model can call tools OR return structured response
structured_model = model.with_structured_output(
    ResponseSchema,
    method="json_schema",
    tools=[search_tool, calculator_tool],  # ChatOpenAI only
    strict=True,
    include_raw=True
)

result = structured_model.invoke(prompt)
if result['raw'].tool_calls:
    # Model chose to call a tool
    pass
else:
    # Model returned structured response
    structured = result['parsed']
```

### vs. Prompting

Prompting (instructing the model to return JSON) is unreliable:

```python
# ❌ Unreliable: depends on model following instructions
prompt = "Return JSON with keys: name, age"
response = model.invoke(prompt)
json.loads(response.content)  # May fail; format not enforced
```

`with_structured_output()` ensures format compliance:

```python
# ✓ Reliable: format guaranteed by binding strategy
structured_model = model.with_structured_output(MySchema)
result = structured_model.invoke(prompt)  # Always MySchema instance or error
```

## See Also

- [Chat Models](/openwiki/chat-models.md): Base model interfaces, binding, tool calling
- [Runnables](/openwiki/runnables.md): `Runnable` interface, chaining, streaming, async
- [Tools](/openwiki/tools.md): Tool definitions, tool calling, schemas
- [LangChain Core Reference](https://python.langchain.com/docs/concepts/structured_output/): Official structured output documentation
