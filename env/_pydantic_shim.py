"""
Minimal pydantic-compatible shim for sandbox testing.
In production (Docker), real pydantic is installed and this is never used.
"""
import json, dataclasses
from typing import Any, Dict, get_type_hints

class _FieldInfo:
    def __init__(self, default=dataclasses.MISSING, **kwargs):
        self.default = default
        self.metadata = kwargs

def Field(default=dataclasses.MISSING, **kwargs):
    return _FieldInfo(default, **kwargs)

def field_validator(*args, **kwargs):
    def decorator(fn): return fn
    return decorator

class ModelMetaclass(type):
    pass

class BaseModel:
    class Config:
        use_enum_values = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def __init__(self, **data):
        hints = {}
        for klass in reversed(type(self).__mro__):
            if hasattr(klass, '__annotations__'):
                hints.update(klass.__annotations__)
        # Apply defaults from class attributes
        for name, hint in hints.items():
            val = data.get(name, dataclasses.MISSING)
            if val is dataclasses.MISSING:
                class_val = getattr(type(self), name, dataclasses.MISSING)
                if isinstance(class_val, _FieldInfo):
                    if class_val.default is not dataclasses.MISSING:
                        val = class_val.default() if callable(class_val.default) else class_val.default
                    else:
                        val = None
                elif class_val is not dataclasses.MISSING and not callable(class_val):
                    val = class_val
                else:
                    val = None
            object.__setattr__(self, name, val)
        # Set any extra data keys
        for k, v in data.items():
            object.__setattr__(self, k, v)

    def model_dump(self) -> Dict[str, Any]:
        result = {}
        hints = {}
        for klass in reversed(type(self).__mro__):
            if hasattr(klass, '__annotations__'):
                hints.update(klass.__annotations__)
        for name in hints:
            val = getattr(self, name, None)
            if hasattr(val, 'model_dump'):
                result[name] = val.model_dump()
            elif isinstance(val, list):
                result[name] = [
                    v.model_dump() if hasattr(v, 'model_dump') else
                    (v.value if hasattr(v, 'value') else v)
                    for v in val
                ]
            elif hasattr(val, 'value'):  # enum
                result[name] = val.value
            else:
                result[name] = val
        return result

    @classmethod
    def model_json_schema(cls) -> Dict[str, Any]:
        return {"type": "object", "title": cls.__name__}
