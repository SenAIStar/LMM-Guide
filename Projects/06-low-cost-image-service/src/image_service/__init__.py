"""Low-cost image generation service control-plane primitives."""

from .admission import AdmissionController, AdmissionPolicy, AdmissionResult
from .backend import BackendOOM, BatchResult, FakeBackend, GeneratedArtifact
from .batching import BatchLimits, MicroBatch, compatibility_key, plan_batches, work_units
from .cache import ContentAddressedCache
from .capacity import CapacityEstimate, estimate_capacity
from .contracts import AdapterRef, GenerationRequest, RequestValidationError
from .metrics import ServiceMetrics
from .scheduler import MicroBatchScheduler
from .service import ImageGenerationService, ServiceResponse

__all__ = [
    "AdapterRef",
    "AdmissionController",
    "AdmissionPolicy",
    "AdmissionResult",
    "BackendOOM",
    "BatchLimits",
    "BatchResult",
    "ContentAddressedCache",
    "CapacityEstimate",
    "FakeBackend",
    "GeneratedArtifact",
    "GenerationRequest",
    "ImageGenerationService",
    "MicroBatch",
    "MicroBatchScheduler",
    "RequestValidationError",
    "ServiceMetrics",
    "ServiceResponse",
    "compatibility_key",
    "estimate_capacity",
    "plan_batches",
    "work_units",
]
