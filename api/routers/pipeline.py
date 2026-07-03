"""
/api/v1/pipeline — trigger and inspect pipeline runs.
Protected: requires X-API-Key header in non-dev environments.
"""

from fastapi import APIRouter, BackgroundTasks, HTTPException, Header, Depends
from typing import Optional

from bti.config import get_settings
from bti.logging_config import get_logger
from api.schemas import PipelineRunRequest, PipelineRunResponse

router = APIRouter(prefix="/pipeline", tags=["Pipeline"])
log = get_logger("api.pipeline")
settings = get_settings()

_last_run: Optional[dict] = None


def _verify_api_key(x_api_key: Optional[str] = Header(default=None)):
    if settings.environment != "development" and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")


@router.post("/run", response_model=PipelineRunResponse,
             dependencies=[Depends(_verify_api_key)])
def trigger_pipeline_run(body: PipelineRunRequest, background_tasks: BackgroundTasks):
    """
    Triggers the full pipeline asynchronously.
    Poll GET /pipeline/status for results.
    """
    global _last_run
    from bti.pipeline import PipelineOrchestrator

    def _run_bg():
        global _last_run
        orch = PipelineOrchestrator(force=body.force)
        _last_run = orch.run()

    background_tasks.add_task(_run_bg)
    log.info("Pipeline triggered via API", extra={"force": body.force})
    return PipelineRunResponse(
        run_id="pending",
        status="TRIGGERED",
        stages={},
        elapsed_seconds=None,
    )


@router.post("/run/sync", response_model=PipelineRunResponse,
             dependencies=[Depends(_verify_api_key)])
def trigger_pipeline_sync(body: PipelineRunRequest):
    """Runs the pipeline synchronously — for testing and dev use."""
    from bti.pipeline import PipelineOrchestrator
    result = PipelineOrchestrator(force=body.force).run()
    return PipelineRunResponse(**result)


@router.get("/status", response_model=Optional[PipelineRunResponse])
def get_last_run_status():
    if _last_run is None:
        return None
    return PipelineRunResponse(**_last_run)
