"""Billing and usage metering API endpoints (Phase 16)."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse

from services.billing_service import BillingService

router = APIRouter(prefix="/api/billing", tags=["Billing"])

_billing = BillingService()


@router.get("/usage")
def get_usage_summary(
    tenant_id: str = Query(..., description="Tenant identifier"),
    days: int = Query(30, ge=1, le=365),
):
    """Return aggregated usage counts by operation for the last N days."""
    summary = _billing.get_usage_summary(tenant_id=tenant_id, days=days)
    return {"success": True, **summary}


@router.get("/invoice")
def get_invoice(
    tenant_id: str = Query(...),
    month: str = Query(..., description="Month in YYYY-MM format"),
):
    """Compute and return an invoice for the given month."""
    try:
        year, mon = int(month[:4]), int(month[5:7])
    except (ValueError, IndexError):
        raise HTTPException(status_code=400, detail=f"Invalid month: {month!r}")

    from datetime import timedelta
    start = datetime(year, mon, 1)
    end = datetime(year + 1, 1, 1) - timedelta(seconds=1) if mon == 12 else datetime(year, mon + 1, 1) - timedelta(seconds=1)
    bill = _billing.compute_bill(tenant_id=tenant_id, start_date=start, end_date=end)
    return {
        "success": True,
        "invoice": {
            "tenant_id": bill.tenant_id,
            "period_start": bill.period_start,
            "period_end": bill.period_end,
            "line_items": [asdict(li) for li in bill.line_items],
            "subtotal": bill.subtotal,
            "total_usd": bill.total_usd,
        },
    }


@router.get("/export")
def export_billing(
    tenant_id: str = Query(...),
    month: str = Query(..., description="Month in YYYY-MM format"),
    format: str = Query("csv", description="Export format (csv only)"),
):
    """Export billing data as CSV."""
    if format.lower() != "csv":
        raise HTTPException(status_code=400, detail="Only 'csv' format is supported")
    try:
        csv_data = _billing.export_csv(tenant_id=tenant_id, month=month)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return PlainTextResponse(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="billing_{tenant_id}_{month}.csv"'},
    )
