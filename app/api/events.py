from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.models import Article, User, UserEvent
from app.db.repositories import UserEventRepository, UserRepository
from app.db.session import get_db
from app.monitoring.metrics import FEEDBACK_EVENTS

router = APIRouter(prefix="/events", tags=["events"])

VALID_EVENT_TYPES = ("impression", "click", "dwell", "skip")


class EventRequest(BaseModel):
    user_id: str = Field(description="User UUID or external id; created if unknown")
    article_id: UUID
    model_version: str | None = None
    position: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] | None = None


class EventResponse(BaseModel):
    event_id: UUID
    event_type: str
    user_id: UUID
    article_id: UUID


def _resolve_or_create_user(db: Session, user_id: str) -> User:
    try:
        user = db.get(User, UUID(user_id))
        if user is None:
            raise HTTPException(status_code=404, detail=f"User {user_id!r} not found")
        return user
    except ValueError:
        pass

    repo = UserRepository(db)
    user = repo.get_by_external_id(user_id)
    if user is None:
        # Feedback can arrive for users we have never served before (e.g. new
        # visitors), so create them on first contact.
        user = repo.create(User(external_id=user_id))
    return user


def _record_event(db: Session, event_type: str, payload: EventRequest) -> EventResponse:
    user = _resolve_or_create_user(db, payload.user_id)

    article = db.get(Article, payload.article_id)
    if article is None:
        raise HTTPException(status_code=404, detail=f"Article {payload.article_id} not found")

    event = UserEventRepository(db).create(
        UserEvent(
            user_id=user.id,
            article_id=payload.article_id,
            event_type=event_type,
            position=payload.position,
            model_version=payload.model_version,
            event_metadata=payload.metadata,
        )
    )
    FEEDBACK_EVENTS.labels(event_type=event_type).inc()

    return EventResponse(
        event_id=event.id,
        event_type=event.event_type,
        user_id=event.user_id,
        article_id=event.article_id,
    )


@router.post("/impression", response_model=EventResponse, status_code=201)
def record_impression(
    payload: EventRequest,
    db: Annotated[Session, Depends(get_db)],
) -> EventResponse:
    return _record_event(db, "impression", payload)


@router.post("/click", response_model=EventResponse, status_code=201)
def record_click(
    payload: EventRequest,
    db: Annotated[Session, Depends(get_db)],
) -> EventResponse:
    return _record_event(db, "click", payload)


@router.post("/dwell", response_model=EventResponse, status_code=201)
def record_dwell(
    payload: EventRequest,
    db: Annotated[Session, Depends(get_db)],
) -> EventResponse:
    return _record_event(db, "dwell", payload)


@router.post("/skip", response_model=EventResponse, status_code=201)
def record_skip(
    payload: EventRequest,
    db: Annotated[Session, Depends(get_db)],
) -> EventResponse:
    return _record_event(db, "skip", payload)
