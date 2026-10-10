"""SQLAlchemy models (SQLite for the MVP; schema is portable to PostgreSQL/PostGIS)."""
import datetime as dt

from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text,
                        create_engine)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from ..core.config import settings

connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
Base = declarative_base()


def utcnow():
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    password_hash = Column(String(256), nullable=False)
    role = Column(String(16), nullable=False)
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=utcnow)


class Mission(Base):
    __tablename__ = "missions"
    id = Column(Integer, primary_key=True)
    code = Column(String(32), unique=True, index=True)
    name = Column(String(128), nullable=False)
    description = Column(Text, default="")
    status = Column(String(16), default="created")  # created|processing|completed|failed
    start_lat = Column(Float, default=24.8607)
    start_lon = Column(Float, default=67.0011)
    altitude = Column(Float, default=60.0)
    heading = Column(Float, default=0.0)
    path_mode = Column(String(16), default="hover")  # hover|linear
    hfov_deg = Column(Float, default=84.0)
    seed = Column(Integer, default=42)
    created_by = Column(String(64))
    created_at = Column(DateTime, default=utcnow)
    videos = relationship("Video", back_populates="mission", cascade="all, delete-orphan")


class Video(Base):
    __tablename__ = "videos"
    id = Column(Integer, primary_key=True)
    code = Column(String(32), unique=True, index=True)
    mission_id = Column(Integer, ForeignKey("missions.id"))
    filename = Column(String(256))
    path = Column(String(512))
    annotated_path = Column(String(512))
    status = Column(String(16), default="uploaded")  # uploaded|processing|done|failed
    progress = Column(Float, default=0.0)
    error = Column(Text)
    fps = Column(Float)
    width = Column(Integer)
    height = Column(Integer)
    frame_count = Column(Integer)
    meta = Column(JSON, default=dict)  # processing stats + dataset metadata
    created_at = Column(DateTime, default=utcnow)
    mission = relationship("Mission", back_populates="videos")


class Zone(Base):
    __tablename__ = "zones"
    id = Column(Integer, primary_key=True)
    name = Column(String(128), nullable=False)
    zone_type = Column(String(24), default="restricted")  # restricted|inspection|emergency|construction|monitoring|custom
    geojson = Column(JSON, nullable=False)  # Polygon geometry
    forbidden_classes = Column(JSON, default=list)  # empty => all classes alert
    max_dwell_s = Column(Float)  # loitering limit for this zone
    active = Column(Boolean, default=True)
    created_by = Column(String(64))
    created_at = Column(DateTime, default=utcnow)


class Detection(Base):
    __tablename__ = "detections"
    id = Column(Integer, primary_key=True)
    video_id = Column(Integer, ForeignKey("videos.id"), index=True)
    mission_id = Column(Integer, index=True)
    frame_id = Column(Integer)
    timestamp = Column(Float)  # seconds from video start
    track_id = Column(Integer, index=True)
    object_class = Column(String(24))
    confidence = Column(Float)
    bbox = Column(JSON)  # [x1,y1,x2,y2] in processed-frame pixels
    lat = Column(Float)
    lon = Column(Float)


class Track(Base):
    __tablename__ = "tracks"
    id = Column(Integer, primary_key=True)
    video_id = Column(Integer, ForeignKey("videos.id"), index=True)
    mission_id = Column(Integer, index=True)
    track_id = Column(Integer)
    object_class = Column(String(24))
    first_seen = Column(Float)
    last_seen = Column(Float)
    n_points = Column(Integer)
    avg_conf = Column(Float)
    trajectory = Column(JSON)  # [[t, lat, lon], ...]
    stats = Column(JSON, default=dict)  # speed, dwell, stationary, zones


class Event(Base):
    __tablename__ = "events"
    id = Column(Integer, primary_key=True)
    video_id = Column(Integer, ForeignKey("videos.id"), index=True)
    mission_id = Column(Integer, index=True)
    incident_id = Column(Integer, ForeignKey("incidents.id"), nullable=True, index=True)
    event_type = Column(String(32), index=True)
    track_id = Column(Integer)
    related_track_id = Column(Integer)
    zone_id = Column(Integer)
    start_t = Column(Float)
    end_t = Column(Float)
    lat = Column(Float)
    lon = Column(Float)
    details = Column(JSON, default=dict)
    created_at = Column(DateTime, default=utcnow)


class Incident(Base):
    __tablename__ = "incidents"
    id = Column(Integer, primary_key=True)
    code = Column(String(24), unique=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), index=True)
    mission_id = Column(Integer, index=True)
    track_id = Column(Integer)
    object_class = Column(String(24))
    primary_type = Column(String(32))
    risk_score = Column(Integer)
    severity = Column(String(12), index=True)
    reasons = Column(JSON, default=list)
    recommendation = Column(Text)
    actions = Column(JSON, default=list)
    requires_review = Column(Boolean, default=True)
    status = Column(String(16), default="pending")  # pending|confirmed|dismissed|investigating
    start_t = Column(Float)
    end_t = Column(Float)
    lat = Column(Float)
    lon = Column(Float)
    zone_id = Column(Integer)
    evidence = Column(JSON, default=list)
    key_frame = Column(String(512))
    vlm = Column(JSON)  # {provider, model, summary, concern_level, rationale}
    model_versions = Column(JSON, default=dict)
    created_at = Column(DateTime, default=utcnow)


class Review(Base):
    __tablename__ = "reviews"
    id = Column(Integer, primary_key=True)
    incident_id = Column(Integer, ForeignKey("incidents.id"), index=True)
    operator = Column(String(64))
    decision = Column(String(16))  # confirmed|dismissed|investigating
    comment = Column(Text, default="")
    created_at = Column(DateTime, default=utcnow)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True)
    user = Column(String(64))
    action = Column(String(64))
    resource = Column(String(128))
    previous_state = Column(JSON)
    new_state = Column(JSON)
    ip = Column(String(64))
    created_at = Column(DateTime, default=utcnow, index=True)


class ModelRecord(Base):
    __tablename__ = "model_registry"
    id = Column(Integer, primary_key=True)
    name = Column(String(64))
    version = Column(String(32))
    task = Column(String(64))
    dataset = Column(String(128))
    metrics = Column(JSON, default=dict)
    status = Column(String(24), default="staging")
    created_at = Column(DateTime, default=utcnow)


class Report(Base):
    __tablename__ = "reports"
    id = Column(Integer, primary_key=True)
    mission_id = Column(Integer, index=True)
    incident_id = Column(Integer, nullable=True)
    kind = Column(String(16))  # mission|incident
    path_json = Column(String(512))
    path_csv = Column(String(512))
    path_pdf = Column(String(512))
    summary = Column(Text)
    created_by = Column(String(64))
    created_at = Column(DateTime, default=utcnow)


def init_db():
    settings.ensure_dirs()
    Base.metadata.create_all(engine)
