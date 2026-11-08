"""Shared activity log so every agent's reasoning is visible in the dashboard."""
from dataclasses import dataclass, asdict
import pandas as pd

AGENT_ICONS = {
    "Inventory Agent": "📦", "Forecast Agent": "📈", "Risk Agent": "⚠️", "Reorder Agent": "🔁",
    "Vendor Agent": "🏭", "Cost Agent": "💰", "PO Agent": "📄", "Warehouse Agent": "🚚",
    "Anomaly Agent": "🔍", "Orchestrator": "🧠", "Simulator": "⏩",
}


@dataclass
class LogEvent:
    step: int
    agent: str
    level: str          # info | ok | warn | critical | action
    message: str
    warehouse_id: str = ""
    product_id: str = ""


class AgentLog:
    def __init__(self):
        self.events: list[LogEvent] = []

    def add(self, agent, message, level="info", warehouse_id="", product_id=""):
        self.events.append(LogEvent(len(self.events) + 1, agent, level, message, warehouse_id, product_id))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(e) for e in self.events])
