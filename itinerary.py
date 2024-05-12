import datetime
from pydantic import BaseModel
from typing import List


class ScheduleItem(BaseModel):
    day: datetime.date
    start_time: datetime.time
    end_time: datetime.time
    duration: int
    location: str
    price: float


class Itinerary(BaseModel):
    name: str
    start_date: datetime.datetime
    end_date: datetime.datetime
    schedule: List[ScheduleItem]
