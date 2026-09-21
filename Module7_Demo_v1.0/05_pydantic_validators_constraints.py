"""
Demo 5: Pydantic validators and constraint patterns.

Goes beyond type checking to enforce business rules on a support ticket:
bounded fields, Literal enums, and a custom @field_validator, all
demonstrated by catching a deliberately bad ticket dict. No model call
needed - this is pure Pydantic.

Setup:
    pip install pydantic
"""

import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

BAD_TICKET = {"ticket_id": "TCK-1", "priority": 8, "issue_type": "urgent"}
GOOD_TICKET = {"ticket_id": "TCK-1001", "priority": 4, "issue_type": "billing"}


class TicketRecord(BaseModel):
    ticket_id: str
    priority: int = Field(ge=1, le=5, description="1 = lowest, 5 = most urgent")
    issue_type: Literal["billing", "technical", "other"]

    @field_validator("ticket_id")
    @classmethod
    def ticket_id_format(cls, v):
        if not re.fullmatch(r"TCK-\d{4}", v):
            raise ValueError("ticket_id must look like 'TCK-1234'")
        return v


def main():
    print(f"Validating: {BAD_TICKET}")
    try:
        TicketRecord.model_validate(BAD_TICKET)
    except ValidationError as exc:
        print("Caught ValidationError:")
        for error in exc.errors():
            print(f"  {error['loc']}: {error['msg']}")

    print(f"\nValidating: {GOOD_TICKET}")
    record = TicketRecord.model_validate(GOOD_TICKET)
    print(f"OK: {record}")


if __name__ == "__main__":
    main()
