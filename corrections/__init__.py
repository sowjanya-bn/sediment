from .schema import CorrectionRecord, CR_TYPES
from .db import save_correction, get_corrections_for_entry
from .apply import apply_correction, get_effective_entry
