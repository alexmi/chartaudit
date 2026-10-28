"""Everything under data/public ships in the public repo: it must be real images, each credited."""
from pathlib import Path

PUBLIC = Path(__file__).resolve().parents[1] / "data" / "public"
MAGIC = (b"\x89PNG", b"\xff\xd8\xff", b"GIF8")


def _images():
    return [p for p in PUBLIC.rglob("*") if p.is_file() and p.name != "CREDITS.md"]


def test_public_files_are_real_images():
    bad = [p.name for p in _images() if not p.read_bytes().startswith(MAGIC)]
    assert not bad, f"not images (error pages?): {bad}"


def test_every_image_is_credited_and_every_credit_exists():
    credits = (PUBLIC / "CREDITS.md").read_text()
    files = {str(p.relative_to(PUBLIC)) for p in _images()}
    listed = {line.split("`")[1] for line in credits.splitlines() if line.startswith("| `")}
    assert files == listed
