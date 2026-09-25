import json
from pathlib import Path

import typer

from .data import prepare_ragtruth
from .detector import Detector
from .training import evaluate, train


app = typer.Typer(no_args_is_help=True)
PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent


@app.command("prepare-data")
def prepare_data(
    dataset_dir: Path = typer.Option(PACKAGE_DIR / "third_party" / "RAGTruth" / "dataset"),
    output_dir: Path = typer.Option(PACKAGE_DIR / "data" / "processed"),
):
    typer.echo(json.dumps(prepare_ragtruth(dataset_dir, output_dir), indent=2))


@app.command()
def train_model(
    data_dir: Path = typer.Option(PACKAGE_DIR / "data" / "processed"),
    output_dir: Path = typer.Option(REPO_ROOT / "artifacts" / "detector-best"),
    base_model: str = typer.Option("microsoft/deberta-v3-xsmall"),
    epochs: int = typer.Option(3),
    batch_size: int = typer.Option(8),
    max_length: int = typer.Option(384),
):
    typer.echo(
        json.dumps(
            train(data_dir, output_dir, base_model, epochs, batch_size, max_length=max_length),
            indent=2,
        )
    )


@app.command()
def evaluate_model(
    data_dir: Path = typer.Option(PACKAGE_DIR / "data" / "processed"),
    model_dir: Path = typer.Option(REPO_ROOT / "artifacts" / "detector-best"),
    batch_size: int = typer.Option(16),
    max_length: int = typer.Option(384),
):
    typer.echo(json.dumps(evaluate(data_dir, model_dir, batch_size, max_length), indent=2))


@app.command()
def predict(
    answer: str = typer.Option(...),
    evidence_file: Path = typer.Option(..., exists=True),
    model_dir: Path = typer.Option(REPO_ROOT / "artifacts" / "detector-best"),
):
    detector = Detector(model_dir)
    result = detector.detect(answer, [evidence_file.read_text(encoding="utf-8")])
    typer.echo(result.model_dump_json(indent=2))


@app.command("predict-text")
def predict_text(
    answer: str = typer.Option(..., help="LLM draft answer to check."),
    evidence: list[str] = typer.Option(..., "--evidence", "-e", help="Evidence passage; repeat for multiple passages."),
    query: str = typer.Option("", help="Original user query."),
    model_dir: Path = typer.Option(REPO_ROOT / "artifacts" / "detector-best"),
):
    """Run grounded sentence detection directly from command-line text."""
    detector = Detector(model_dir)
    result = detector.detect(answer, evidence, user_query=query)
    typer.echo(result.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
