import pytest
from typer.testing import CliRunner

from genescope.cli import app
from genescope.models import available_models, get_model


def test_checkpoint_requires_explicit_code_opt_in_and_immutable_revision():
    assert len(available_models()[0].revision) == 40
    with pytest.raises(ValueError, match="allow-remote-code"):
        get_model("nucleotide-transformer")
    with pytest.raises(ValueError, match="immutable"):
        get_model("nucleotide-transformer", trust_remote_code=True, revision="main")
    with pytest.raises(ValueError, match="1000-token"):
        get_model("nucleotide-transformer", trust_remote_code=True, max_length=1001)
    with pytest.raises(ValueError, match="Unknown model"):
        get_model("invented-backend")


def test_embed_cli_explains_opt_in_without_writing_output(tmp_path):
    fasta = tmp_path / "input.fasta"
    fasta.write_text(">a\nACGT\n")
    output = tmp_path / "embeddings.csv"
    result = CliRunner().invoke(app, ["embed", str(fasta), "-o", str(output)])
    assert result.exit_code == 1
    assert "allow-remote-code" in result.output
    assert not output.exists()
