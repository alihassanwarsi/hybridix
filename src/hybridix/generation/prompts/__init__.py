from importlib.resources import files

def load_prompt(filename: str) -> str:
    return (
        files("hybridix.generation.prompts")
        .joinpath(filename)
        .read_text(encoding="utf-8")
        .strip()
    )