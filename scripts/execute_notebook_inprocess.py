from __future__ import annotations

import ast
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import sys
import traceback


def main(source: str, destination: str) -> None:
    source_path = Path(source)
    destination_path = Path(destination)
    notebook = json.loads(source_path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    execution_count = 0
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        execution_count += 1
        cell["execution_count"] = execution_count
        buffer = io.StringIO()
        outputs = []
        source_code = "".join(cell.get("source", []))
        try:
            tree = ast.parse(source_code, filename=f"{source_path.name}:cell-{index}")
            trailing = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                trailing = ast.Expression(tree.body.pop().value)
            with redirect_stdout(buffer), redirect_stderr(buffer):
                if tree.body:
                    exec(compile(tree, str(source_path), "exec"), namespace)
                if trailing is not None:
                    value = eval(compile(trailing, str(source_path), "eval"), namespace)
                    if value is not None:
                        outputs.append({
                            "output_type": "execute_result",
                            "execution_count": execution_count,
                            "metadata": {},
                            "data": {"text/plain": repr(value)},
                        })
            text = buffer.getvalue()
            if text:
                outputs.insert(0, {"output_type": "stream", "name": "stdout", "text": text})
        except Exception as error:
            text = buffer.getvalue() + traceback.format_exc()
            outputs.append({
                "output_type": "error",
                "ename": type(error).__name__,
                "evalue": str(error),
                "traceback": text.splitlines(),
            })
            cell["outputs"] = outputs
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            destination_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
            raise RuntimeError(f"Cell {index} failed") from error
        cell["outputs"] = outputs
    notebook.setdefault("metadata", {})["execution_mode"] = "in-process deterministic review (sandbox kernel transport unavailable)"
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(destination_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
