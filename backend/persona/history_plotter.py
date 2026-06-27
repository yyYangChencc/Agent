from __future__ import annotations

import csv
import html
from pathlib import Path
from typing import Iterable


HISTORY_DIR = Path(__file__).resolve().parents[1] / "history"
REQUIRED_COLUMNS = ("tick", "opinion")


def resolve_history_run_dir(history_path: str | Path = HISTORY_DIR) -> Path:
    """Return a run directory containing per-agent CSV history files."""
    path = Path(history_path)
    if not path.exists():
        raise FileNotFoundError(f"History path does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"History path is not a directory: {path}")

    csv_files = sorted(path.glob("*.csv"))
    if csv_files:
        return path

    run_dirs = sorted(
        (child for child in path.iterdir() if child.is_dir()),
        key=lambda child: child.stat().st_mtime,
        reverse=True,
    )
    for run_dir in run_dirs:
        if any(run_dir.glob("*.csv")):
            return run_dir

    raise FileNotFoundError(f"No agent CSV files found under: {path}")


def load_agent_opinion_series(
    history_path: str | Path = HISTORY_DIR,
    agent_ids: Iterable[str] | None = None,
    data_length: int | None = None,
) -> dict[str, list[tuple[int, float]]]:
    """Load each selected agent's ``(tick, opinion)`` series from history CSV files."""
    _validate_data_length(data_length)
    run_dir = resolve_history_run_dir(history_path)
    selected_agent_ids = set(agent_ids) if agent_ids is not None else None
    series_by_agent: dict[str, list[tuple[int, float]]] = {}

    for csv_path in sorted(run_dir.glob("*.csv")):
        agent_id = csv_path.stem
        if selected_agent_ids is not None and agent_id not in selected_agent_ids:
            continue
        series = _read_opinion_series(csv_path)
        if data_length is not None:
            series = series[-data_length:]
        if series:
            series_by_agent[agent_id] = series

    if selected_agent_ids is not None:
        missing_agent_ids = sorted(selected_agent_ids - set(series_by_agent))
        if missing_agent_ids:
            missing = ", ".join(missing_agent_ids)
            raise FileNotFoundError(f"No opinion history found for agent ids: {missing}")

    if not series_by_agent:
        raise ValueError(f"No opinion rows found in history run: {run_dir}")

    return series_by_agent


def plot_agent_opinion_trends(
    history_path: str | Path = HISTORY_DIR,
    output_path: str | Path | None = None,
    agent_ids: Iterable[str] | None = None,
    data_length: int | None = None,
    title: str | None = None,
) -> Path:
    """
    Plot opinion trends for agents recorded under ``backend/history``.

    ``history_path`` can point either to ``backend/history`` or to one concrete
    run directory such as ``backend/history/20260624_185349``.
    ``data_length`` limits each selected agent to its most recent N rows.
    """
    series_by_agent = load_agent_opinion_series(
        history_path,
        agent_ids=agent_ids,
        data_length=data_length,
    )
    run_dir = resolve_history_run_dir(history_path)

    if output_path is None:
        output_path = run_dir / "opinion_trends.svg"
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.suffix.lower() == ".svg":
        _write_svg_opinion_plot(
            output,
            series_by_agent,
            title or f"Agent Opinion Trends - {run_dir.name}",
        )
        return output

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for non-SVG opinion trend plots. "
            "Install project dependencies with: pip install -r requirements.txt"
        ) from exc

    fig, ax = plt.subplots(figsize=(10, 6))
    for agent_id, series in series_by_agent.items():
        ticks = [tick for tick, _ in series]
        opinions = [opinion for _, opinion in series]
        ax.plot(ticks, opinions, marker="o", linewidth=1.8, markersize=3, label=agent_id)

    ax.set_title(title or f"Agent Opinion Trends - {run_dir.name}")
    ax.set_xlabel("tick")
    ax.set_ylabel("opinion")
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0, color="#666666", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(title="agent")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)

    return output


def _write_svg_opinion_plot(
    output_path: Path,
    series_by_agent: dict[str, list[tuple[int, float]]],
    title: str,
) -> None:
    width = 1000
    height = 600
    left = 78
    right = 28
    top = 58
    bottom = 78
    plot_width = width - left - right
    plot_height = height - top - bottom
    colors = (
        "#2563eb",
        "#dc2626",
        "#16a34a",
        "#9333ea",
        "#ea580c",
        "#0891b2",
        "#be123c",
        "#4d7c0f",
    )
    all_ticks = [tick for series in series_by_agent.values() for tick, _ in series]
    min_tick = min(all_ticks)
    max_tick = max(all_ticks)

    def x_pos(tick: int) -> float:
        if min_tick == max_tick:
            return left + plot_width / 2
        return left + (tick - min_tick) * plot_width / (max_tick - min_tick)

    def y_pos(opinion: float) -> float:
        bounded = min(1.0, max(-1.0, opinion))
        return top + (1.0 - bounded) * plot_height / 2

    lines: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        "<style>"
        "text{font-family:Arial,'Microsoft YaHei',sans-serif;fill:#111827;}"
        ".axis{stroke:#374151;stroke-width:1.2;}"
        ".grid{stroke:#d1d5db;stroke-width:0.8;stroke-dasharray:4 4;}"
        ".label{font-size:14px;}"
        ".tick{font-size:12px;fill:#4b5563;}"
        ".title{font-size:22px;font-weight:700;}"
        ".legend{font-size:13px;}"
        "</style>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text class="title" x="{width / 2}" y="34" text-anchor="middle">{html.escape(title)}</text>',
    ]

    for value in (-1.0, -0.5, 0.0, 0.5, 1.0):
        y = y_pos(value)
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}"/>')
        lines.append(f'<text class="tick" x="{left - 12}" y="{y + 4:.2f}" text-anchor="end">{value:.1f}</text>')

    tick_marks = _tick_marks(min_tick, max_tick)
    for tick in tick_marks:
        x = x_pos(tick)
        lines.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}"/>')
        lines.append(f'<text class="tick" x="{x:.2f}" y="{top + plot_height + 24}" text-anchor="middle">{tick}</text>')

    lines.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>',
            (
                f'<line class="axis" x1="{left}" y1="{top + plot_height}" '
                f'x2="{left + plot_width}" y2="{top + plot_height}"/>'
            ),
            f'<text class="label" x="{width / 2}" y="{height - 24}" text-anchor="middle">tick</text>',
            (
                f'<text class="label" x="22" y="{top + plot_height / 2}" '
                'text-anchor="middle" transform="rotate(-90 22 '
                f'{top + plot_height / 2})">opinion</text>'
            ),
        ]
    )

    for index, (agent_id, series) in enumerate(series_by_agent.items()):
        color = colors[index % len(colors)]
        points = " ".join(f"{x_pos(tick):.2f},{y_pos(opinion):.2f}" for tick, opinion in series)
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            'stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for tick, opinion in series:
            lines.append(
                f'<circle cx="{x_pos(tick):.2f}" cy="{y_pos(opinion):.2f}" '
                f'r="2.8" fill="{color}"/>'
            )
        legend_x = left + 18 + (index % 4) * 180
        legend_y = height - 50 + (index // 4) * 18
        lines.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 22}" y2="{legend_y}" stroke="{color}" stroke-width="2.2"/>')
        lines.append(f'<text class="legend" x="{legend_x + 30}" y="{legend_y + 4}">{html.escape(agent_id)}</text>')

    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _tick_marks(min_tick: int, max_tick: int) -> list[int]:
    if min_tick == max_tick:
        return [min_tick]
    span = max_tick - min_tick
    raw_step = max(1, span // 8)
    step = _rounded_step(raw_step)
    first = ((min_tick + step - 1) // step) * step
    marks = [min_tick]
    marks.extend(range(first, max_tick + 1, step))
    if marks[-1] != max_tick:
        marks.append(max_tick)
    return sorted(set(marks))


def _rounded_step(raw_step: int) -> int:
    for step in (1, 2, 5, 10, 20, 50, 100):
        if raw_step <= step:
            return step
    magnitude = 10 ** (len(str(raw_step)) - 1)
    return ((raw_step + magnitude - 1) // magnitude) * magnitude


def _validate_data_length(data_length: int | None) -> None:
    if data_length is None:
        return
    if not isinstance(data_length, int):
        raise TypeError("data_length must be an int or None")
    if data_length < 1:
        raise ValueError("data_length must be greater than 0")


def _read_opinion_series(csv_path: Path) -> list[tuple[int, float]]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(f"{csv_path} is missing required columns: {missing}")

        series: list[tuple[int, float]] = []
        for row_number, row in enumerate(reader, start=2):
            tick_text = row.get("tick", "")
            opinion_text = row.get("opinion", "")
            if tick_text == "" or opinion_text == "":
                continue
            try:
                tick = int(float(tick_text))
                opinion = float(opinion_text)
            except ValueError as exc:
                raise ValueError(
                    f"{csv_path} row {row_number} has invalid tick/opinion values"
                ) from exc
            series.append((tick, opinion))
    return series

