"""
Tiny helper: parse CLI args normally, but if the required positional args
are missing, open Tk file-open dialogs for each so users can double-click
the script instead of typing paths.
"""
import sys
import tkinter as tk
from tkinter import filedialog, messagebox


def parse_or_pick(parser, positional_specs, ask_missing_options=None):
    """
    parser: an argparse.ArgumentParser already populated with add_argument calls.
    positional_specs: list of (arg_name, dialog_title, [(label, "*.ext"), ...])
                      in the same order the parser expects them.
    ask_missing_options: optional list of (arg_name, prompt) for --flags that
                        are required=True; if omitted from CLI, ask via a
                        simple Tk entry dialog.

    Returns the parsed args namespace. Exits the process if the user
    cancels a dialog for a required argument.
    """
    if len(sys.argv) > 1:
        return parser.parse_args()

    root = tk.Tk()
    root.withdraw()

    picked = []
    for name, title, filetypes in positional_specs:
        path = filedialog.askopenfilename(title=title, filetypes=filetypes)
        if not path:
            messagebox.showerror("Cancelled", f"No file chosen for {name}.")
            sys.exit(1)
        picked.append(path)

    extras = []
    for name, prompt in (ask_missing_options or []):
        val = _ask_string(root, prompt)
        if val is None:
            messagebox.showerror("Cancelled", f"No value entered for {name}.")
            sys.exit(1)
        extras += [f"--{name.replace('_','-')}", val]

    root.destroy()
    return parser.parse_args(picked + extras)


def _ask_string(root, prompt):
    from tkinter import simpledialog
    return simpledialog.askstring("Input", prompt, parent=root)
