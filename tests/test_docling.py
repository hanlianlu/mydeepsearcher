import os
import warnings
import logging
from threading import Thread
import queue
import tkinter as tk
from tkinter import filedialog, scrolledtext

# suppress the docling deprecation warning
warnings.filterwarnings("ignore", ".*strict_text.*")

from deepsearcher.loader.file_loader.docling_loader import DoclingLoader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class DoclingTester:
    def __init__(self, root):
        self.root = root
        self.queue = queue.Queue()
        self.loader = DoclingLoader()

        root.title("Docling Loader Tester")
        root.geometry("700x450")

        btn_frame = tk.Frame(root); btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="Select Files", command=self.select_files).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Clear Results", command=self.clear_results).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Quit", command=root.quit).pack(side=tk.LEFT, padx=5)

        self.results_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=20)
        self.results_text.pack(padx=10, pady=5, expand=True, fill=tk.BOTH)
        self.status_label = tk.Label(root, text="Ready", anchor="w")
        self.status_label.pack(fill=tk.X, padx=10, pady=5)

        self.check_queue()

    def select_files(self):
        file_paths = filedialog.askopenfilenames(
            filetypes=[("Supported", "*.pdf *.docx *.png *.jpg *.jpeg *.bmp *.tiff"), ("All files", "*.*")]
        )
        if not file_paths:
            return
        self.status_label.config(text=f"Parsing {len(file_paths)} file(s)…")
        self.results_text.delete("1.0", tk.END)
        Thread(target=self.parse_files, args=(file_paths,), daemon=True).start()

    def parse_files(self, file_paths):
        total = len(file_paths)
        for i, path in enumerate(file_paths, 1):
            fname = os.path.basename(path)
            try:
                docs = self.loader.load_file(path)
                ocr_used = any(d.metadata.get("ocr_used") for d in docs)
                if docs:
                    preview = docs[0].page_content[:100].replace("\n", " ")
                    msg = (
                        f"File: {fname}\n"
                        f"OCR Used: {ocr_used}\n"
                        f"Documents: {len(docs)}\n"
                        f"Preview: {preview} …\n"
                        + "-"*60 + "\n"
                    )
                else:
                    msg = (
                        f"File: {fname}\n"
                        f"OCR Used: {ocr_used}\n"
                        "No documents extracted\n"
                        + "-"*60 + "\n"
                    )
            except Exception as e:
                msg = f"File: {fname}\nError: {e}\n" + "-"*60 + "\n"

            self.queue.put(("result", msg))
            self.queue.put(("progress", f"Processed {i}/{total} files…"))

        self.queue.put(("done", "Parsing completed"))

    def check_queue(self):
        try:
            while True:
                typ, content = self.queue.get_nowait()
                if typ == "result":
                    self.results_text.insert(tk.END, content)
                    self.results_text.see(tk.END)
                elif typ == "progress":
                    self.status_label.config(text=content)
                else:
                    self.status_label.config(text=content)
        except queue.Empty:
            pass
        self.root.after(100, self.check_queue)

    def clear_results(self):
        self.results_text.delete("1.0", tk.END)
        self.status_label.config(text="Ready")


if __name__ == "__main__":
    root = tk.Tk()
    DoclingTester(root)
    root.mainloop()
