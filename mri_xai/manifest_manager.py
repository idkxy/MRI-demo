import os
import glob


class ManifestManager:
    def __init__(
        self,
        validation_root="validation_stage1",
        manifest_path="validation_manifest.md",
    ):
        self.validation_root = validation_root
        self.manifest_path = manifest_path

    def create_manifest(self):
        labels = ["AD", "MCI", "CN"]
        rows = []

        for label in labels:
            folder_path = os.path.join(self.validation_root, label)

            if not os.path.exists(folder_path):
                folder_path = os.path.join(self.validation_root, label.lower())

            if not os.path.exists(folder_path):
                continue

            files = sorted(glob.glob(os.path.join(folder_path, "*.npy")))

            for fpath in files:
                file_name = os.path.basename(fpath)
                relative_path = os.path.relpath(fpath, self.validation_root)

                rows.append({
                    "file_name": file_name,
                    "true_label": label,
                    "relative_path": relative_path,
                })

        with open(self.manifest_path, "w", encoding="utf-8") as f:
            f.write("| file_name | true_label | relative_path |\n")
            f.write("|---|---|---|\n")

            for row in rows:
                f.write(
                    f"| {row['file_name']} | "
                    f"{row['true_label']} | "
                    f"{row['relative_path']} |\n"
                )

        return rows

    def load_manifest(self):
        if not os.path.exists(self.manifest_path):
            return []

        rows = []

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()

            if not line.startswith("|"):
                continue

            if "---" in line:
                continue

            parts = [p.strip() for p in line.strip("|").split("|")]

            if len(parts) != 3:
                continue

            if parts[0] == "file_name":
                continue

            rows.append({
                "file_name": parts[0],
                "true_label": parts[1],
                "relative_path": parts[2],
            })

        return rows

    def find_label_by_filename(self, uploaded_file_name):
        rows = self.load_manifest()

        matches = [
            row for row in rows
            if row["file_name"] == uploaded_file_name
        ]

        if len(matches) == 1:
            return matches[0]["true_label"], matches

        return None, matches

    def ensure_manifest_exists(self):
        if os.path.exists(self.validation_root) and not os.path.exists(self.manifest_path):
            return self.create_manifest()

        return self.load_manifest()